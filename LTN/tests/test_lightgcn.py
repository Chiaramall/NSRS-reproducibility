import numpy as np
import pytest
import torch
from unittest import mock

from movielens_paper_benchmark.models.lightgcn import LightGCN


TINY_INTERACTIONS = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 2],
        [2, 2],
        [2, 3],
    ],
    dtype=np.int64,
)


def _tiny_model(seed=7):
    return LightGCN(
        n_users=3,
        n_items=5,
        embedding_dim=4,
        num_layers=2,
        learning_rate=0.02,
        l2_reg=1e-4,
        batch_size=3,
        epochs=3,
        seed=seed,
        device="cpu",
    )


def test_normalized_adjacency_is_symmetric_and_has_expected_shape():
    model = _tiny_model()
    adjacency = model.build_graph(TINY_INTERACTIONS).to_dense()

    assert adjacency.shape == (8, 8)
    assert torch.allclose(adjacency, adjacency.t())
    # User 0 and item 0 have degrees 2 and 1, respectively.
    assert adjacency[0, 3].item() == pytest.approx(1 / np.sqrt(2))
    assert torch.count_nonzero(adjacency).item() == 2 * len(TINY_INTERACTIONS)


def test_target_column_excludes_nonpositive_feedback():
    interactions = np.array([[0, 0, 1], [0, 2, 0], [1, 1, 1], [1, 3, -1]])
    model = _tiny_model()
    adjacency = model.build_graph(interactions).to_dense()

    assert torch.count_nonzero(adjacency).item() == 4
    assert adjacency[0, 3 + 2].item() == 0
    assert adjacency[1, 3 + 3].item() == 0


def test_fit_is_deterministic_and_never_scores_seen_items_when_masked():
    first = _tiny_model(seed=11)
    second = _tiny_model(seed=11)

    assert first.fit(TINY_INTERACTIONS) is first
    assert second.fit(TINY_INTERACTIONS) is second
    first_scores = first.score_users(
        [0, 1, 2], exclude_interactions=TINY_INTERACTIONS
    )
    second_scores = second.score_users(
        [0, 1, 2], exclude_interactions=TINY_INTERACTIONS
    )

    assert first.history_["loss"] == pytest.approx(second.history_["loss"])
    np.testing.assert_allclose(first_scores, second_scores)
    assert np.isneginf(first_scores[0, 0])
    assert np.isneginf(first_scores[1, 2])
    assert np.isfinite(first_scores[0, 4])


def test_checkpoint_round_trip_preserves_scores(tmp_path):
    model = _tiny_model()
    model.fit(TINY_INTERACTIONS, epochs=1)
    expected = model.score_users([0, 2])
    checkpoint = tmp_path / "lightgcn.pt"

    model.save_checkpoint(checkpoint, extra={"validation_hit_at_10": 0.5})
    restored = LightGCN.load_checkpoint(checkpoint, device="cpu")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert restored.get_config()["embedding_dim"] == 4
    assert payload["checkpoint_purpose"] == "inference_only"
    assert "optimizer_state_dict" not in payload
    assert restored._optimizer is None
    np.testing.assert_allclose(restored.score_users([0, 2]), expected)


def test_bpr_loss_is_finite_and_backpropagates():
    model = _tiny_model()
    model.build_graph(TINY_INTERACTIONS)
    total, ranking, regularization = model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([4, 4]),
    )

    assert torch.isfinite(total)
    assert ranking.item() > 0
    assert regularization.item() > 0
    total.backward()
    assert model.user_embedding.weight.grad is not None


def test_forbidden_items_extend_sampling_exclusions_without_changing_graph():
    model = _tiny_model()
    model.fit(
        TINY_INTERACTIONS,
        epochs=0,
        forbidden_items_by_user={0: [2, 3]},
    )
    generator = torch.Generator().manual_seed(123)
    negatives = model._sample_negatives(torch.tensor([0] * 20), generator)

    assert model._normalized_adj._nnz() == 2 * len(TINY_INTERACTIONS)
    assert model._positive_items[0] == {0, 1}
    assert model._negative_exclusions[0] == {0, 1, 2, 3}
    assert negatives.tolist() == [4] * 20


@pytest.mark.parametrize(
    "forbidden,error",
    [
        ({3: [0]}, IndexError),
        ({0: [5]}, IndexError),
        ({0.5: [1]}, TypeError),
        ({0: [1.5]}, TypeError),
    ],
)
def test_forbidden_items_are_validated(forbidden, error):
    with pytest.raises(error):
        _tiny_model().fit(
            TINY_INTERACTIONS,
            epochs=0,
            forbidden_items_by_user=forbidden,
        )


def test_eval_cache_reuses_propagation_and_invalidates_on_graph_and_state_load():
    model = _tiny_model()
    model.build_graph(TINY_INTERACTIONS)
    model.eval()

    with mock.patch.object(
        model, "_propagate_once", wraps=model._propagate_once
    ) as propagate:
        model.score_users([0])
        model.score_users([1])
        assert propagate.call_count == 1

        model.build_graph(TINY_INTERACTIONS)
        model.score_users([0])
        assert propagate.call_count == 2

        state = {key: value.clone() for key, value in model.state_dict().items()}
        model.load_state_dict(state)
        model.score_users([0])
        assert propagate.call_count == 3


def test_fit_propagates_once_per_epoch_and_once_per_cached_validation():
    model = _tiny_model()

    def validation(fitted):
        fitted.score_users([0])
        fitted.score_users([1])
        return 0.5

    with mock.patch.object(
        model, "_propagate_once", wraps=model._propagate_once
    ) as propagate:
        model.fit(
            TINY_INTERACTIONS,
            epochs=10,
            patience=1,
            validation_callback=validation,
        )

    # Epoch 1 establishes the best score; epoch 2 is the first stale epoch.
    assert len(model.history_["loss"]) == 2
    assert model.best_epoch_ == 0
    assert propagate.call_count == 4


def test_minibatch_accumulation_matches_full_fixed_propagation_gradient():
    full = _tiny_model(seed=19)
    accumulated = _tiny_model(seed=19)
    full.build_graph(TINY_INTERACTIONS)
    accumulated.build_graph(TINY_INTERACTIONS)
    full.train()
    accumulated.train()
    users = torch.tensor([0, 0, 1, 1, 2, 2])
    positives = torch.tensor([0, 1, 1, 2, 2, 3])
    negatives = torch.tensor([4, 3, 4, 0, 1, 0])

    full_loss, _, _ = full.bpr_loss(users, positives, negatives)
    full_loss.backward()

    user_vectors, item_vectors = accumulated.propagated_embeddings()
    for start in range(0, len(users), 2):
        end = start + 2
        batch_loss, _, _ = accumulated._bpr_loss_from_embeddings(
            users[start:end],
            positives[start:end],
            negatives[start:end],
            user_vectors,
            item_vectors,
        )
        (batch_loss * ((end - start) / len(users))).backward(
            retain_graph=end < len(users)
        )

    torch.testing.assert_close(
        accumulated.user_embedding.weight.grad,
        full.user_embedding.weight.grad,
    )
    torch.testing.assert_close(
        accumulated.item_embedding.weight.grad,
        full.item_embedding.weight.grad,
    )
