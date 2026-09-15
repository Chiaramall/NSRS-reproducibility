import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import movielens_paper_benchmark.models.mf as mf_module
from movielens_paper_benchmark.models import EASE, MatrixFactorization


class MatrixFactorizationTest(unittest.TestCase):
    def setUp(self):
        self.pointwise = np.asarray(
            [
                [0, 0, 1], [0, 1, 0], [0, 2, 0],
                [1, 0, 0], [1, 1, 1], [1, 2, 0],
                [2, 0, 0], [2, 1, 0], [2, 2, 1],
            ],
            dtype=np.float32,
        )
        self.positives = self.pointwise[self.pointwise[:, 2] > 0, :2].astype(np.int64)

    def test_mse_fit_scoring_and_checkpoint(self):
        model = MatrixFactorization(
            3, 3, embedding_dim=256, objective="mse", biased=True, seed=7
        )
        model.fit(self.pointwise, epochs=80, batch_size=9, learning_rate=0.03)
        self.assertLess(model.history_["loss"][-1], model.history_["loss"][0])
        scores = model.score_users([0, 1], batch_size=1)
        self.assertEqual(scores.shape, (2, 3))
        paired = model.predict([0, 1], [0, 1])
        np.testing.assert_allclose(
            paired, [scores[0, 0], scores[1, 1]], rtol=1e-6, atol=1e-7
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "mf.pt"
            model.save(checkpoint)
            restored = MatrixFactorization.load(checkpoint)
            np.testing.assert_allclose(restored.score_users([0, 1]), scores, atol=1e-7)

    def test_bpr_is_deterministic_and_ranks_training_positive(self):
        kwargs = dict(
            n_users=3,
            n_items=3,
            embedding_dim=16,
            objective="bpr",
            biased=False,
            seed=11,
        )
        first = MatrixFactorization(**kwargs).fit(
            self.positives, epochs=100, batch_size=3, learning_rate=0.03
        )
        second = MatrixFactorization(**kwargs).fit(
            self.positives, epochs=100, batch_size=3, learning_rate=0.03
        )
        first_scores = first.score_users([0, 1, 2])
        np.testing.assert_allclose(first_scores, second.score_users([0, 1, 2]), atol=0)
        for user in range(3):
            self.assertEqual(int(first_scores[user].argmax()), user)

    def _assert_forbidden_items_are_not_sampled(self, objective, num_negatives):
        positives = np.asarray([[0, 0], [1, 1]], dtype=np.int64)
        forbidden = {0: {2, 3}, 1: {0, 2}}
        expected_only_candidate = {0: 1, 1: 3}
        sampled = []
        original_sampler = mf_module.sample_unobserved_items

        def capture(users, seen_items, n_items, rng):
            kept_users, negative_items = original_sampler(
                users, seen_items, n_items, rng
            )
            sampled.extend(zip(kept_users.tolist(), negative_items.tolist()))
            return kept_users, negative_items

        model = MatrixFactorization(
            2, 4, embedding_dim=4, objective=objective, biased=False, seed=19
        )
        with mock.patch.object(
            mf_module, "sample_unobserved_items", side_effect=capture
        ):
            model.fit(
                positives,
                epochs=3,
                batch_size=2,
                learning_rate=0.01,
                num_negatives=num_negatives,
                forbidden_items_by_user=forbidden,
            )

        self.assertTrue(sampled)
        for user, item in sampled:
            self.assertEqual(item, expected_only_candidate[user])
            self.assertNotIn(item, forbidden[user])

    def test_bpr_never_samples_forbidden_holdout_or_removed_positives(self):
        self._assert_forbidden_items_are_not_sampled("bpr", num_negatives=0)

    def test_mse_never_samples_forbidden_holdout_or_removed_positives(self):
        self._assert_forbidden_items_are_not_sampled("mse", num_negatives=2)

    def test_forbidden_item_validation(self):
        model = MatrixFactorization(2, 4, embedding_dim=4, objective="bpr")
        positives = np.asarray([[0, 0], [1, 1]], dtype=np.int64)
        with self.assertRaises(IndexError):
            model.fit(positives, epochs=1, forbidden_items_by_user={2: {0}})
        with self.assertRaises(IndexError):
            model.fit(positives, epochs=1, forbidden_items_by_user={0: {4}})
        with self.assertRaises(TypeError):
            model.fit(positives, epochs=1, forbidden_items_by_user={"0": {1}})
        with self.assertRaises(TypeError):
            model.fit(positives, epochs=1, forbidden_items_by_user={0: {1.5}})
        with self.assertRaises(ValueError):
            model.fit(positives, epochs=1, forbidden_items_by_user=[{0}])

    def test_forbidden_item_sequence_is_supported(self):
        model = MatrixFactorization(2, 4, embedding_dim=4, objective="bpr")
        positives = np.asarray([[0, 0], [1, 1]], dtype=np.int64)
        model.fit(
            positives,
            epochs=1,
            forbidden_items_by_user=[{2}, np.asarray([0, 3])],
        )

    def test_per_user_item_subset(self):
        model = MatrixFactorization(3, 3, embedding_dim=8, seed=1)
        requested = np.asarray([[0, 2], [1, 0]])
        subset = model.score_users([0, 1], requested)
        full = model.score_users([0, 1])
        np.testing.assert_allclose(
            subset, np.take_along_axis(full, requested, axis=1), atol=1e-7
        )

    def test_early_stopping_restores_best_state(self):
        model = MatrixFactorization(3, 3, embedding_dim=8, objective="mse", seed=3)
        calls = []

        def validation_callback(candidate):
            calls.append(candidate.score_users([0])[0, 0])
            return -float(len(calls))

        model.fit(
            self.pointwise,
            epochs=10,
            batch_size=9,
            learning_rate=0.01,
            validation_callback=validation_callback,
            patience=2,
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(model.best_epoch_, 0)
        self.assertAlmostEqual(float(model.score_users([0])[0, 0]), float(calls[0]), places=7)


class EASETest(unittest.TestCase):
    def setUp(self):
        self.matrix = np.asarray(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.float64,
        )

    def test_fit_zero_diagonal_and_batched_scoring(self):
        model = EASE(regularization=1.0).fit(self.matrix)
        np.testing.assert_allclose(np.diag(model.coef_), 0.0)
        scores = model.score_users([0, 1, 2], batch_size=1)
        self.assertEqual(scores.shape, (3, 4))
        subset = model.score_users([0, 1, 2], [1, 3], batch_size=2)
        np.testing.assert_allclose(subset, scores[:, [1, 3]], rtol=1e-6)

    def test_coordinate_input_without_scipy(self):
        coordinates = np.asarray([[0, 0], [0, 1], [1, 1], [1, 2]], dtype=np.int64)
        model = EASE(regularization=1.0, n_users=2, n_items=3).fit(coordinates)
        self.assertEqual(model.score_users([0, 1]).shape, (2, 3))

    def test_checkpoint_round_trip(self):
        model = EASE(regularization=2.0).fit(self.matrix)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ease.pt"
            model.save(checkpoint)
            restored = EASE.load(checkpoint)
            np.testing.assert_allclose(
                restored.score_users([0, 3]), model.score_users([0, 3]), atol=0
            )


if __name__ == "__main__":
    unittest.main()
