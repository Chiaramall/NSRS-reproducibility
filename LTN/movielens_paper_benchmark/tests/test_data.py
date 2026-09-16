from pathlib import Path

import pytest

from movielens_paper_benchmark.data import Feedback, Interaction, MovieLensData, load_movielens, load_movielens_100k
from movielens_paper_benchmark.splits import (
    DataSplit,
    derive_user_genre_preferences,
    sparsify_training,
    temporal_leave_two_out,
)


GENRES_100K = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)


def _u_item_row(item_id, title, enabled_genres):
    flags = ["1" if genre in enabled_genres else "0" for genre in GENRES_100K]
    return "|".join([str(item_id), title, "01-Jan-1995", "", "http://example"] + flags)


def _write_100k(root: Path):
    release = root / "ml-100k"
    release.mkdir()
    # Metadata is deliberately not ordered by raw id.
    rows = [
        _u_item_row(4, "Four", {"Comedy", "unknown"}),
        _u_item_row(2, "Two", {"Drama"}),
        _u_item_row(1, "One", {"Action"}),
        _u_item_row(3, "Three", {"Comedy"}),
        _u_item_row(99, "Metadata only", {"unknown"}),
    ]
    (release / "u.item").write_text("\n".join(rows) + "\n", encoding="latin-1")
    ratings = [
        # User 10 has three positives and is eligible.  The negative at t=40
        # lies between validation and test and must not enter training.
        "10\t1\t5\t10",
        "10\t2\t1\t20",
        "10\t3\t4\t30",
        "10\t4\t2\t40",
        "10\t2\t5\t50",
        # User 20 has only two positives and is dropped.
        "20\t1\t4\t10",
        "20\t2\t4\t20",
    ]
    (release / "u.data").write_text("\n".join(ratings) + "\n", encoding="utf-8")
    return release


def test_load_100k_has_deterministic_contiguous_mappings(tmp_path):
    _write_100k(tmp_path)
    data = load_movielens_100k(tmp_path)

    assert data.variant == "ml-100k"
    assert data.raw_user_ids == (10, 20)
    assert data.raw_item_ids == (1, 2, 3, 4)
    assert dict(data.user_to_index) == {10: 0, 20: 1}
    assert dict(data.item_to_index) == {1: 0, 2: 1, 3: 2, 4: 3}
    assert data.genre_names == ("Action", "Comedy", "Drama")
    assert "unknown" not in data.genre_names
    assert data.item_genres[data.item_to_index[1]] == (data.genre_to_index["Action"],)
    assert [event.feedback for event in data.interactions[:5]] == [
        Feedback.POSITIVE,
        Feedback.NEGATIVE,
        Feedback.POSITIVE,
        Feedback.NEGATIVE,
        Feedback.POSITIVE,
    ]


def test_temporal_split_is_strict_and_drops_ineligible_users(tmp_path):
    _write_100k(tmp_path)
    data = load_movielens(tmp_path, "100k")
    split = temporal_leave_two_out(data)

    assert split.eligible_users == (0,)
    assert split.dropped_users == (1,)
    assert [(x.item_idx, x.timestamp) for x in split.train] == [(0, 10), (1, 20)]
    assert [(x.item_idx, x.timestamp) for x in split.validation] == [(2, 30)]
    assert [(x.item_idx, x.timestamp) for x in split.test] == [(1, 50)]
    assert split.excluded_future_count == 1
    assert max(x.timestamp for x in split.train) < split.validation[0].timestamp


def test_sparsification_is_deterministic_and_preserves_positive(tmp_path):
    _write_100k(tmp_path)
    split = temporal_leave_two_out(load_movielens_100k(tmp_path))

    first = sparsify_training(split, 0.5, seed=17)
    second = sparsify_training(split, 0.5, seed=17)
    assert first == second
    assert len(first.train) == 1
    assert first.train[0].feedback is Feedback.POSITIVE
    assert first.validation == split.validation
    assert first.test == split.test
    assert first.train_proportion == 0.5
    assert first.sparsity_seed == 17


def test_sparsification_proportions_are_nested_prefixes():
    train = tuple(
        Interaction(
            user_idx=0,
            item_idx=item,
            rating=5.0 if item % 5 == 0 else 3.0,
            timestamp=item + 1,
            feedback=(
                Feedback.POSITIVE
                if item % 5 == 0
                else Feedback.NEUTRAL
            ),
        )
        for item in range(27)
    )
    split = DataSplit(
        train=train,
        validation=(Interaction(0, 27, 5.0, 28, Feedback.POSITIVE),),
        test=(Interaction(0, 28, 5.0, 29, Feedback.POSITIVE),),
        eligible_users=(0,),
        dropped_users=(),
    )
    samples = [
        {event.item_idx for event in sparsify_training(split, p, seed=123).train}
        for p in (0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    assert all(smaller <= larger for smaller, larger in zip(samples, samples[1:]))
    assert all(any(item % 5 == 0 for item in sample) for sample in samples)


def test_genre_preferences_use_only_supplied_sparse_training(tmp_path):
    _write_100k(tmp_path)
    data = load_movielens_100k(tmp_path)
    split = temporal_leave_two_out(data)
    sparse = sparsify_training(split, 0.5, seed=9)

    preferences = derive_user_genre_preferences(data, sparse.train, min_support=1)
    assert len(preferences) == 1
    preference = preferences[0]
    assert preference.user_idx == 0
    assert data.genre_names[preference.genre_idx] == "Action"
    assert preference.mean_rating == 5.0
    assert preference.feedback is Feedback.POSITIVE
    # Comedy occurs only in validation/future events, so it cannot leak in.
    assert all(data.genre_names[p.genre_idx] != "Comedy" for p in preferences)


def test_load_1m_from_direct_release_layout(tmp_path):
    (tmp_path / "movies.dat").write_text(
        "20::Later Movie (2000)::Comedy|Drama\n"
        "10::Earlier Movie (1999)::Action|Comedy|(no genres listed)\n"
        "30::Metadata Only (2001)::(no genres listed)\n",
        encoding="latin-1",
    )
    (tmp_path / "ratings.dat").write_text(
        "7::20::3::200\n7::10::4::100\n", encoding="latin-1"
    )

    data = load_movielens(tmp_path, "ml_1m")
    assert data.raw_user_ids == (7,)
    assert data.raw_item_ids == (10, 20)
    assert data.genre_names == ("Action", "Comedy", "Drama")
    assert "(no genres listed)" not in data.genre_names
    assert data.interactions[0].item_idx == 0
    assert data.interactions[0].feedback is Feedback.POSITIVE
    assert data.interactions[1].feedback is Feedback.NEUTRAL


def test_loader_rejects_rating_for_unknown_movie(tmp_path):
    release = _write_100k(tmp_path)
    with (release / "u.data").open("a", encoding="utf-8") as stream:
        stream.write("10\t999\t5\t60\n")
    with pytest.raises(ValueError, match="absent from movie metadata"):
        load_movielens_100k(tmp_path)


@pytest.mark.parametrize("proportion", [0.0, -0.1, 1.1])
def test_sparsification_rejects_invalid_proportions(tmp_path, proportion):
    _write_100k(tmp_path)
    split = temporal_leave_two_out(load_movielens_100k(tmp_path))
    with pytest.raises(ValueError, match="proportion"):
        sparsify_training(split, proportion, seed=0)


def test_warm_start_target_filter_is_iterative():
    histories = {
        0: (0, 1, 2),
        1: (1, 2, 3, 4),
        2: (5, 6, 7, 8),
        3: (7, 8, 5, 6),
    }
    interactions = tuple(
        Interaction(
            user_idx=user,
            item_idx=item,
            rating=5.0,
            timestamp=position + 1,
            feedback=Feedback.POSITIVE,
        )
        for user, items in histories.items()
        for position, item in enumerate(items)
    )
    data = MovieLensData(
        variant="ml-1m",
        interactions=interactions,
        raw_user_ids=(0, 1, 2, 3),
        raw_item_ids=tuple(range(9)),
        genre_names=(),
        item_genres=((),) * 9,
        item_titles=tuple(str(item) for item in range(9)),
    )

    unfiltered = temporal_leave_two_out(data)
    filtered = temporal_leave_two_out(
        data, require_warm_start_targets=True
    )

    assert unfiltered.eligible_users == (0, 1, 2, 3)
    assert filtered.eligible_users == (2, 3)
    assert filtered.dropped_users == (0, 1)
    assert {event.user_idx for event in filtered.train} == {2, 3}
    training_items = {event.item_idx for event in filtered.train}
    assert all(
        event.item_idx in training_items
        for event in filtered.validation + filtered.test
    )
