"""MovieLens 100K/1M parsers and shared, model-independent data types.

Only stable GroupLens releases are supported.  This module deliberately performs
no downloading: ``root`` must point either at an extracted release directory or
at its parent (for example, ``data/ml-100k`` or ``data``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Union


PathLike = Union[str, Path]

_PLACEHOLDER_GENRES = {"unknown", "(no genres listed)"}


class Feedback(IntEnum):
    """Three-way interpretation used throughout the benchmark."""

    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1


def classify_rating(
    rating: float,
    positive_threshold: float = 4.0,
    negative_threshold: float = 2.0,
) -> Feedback:
    """Convert a rating into positive, neutral, or explicit-negative feedback."""

    if negative_threshold >= positive_threshold:
        raise ValueError("negative_threshold must be lower than positive_threshold")
    if rating >= positive_threshold:
        return Feedback.POSITIVE
    if rating <= negative_threshold:
        return Feedback.NEGATIVE
    return Feedback.NEUTRAL


@dataclass(frozen=True, order=True)
class Interaction:
    """One rating after conversion to contiguous benchmark indices."""

    user_idx: int
    item_idx: int
    rating: float
    timestamp: int
    feedback: Feedback

    def __post_init__(self) -> None:
        if self.user_idx < 0 or self.item_idx < 0:
            raise ValueError("user_idx and item_idx must be non-negative")
        if not 1.0 <= self.rating <= 5.0:
            raise ValueError(f"MovieLens rating must be in [1, 5], got {self.rating}")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")


@dataclass(frozen=True)
class MovieLensData:
    """Canonical in-memory representation shared by all benchmark models.

    Raw identifiers and genre names are sorted before indexing, making every
    mapping deterministic and contiguous.  ``item_genres`` and ``item_titles``
    are indexed by ``item_idx``.
    """

    variant: str
    interactions: Tuple[Interaction, ...]
    raw_user_ids: Tuple[int, ...]
    raw_item_ids: Tuple[int, ...]
    genre_names: Tuple[str, ...]
    item_genres: Tuple[Tuple[int, ...], ...]
    item_titles: Tuple[str, ...]
    positive_threshold: float = 4.0
    negative_threshold: float = 2.0

    def __post_init__(self) -> None:
        if self.variant not in {"ml-100k", "ml-1m"}:
            raise ValueError(f"unsupported MovieLens variant: {self.variant!r}")
        if self.negative_threshold >= self.positive_threshold:
            raise ValueError("negative_threshold must be lower than positive_threshold")
        if tuple(sorted(set(self.raw_user_ids))) != self.raw_user_ids:
            raise ValueError("raw_user_ids must be sorted and unique")
        if tuple(sorted(set(self.raw_item_ids))) != self.raw_item_ids:
            raise ValueError("raw_item_ids must be sorted and unique")
        if tuple(sorted(set(self.genre_names))) != self.genre_names:
            raise ValueError("genre_names must be sorted and unique")
        if len(self.item_genres) != len(self.raw_item_ids):
            raise ValueError("item_genres must have one entry per item")
        if len(self.item_titles) != len(self.raw_item_ids):
            raise ValueError("item_titles must have one entry per item")
        for genres in self.item_genres:
            if tuple(sorted(set(genres))) != genres:
                raise ValueError("each item genre list must be sorted and unique")
            if any(g < 0 or g >= len(self.genre_names) for g in genres):
                raise ValueError("item genre index is out of range")
        for interaction in self.interactions:
            if interaction.user_idx >= len(self.raw_user_ids):
                raise ValueError("interaction user index is out of range")
            if interaction.item_idx >= len(self.raw_item_ids):
                raise ValueError("interaction item index is out of range")

    @property
    def n_users(self) -> int:
        return len(self.raw_user_ids)

    @property
    def n_items(self) -> int:
        return len(self.raw_item_ids)

    @property
    def n_genres(self) -> int:
        return len(self.genre_names)

    @property
    def user_to_index(self) -> Mapping[int, int]:
        return MappingProxyType({raw_id: i for i, raw_id in enumerate(self.raw_user_ids)})

    @property
    def item_to_index(self) -> Mapping[int, int]:
        return MappingProxyType({raw_id: i for i, raw_id in enumerate(self.raw_item_ids)})

    @property
    def genre_to_index(self) -> Mapping[str, int]:
        return MappingProxyType({name: i for i, name in enumerate(self.genre_names)})


_ML100K_GENRES = (
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


def _resolve_release(root: PathLike, release: str, required: Sequence[str]) -> Path:
    root_path = Path(root).expanduser()
    candidates = (root_path, root_path / release)
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in required):
            return candidate
    required_text = ", ".join(required)
    raise FileNotFoundError(
        f"could not find {release} below {root_path}; expected files: {required_text}"
    )


def _read_nonempty_lines(path: Path, encoding: str) -> Iterable[Tuple[int, str]]:
    with path.open("r", encoding=encoding) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if line:
                yield line_number, line


def _parse_int(value: str, path: Path, line_number: int, field: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: invalid {field}: {value!r}") from error


def _parse_rating(value: str, path: Path, line_number: int) -> float:
    try:
        rating = float(value)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: invalid rating: {value!r}") from error
    if not 1.0 <= rating <= 5.0:
        raise ValueError(f"{path}:{line_number}: rating must be in [1, 5], got {rating}")
    return rating


def _build_dataset(
    *,
    variant: str,
    ratings: Sequence[Tuple[int, int, float, int]],
    movies: Mapping[int, Tuple[str, Tuple[str, ...]]],
    positive_threshold: float,
    negative_threshold: float,
) -> MovieLensData:
    if not ratings:
        raise ValueError(f"{variant} ratings file is empty")
    if not movies:
        raise ValueError(f"{variant} movies file is empty")
    classify_rating(3.0, positive_threshold, negative_threshold)  # validate thresholds

    unknown_items = sorted({item for _, item, _, _ in ratings} - set(movies))
    if unknown_items:
        sample = ", ".join(map(str, unknown_items[:5]))
        raise ValueError(f"ratings reference items absent from movie metadata: {sample}")

    raw_users = tuple(sorted({user for user, _, _, _ in ratings}))
    # Collaborative-filtering models cannot meaningfully rank metadata-only
    # movies. Keep the catalog to items occurring in the rating file, while
    # still parsing and validating every metadata row in the loaders.
    raw_items = tuple(sorted({item for _, item, _, _ in ratings}))
    genres = tuple(
        sorted(
            {
                genre
                for item in raw_items
                for genre in movies[item][1]
                if genre not in _PLACEHOLDER_GENRES
            }
        )
    )
    user_map = {raw_id: idx for idx, raw_id in enumerate(raw_users)}
    item_map = {raw_id: idx for idx, raw_id in enumerate(raw_items)}
    genre_map = {name: idx for idx, name in enumerate(genres)}

    encoded = tuple(
        sorted(
            (
                Interaction(
                    user_idx=user_map[user],
                    item_idx=item_map[item],
                    rating=rating,
                    timestamp=timestamp,
                    feedback=classify_rating(
                        rating, positive_threshold, negative_threshold
                    ),
                )
                for user, item, rating, timestamp in ratings
            ),
            key=lambda x: (x.user_idx, x.timestamp, x.item_idx, x.rating),
        )
    )
    item_titles = tuple(movies[item][0] for item in raw_items)
    item_genres = tuple(
        tuple(
            sorted(
                genre_map[genre]
                for genre in movies[item][1]
                if genre not in _PLACEHOLDER_GENRES
            )
        )
        for item in raw_items
    )
    return MovieLensData(
        variant=variant,
        interactions=encoded,
        raw_user_ids=raw_users,
        raw_item_ids=raw_items,
        genre_names=genres,
        item_genres=item_genres,
        item_titles=item_titles,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
    )


def load_movielens_100k(
    root: PathLike,
    *,
    positive_threshold: float = 4.0,
    negative_threshold: float = 2.0,
) -> MovieLensData:
    """Load an extracted stable MovieLens 100K release."""

    release = _resolve_release(root, "ml-100k", ("u.data", "u.item"))
    ratings_path = release / "u.data"
    movies_path = release / "u.item"
    ratings = []
    for line_number, line in _read_nonempty_lines(ratings_path, "utf-8"):
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"{ratings_path}:{line_number}: expected 4 tab-separated fields")
        user = _parse_int(fields[0], ratings_path, line_number, "user id")
        item = _parse_int(fields[1], ratings_path, line_number, "item id")
        rating = _parse_rating(fields[2], ratings_path, line_number)
        timestamp = _parse_int(fields[3], ratings_path, line_number, "timestamp")
        ratings.append((user, item, rating, timestamp))

    movies: Dict[int, Tuple[str, Tuple[str, ...]]] = {}
    for line_number, line in _read_nonempty_lines(movies_path, "latin-1"):
        fields = line.split("|")
        if len(fields) != 24:
            raise ValueError(
                f"{movies_path}:{line_number}: expected 24 pipe-separated fields, got {len(fields)}"
            )
        item = _parse_int(fields[0], movies_path, line_number, "item id")
        if item in movies:
            raise ValueError(f"{movies_path}:{line_number}: duplicate item id {item}")
        flags = []
        for genre, flag_text in zip(_ML100K_GENRES, fields[5:]):
            flag = _parse_int(flag_text, movies_path, line_number, f"{genre} flag")
            if flag not in (0, 1):
                raise ValueError(f"{movies_path}:{line_number}: genre flag must be 0 or 1")
            if flag:
                flags.append(genre)
        movies[item] = (fields[1], tuple(flags))

    return _build_dataset(
        variant="ml-100k",
        ratings=ratings,
        movies=movies,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
    )


def load_movielens_1m(
    root: PathLike,
    *,
    positive_threshold: float = 4.0,
    negative_threshold: float = 2.0,
) -> MovieLensData:
    """Load an extracted stable MovieLens 1M release."""

    release = _resolve_release(root, "ml-1m", ("ratings.dat", "movies.dat"))
    ratings_path = release / "ratings.dat"
    movies_path = release / "movies.dat"
    ratings = []
    for line_number, line in _read_nonempty_lines(ratings_path, "latin-1"):
        fields = line.split("::")
        if len(fields) != 4:
            raise ValueError(f"{ratings_path}:{line_number}: expected 4 ::-separated fields")
        user = _parse_int(fields[0], ratings_path, line_number, "user id")
        item = _parse_int(fields[1], ratings_path, line_number, "movie id")
        rating = _parse_rating(fields[2], ratings_path, line_number)
        timestamp = _parse_int(fields[3], ratings_path, line_number, "timestamp")
        ratings.append((user, item, rating, timestamp))

    movies: Dict[int, Tuple[str, Tuple[str, ...]]] = {}
    for line_number, line in _read_nonempty_lines(movies_path, "latin-1"):
        fields = line.split("::", 2)
        if len(fields) != 3:
            raise ValueError(f"{movies_path}:{line_number}: expected 3 ::-separated fields")
        item = _parse_int(fields[0], movies_path, line_number, "movie id")
        if item in movies:
            raise ValueError(f"{movies_path}:{line_number}: duplicate movie id {item}")
        item_genres = tuple(sorted(set(filter(None, fields[2].split("|")))))
        movies[item] = (fields[1], item_genres)

    return _build_dataset(
        variant="ml-1m",
        ratings=ratings,
        movies=movies,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
    )


def load_movielens(
    root: PathLike,
    variant: str,
    *,
    positive_threshold: float = 4.0,
    negative_threshold: float = 2.0,
) -> MovieLensData:
    """Load ``ml-100k`` or ``ml-1m`` using a normalized variant alias."""

    normalized = variant.strip().lower().replace("_", "-")
    if normalized in {"100k", "ml-100k", "movielens-100k"}:
        return load_movielens_100k(
            root,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        )
    if normalized in {"1m", "ml-1m", "movielens-1m"}:
        return load_movielens_1m(
            root,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        )
    raise ValueError(f"unsupported MovieLens variant: {variant!r}")
