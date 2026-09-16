#!/usr/bin/env python3
"""Run review-aware BPR-HFT and DeepCoNN top-N baselines.

The CAFE paper names these baselines but does not release their code,
hyperparameters, or top-N adaptations. This runner implements declared
common-protocol variants on the exact public PGPR/CAFE 70/30 split.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "baselines" / "source" / "PGPR-datasets"
ATOMIC_ROOT = ROOT / "baselines" / "data"
CACHE_ROOT = ROOT / "baselines" / "cache" / "review_baselines"
DATASETS = {
    "beauty": ("Amazon_Beauty", "beauty_paper"),
    "cell": ("Amazon_Cellphones", "cell_paper"),
    "clothing": ("Amazon_Clothing", "clothing_paper"),
    "cd": ("Amazon_CDs", "cds_paper"),
}
MODELS = ("BPR-HFT", "DeepCoNN")
METRICS = ("ndcg", "recall", "hr", "precision")
CACHE_VERSION = 1


@dataclass
class ReviewData:
    dataset: str
    n_users: int
    n_items: int
    n_words: int
    train_pairs: np.ndarray
    review_words: np.ndarray
    user_docs: np.ndarray
    item_docs: np.ndarray
    train_items: list[np.ndarray]
    test_items: list[np.ndarray]


def count_gzip_lines(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def atomic_save_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_atomic_pairs(path: Path) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            uid, pid = line.rstrip("\n").split("\t")
            rows.append((int(uid), int(pid)))
    return np.asarray(rows, dtype=np.int64)


def group_items(
    pairs: np.ndarray, n_users: int, *, unique: bool
) -> list[np.ndarray]:
    grouped: list[list[int]] = [[] for _ in range(n_users)]
    for uid, pid in pairs:
        grouped[int(uid)].append(int(pid))
    if unique:
        return [
            np.asarray(sorted(set(items)), dtype=np.int64) for items in grouped
        ]
    return [np.asarray(items, dtype=np.int64) for items in grouped]


def cache_directory(args: argparse.Namespace) -> Path:
    return (
        CACHE_ROOT
        / args.dataset
        / (
            f"v{CACHE_VERSION}_u{args.max_user_words}"
            f"_i{args.max_item_words}_r{args.review_words}"
        )
    )


def required_cache_files(cache: Path) -> tuple[Path, ...]:
    return (
        cache / "metadata.json",
        cache / "train_pairs.npy",
        cache / "review_words.npy",
        cache / "user_docs.npy",
        cache / "item_docs.npy",
    )


def select_review_words(words: list[int], count: int) -> np.ndarray:
    selected = np.zeros(count, dtype=np.int32)
    if not words:
        return selected
    if len(words) <= count:
        selected[: len(words)] = np.asarray(words, dtype=np.int32) + 1
        return selected
    positions = np.linspace(0, len(words) - 1, num=count, dtype=np.int64)
    selected[:] = np.asarray([words[int(pos)] + 1 for pos in positions])
    return selected


def append_document_words(
    matrix: np.memmap,
    cursors: np.ndarray,
    entity_id: int,
    words: list[int],
) -> None:
    start = int(cursors[entity_id])
    if start >= matrix.shape[1] or not words:
        return
    count = min(matrix.shape[1] - start, len(words))
    matrix[entity_id, start : start + count] = (
        np.asarray(words[:count], dtype=np.int32) + 1
    )
    cursors[entity_id] += count


def build_review_cache(args: argparse.Namespace) -> None:
    source_name, token = DATASETS[args.dataset]
    source = SOURCE_ROOT / source_name
    atomic = ATOMIC_ROOT / token
    metadata_path = atomic / "metadata.json"
    review_path = source / "train.txt.gz"
    users_path = source / "users.txt.gz"
    vocab_path = source / "vocab.txt.gz"
    for path in (metadata_path, review_path, users_path, vocab_path):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    n_users = count_gzip_lines(users_path)
    n_items = int(metadata["items"])
    n_words = count_gzip_lines(vocab_path)
    n_reviews = int(metadata["train_rows"])
    cache = cache_directory(args)
    cache.mkdir(parents=True, exist_ok=True)
    print(
        f"Building review cache for {args.dataset}: "
        f"{n_reviews} reviews, {n_users} users, {n_items} items, "
        f"{n_words} words",
        flush=True,
    )

    train_pairs = np.lib.format.open_memmap(
        cache / "train_pairs.npy", mode="w+", dtype=np.int64,
        shape=(n_reviews, 2),
    )
    review_words = np.lib.format.open_memmap(
        cache / "review_words.npy", mode="w+", dtype=np.int32,
        shape=(n_reviews, args.review_words),
    )
    user_docs = np.lib.format.open_memmap(
        cache / "user_docs.npy", mode="w+", dtype=np.int32,
        shape=(n_users, args.max_user_words),
    )
    item_docs = np.lib.format.open_memmap(
        cache / "item_docs.npy", mode="w+", dtype=np.int32,
        shape=(n_items, args.max_item_words),
    )
    train_pairs[:] = 0
    review_words[:] = 0
    user_docs[:] = 0
    item_docs[:] = 0
    user_cursors = np.zeros(n_users, dtype=np.int32)
    item_cursors = np.zeros(n_items, dtype=np.int32)

    seen = 0
    with gzip.open(review_path, "rt", encoding="utf-8") as handle:
        for seen, line in enumerate(handle, start=1):
            cells = line.split()
            uid = int(cells[0])
            pid = int(cells[1])
            words = [int(word) for word in cells[2:]]
            if uid >= n_users or pid >= n_items:
                raise ValueError(
                    f"Out-of-range review id uid={uid}, pid={pid}"
                )
            index = seen - 1
            train_pairs[index] = (uid, pid)
            review_words[index] = select_review_words(words, args.review_words)
            append_document_words(user_docs, user_cursors, uid, words)
            append_document_words(item_docs, item_cursors, pid, words)
    if seen != n_reviews:
        raise RuntimeError(
            f"Expected {n_reviews} training reviews, observed {seen}"
        )

    for array in (train_pairs, review_words, user_docs, item_docs):
        array.flush()
    atomic_pairs = read_atomic_pairs(atomic / f"{token}.train.inter")
    review_array = np.asarray(train_pairs)
    review_order = np.lexsort((review_array[:, 1], review_array[:, 0]))
    atomic_order = np.lexsort((atomic_pairs[:, 1], atomic_pairs[:, 0]))
    if not np.array_equal(
        review_array[review_order], atomic_pairs[atomic_order]
    ):
        raise RuntimeError(
            "Review rows do not match the common-protocol training split"
        )
    atomic_save_json(
        cache / "metadata.json",
        {
            "version": CACHE_VERSION,
            "dataset": args.dataset,
            "source": str(review_path.relative_to(ROOT)),
            "n_users": n_users,
            "n_items": n_items,
            "n_words": n_words,
            "n_reviews": n_reviews,
            "max_user_words": args.max_user_words,
            "max_item_words": args.max_item_words,
            "review_words": args.review_words,
            "training_reviews_only": True,
        },
    )
    print(f"Review cache ready: {cache.relative_to(ROOT)}", flush=True)


def load_review_data(args: argparse.Namespace) -> ReviewData:
    cache = cache_directory(args)
    if not all(path.exists() for path in required_cache_files(cache)):
        build_review_cache(args)
    metadata = json.loads(
        (cache / "metadata.json").read_text(encoding="utf-8")
    )
    _source_name, token = DATASETS[args.dataset]
    atomic = ATOMIC_ROOT / token
    train_pairs = np.load(cache / "train_pairs.npy", mmap_mode="r")
    review_words = np.load(cache / "review_words.npy", mmap_mode="r")
    user_docs = np.load(cache / "user_docs.npy", mmap_mode="r")
    item_docs = np.load(cache / "item_docs.npy", mmap_mode="r")
    test_pairs = read_atomic_pairs(atomic / f"{token}.test.inter")
    n_users = int(metadata["n_users"])
    return ReviewData(
        dataset=args.dataset,
        n_users=n_users,
        n_items=int(metadata["n_items"]),
        n_words=int(metadata["n_words"]),
        train_pairs=train_pairs,
        review_words=review_words,
        user_docs=user_docs,
        item_docs=item_docs,
        train_items=group_items(train_pairs, n_users, unique=True),
        test_items=group_items(test_pairs, n_users, unique=False),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_negative_items(
    users: np.ndarray,
    n_items: int,
    train_items: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.integers(0, n_items, size=len(users), dtype=np.int64)
    for index, uid in enumerate(users):
        positives = train_items[int(uid)]
        while np.any(positives == negatives[index]):
            negatives[index] = rng.integers(0, n_items)
    return negatives


class BPRHFT(nn.Module):
    """Pairwise implicit HFT with a negative-sampled topic objective."""

    def __init__(
        self, n_users: int, n_items: int, n_words: int, factors: int
    ) -> None:
        super().__init__()
        self.user_factors = nn.Embedding(n_users, factors)
        self.item_factors = nn.Embedding(n_items, factors)
        self.item_bias = nn.Embedding(n_items, 1)
        self.topic_words = nn.Embedding(n_words + 1, factors, padding_idx=0)
        self.kappa_raw = nn.Parameter(torch.tensor(0.0))
        for embedding in (
            self.user_factors,
            self.item_factors,
            self.topic_words,
        ):
            nn.init.normal_(embedding.weight, std=0.02)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.topic_words.weight[0].zero_()

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (
            (self.user_factors(users) * self.item_factors(items)).sum(dim=1)
            + self.item_bias(items).squeeze(1)
        )

    def review_loss(
        self,
        items: torch.Tensor,
        words: torch.Tensor,
        negative_words: int,
    ) -> torch.Tensor:
        valid = words.ne(0)
        if not torch.any(valid):
            return self.item_factors.weight.new_zeros(())
        repeated_items = items[:, None].expand_as(words)[valid]
        positive_words = words[valid]
        kappa = F.softplus(self.kappa_raw) + 1e-4
        topics = F.softmax(
            kappa * self.item_factors(repeated_items), dim=1
        )
        positive_score = (
            topics * self.topic_words(positive_words)
        ).sum(dim=1)
        sampled = torch.randint(
            1,
            self.topic_words.num_embeddings,
            (len(positive_words), negative_words),
            device=items.device,
        )
        negative_score = torch.einsum(
            "bd,bnd->bn", topics, self.topic_words(sampled)
        )
        return (
            F.softplus(-positive_score).mean()
            + F.softplus(negative_score).mean()
        )

    def all_item_representations(self) -> torch.Tensor:
        return self.item_factors.weight

    def user_representations(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_factors(users)


class TextCNN(nn.Module):
    def __init__(
        self,
        word_embedding: nn.Embedding,
        embedding_size: int,
        filters: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.word_embedding = word_embedding
        self.convolution = nn.Conv1d(
            embedding_size,
            filters,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.projection = nn.Linear(filters, filters)

    def forward(self, documents: torch.Tensor) -> torch.Tensor:
        valid_document = documents.ne(0).any(dim=1, keepdim=True)
        embedded = self.word_embedding(documents).transpose(1, 2)
        convolved = F.relu(self.convolution(embedded))
        pooled = torch.amax(convolved, dim=2)
        encoded = torch.tanh(self.projection(pooled))
        return encoded * valid_document


class DeepCoNN(nn.Module):
    """Two review CNNs coupled by a pairwise top-N interaction head."""

    def __init__(
        self,
        n_items: int,
        n_words: int,
        word_embedding_size: int,
        filters: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        word_embedding = nn.Embedding(
            n_words + 1, word_embedding_size, padding_idx=0
        )
        nn.init.normal_(word_embedding.weight, std=0.02)
        with torch.no_grad():
            word_embedding.weight[0].zero_()
        self.user_encoder = TextCNN(
            word_embedding, word_embedding_size, filters, kernel_size
        )
        self.item_encoder = TextCNN(
            word_embedding, word_embedding_size, filters, kernel_size
        )
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.item_bias.weight)

    def score_encoded(
        self,
        user_representation: torch.Tensor,
        item_representation: torch.Tensor,
        items: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (user_representation * item_representation).sum(dim=1)
            / math.sqrt(user_representation.shape[1])
            + self.item_bias(items).squeeze(1)
        )


def numpy_documents(
    matrix: np.ndarray, ids: np.ndarray, device: torch.device
) -> torch.Tensor:
    values = np.asarray(matrix[ids], dtype=np.int64)
    return torch.from_numpy(values).to(device=device, non_blocking=True)


def train_epoch_bpr_hft(
    model: BPRHFT,
    data: ReviewData,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    order = rng.permutation(len(data.train_pairs))
    total_loss = 0.0
    batches = 0
    for start in range(0, len(order), args.batch_size):
        if args.max_train_batches and batches >= args.max_train_batches:
            break
        indices = order[start : start + args.batch_size]
        pairs = np.asarray(data.train_pairs[indices])
        user_ids = pairs[:, 0]
        positive_ids = pairs[:, 1]
        negative_ids = sample_negative_items(
            user_ids, data.n_items, data.train_items, rng
        )
        users = torch.from_numpy(user_ids).to(device)
        positives = torch.from_numpy(positive_ids).to(device)
        negatives = torch.from_numpy(negative_ids).to(device)
        words = torch.from_numpy(
            np.asarray(data.review_words[indices], dtype=np.int64)
        ).to(device)

        optimizer.zero_grad(set_to_none=True)
        ranking_loss = F.softplus(
            -(model.score(users, positives) - model.score(users, negatives))
        ).mean()
        text_loss = model.review_loss(positives, words, args.word_negatives)
        loss = ranking_loss + args.review_weight * text_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        batches += 1
    return total_loss / max(batches, 1)


def train_epoch_deepconn(
    model: DeepCoNN,
    data: ReviewData,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    order = rng.permutation(len(data.train_pairs))
    total_loss = 0.0
    batches = 0
    batch_size = args.deepconn_batch_size
    for start in range(0, len(order), batch_size):
        if args.max_train_batches and batches >= args.max_train_batches:
            break
        indices = order[start : start + batch_size]
        pairs = np.asarray(data.train_pairs[indices])
        user_ids = pairs[:, 0]
        positive_ids = pairs[:, 1]
        negative_ids = sample_negative_items(
            user_ids, data.n_items, data.train_items, rng
        )
        users = numpy_documents(data.user_docs, user_ids, device)
        positive_docs = numpy_documents(data.item_docs, positive_ids, device)
        negative_docs = numpy_documents(data.item_docs, negative_ids, device)
        positives = torch.from_numpy(positive_ids).to(device)
        negatives = torch.from_numpy(negative_ids).to(device)

        optimizer.zero_grad(set_to_none=True)
        user_representation = model.user_encoder(users)
        positive_score = model.score_encoded(
            user_representation,
            model.item_encoder(positive_docs),
            positives,
        )
        negative_score = model.score_encoded(
            user_representation,
            model.item_encoder(negative_docs),
            negatives,
        )
        loss = F.softplus(-(positive_score - negative_score)).mean()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        batches += 1
    return total_loss / max(batches, 1)


def encode_deepconn_items(
    model: DeepCoNN,
    data: ReviewData,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    representations: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, data.n_items, args.encode_batch_size):
            ids = np.arange(
                start,
                min(start + args.encode_batch_size, data.n_items),
                dtype=np.int64,
            )
            docs = numpy_documents(data.item_docs, ids, device)
            representations.append(model.item_encoder(docs))
    return torch.cat(representations, dim=0)


def evaluate(
    model: nn.Module,
    model_name: str,
    data: ReviewData,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    evaluation_users = np.asarray(
        [uid for uid, items in enumerate(data.test_items) if len(items)],
        dtype=np.int64,
    )
    if args.max_eval_users:
        evaluation_users = evaluation_users[: args.max_eval_users]
    if model_name == "DeepCoNN":
        assert isinstance(model, DeepCoNN)
        item_representations = encode_deepconn_items(model, data, args, device)
        item_bias = model.item_bias.weight.squeeze(1)
    else:
        assert isinstance(model, BPRHFT)
        item_representations = model.all_item_representations()
        item_bias = model.item_bias.weight.squeeze(1)

    ndcg_values: list[float] = []
    recall_values: list[float] = []
    hr_values: list[float] = []
    precision_values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(evaluation_users), args.eval_batch_size):
            user_ids = evaluation_users[start : start + args.eval_batch_size]
            users = torch.from_numpy(user_ids).to(device)
            if model_name == "DeepCoNN":
                assert isinstance(model, DeepCoNN)
                user_documents = numpy_documents(data.user_docs, user_ids, device)
                user_representations = model.user_encoder(user_documents)
                scores = (
                    user_representations @ item_representations.T
                ) / math.sqrt(user_representations.shape[1])
            else:
                assert isinstance(model, BPRHFT)
                scores = (
                    model.user_representations(users) @ item_representations.T
                )
            scores = scores + item_bias[None, :]
            for row, uid in enumerate(user_ids):
                seen = data.train_items[int(uid)]
                if len(seen):
                    seen_tensor = torch.from_numpy(seen).to(device)
                    scores[row, seen_tensor] = -torch.inf
            recommendations = (
                torch.topk(scores, k=10, dim=1).indices.cpu().numpy()
            )
            for uid, predicted in zip(user_ids, recommendations):
                targets = set(data.test_items[int(uid)].tolist())
                hit_positions = [
                    rank
                    for rank, pid in enumerate(predicted)
                    if int(pid) in targets
                ]
                hits = len(hit_positions)
                dcg = sum(
                    1.0 / math.log2(position + 2)
                    for position in hit_positions
                )
                ideal = min(len(targets), 10)
                idcg = sum(
                    1.0 / math.log2(position + 2)
                    for position in range(ideal)
                )
                ndcg_values.append(dcg / idcg if idcg else 0.0)
                recall_values.append(hits / len(targets))
                hr_values.append(float(hits > 0))
                precision_values.append(hits / 10.0)
    return {
        "ndcg": 100.0 * statistics.fmean(ndcg_values),
        "recall": 100.0 * statistics.fmean(recall_values),
        "hr": 100.0 * statistics.fmean(hr_values),
        "precision": 100.0 * statistics.fmean(precision_values),
    }


def create_model(
    model_name: str, data: ReviewData, args: argparse.Namespace
) -> nn.Module:
    if model_name == "BPR-HFT":
        return BPRHFT(
            data.n_users,
            data.n_items,
            data.n_words,
            args.embedding_size,
        )
    if model_name == "DeepCoNN":
        return DeepCoNN(
            data.n_items,
            data.n_words,
            args.word_embedding_size,
            args.cnn_filters,
            args.cnn_kernel_size,
        )
    raise ValueError(model_name)


def train_and_evaluate(
    model_name: str,
    seed: int,
    data: ReviewData,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    set_seed(seed)
    model = create_model(model_name, data, args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoint_dir = (
        args.results_dir
        / "checkpoints"
        / model_name.lower().replace("-", "_")
        / f"seed_{seed}"
    )
    checkpoint = checkpoint_dir / "last.pt"
    start_epoch = 1
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        print(
            f"Resuming {model_name} seed={seed} at epoch {start_epoch}",
            flush=True,
        )

    rng = np.random.default_rng(seed)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        started = time.monotonic()
        if model_name == "BPR-HFT":
            assert isinstance(model, BPRHFT)
            loss = train_epoch_bpr_hft(
                model, data, optimizer, args, rng, device
            )
        else:
            assert isinstance(model, DeepCoNN)
            loss = train_epoch_deepconn(
                model, data, optimizer, args, rng, device
            )
        atomic_torch_save(
            checkpoint,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "model_name": model_name,
                "seed": seed,
            },
        )
        print(
            f"{model_name} seed={seed} epoch={epoch}/{args.epochs} "
            f"loss={loss:.6f} seconds={time.monotonic() - started:.1f}",
            flush=True,
        )

    metrics = evaluate(model, model_name, data, args, device)
    return {
        "dataset": args.dataset,
        "model": model_name,
        "seed": seed,
        "epochs": args.epochs,
        "embedding_size": (
            args.embedding_size
            if model_name == "BPR-HFT"
            else args.cnn_filters
        ),
        "learning_rate": args.learning_rate,
        "review_weight": (
            args.review_weight if model_name == "BPR-HFT" else ""
        ),
        **metrics,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    materialized = list(rows)
    output: list[dict[str, object]] = []
    for model_name in MODELS:
        model_rows = [
            row for row in materialized if row["model"] == model_name
        ]
        if not model_rows:
            continue
        summary: dict[str, object] = {
            "model": model_name,
            "seeds": len(model_rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in model_rows]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(summary)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="beauty")
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[123, 124, 125, 126, 127]
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--deepconn-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--word-embedding-size", type=int, default=64)
    parser.add_argument("--cnn-filters", type=int, default=64)
    parser.add_argument("--cnn-kernel-size", type=int, default=3)
    parser.add_argument("--max-user-words", type=int, default=256)
    parser.add_argument("--max-item-words", type=int, default=256)
    parser.add_argument("--review-words", type=int, default=8)
    parser.add_argument("--word-negatives", type=int, default=4)
    parser.add_argument("--review-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-users", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cnn_kernel_size % 2 != 1:
        raise ValueError("--cnn-kernel-size must be odd")
    if args.smoke_test:
        args.epochs = 1
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_users = args.max_eval_users or 128
    if args.results_dir is None:
        parent = "smoke_review" if args.smoke_test else args.dataset
        args.results_dir = (
            ROOT / "results" / "paper_baselines" / parent / "review_models"
        )
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    )
    print(f"Using device {device}", flush=True)
    atomic_save_json(
        args.results_dir / "protocol.json",
        {
            "status": "declared common-protocol reimplementation",
            "paper_code_or_top_n_hyperparameters_available": False,
            "split": "public PGPR/CAFE 70/30 train-test split",
            "review_scope": "training reviews only",
            "evaluation": "full-ranking top-10 over non-training items",
            "metrics_percent": list(METRICS),
            "models": {
                "BPR-HFT": (
                    "pairwise BPR plus HFT item-factor-to-topic transform "
                    "and negative-sampled review likelihood"
                ),
                "DeepCoNN": (
                    "separate user/item review CNNs with pairwise BPR "
                    "top-N adaptation"
                ),
            },
            "hyperparameters": {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "embedding_size": args.embedding_size,
                "word_embedding_size": args.word_embedding_size,
                "cnn_filters": args.cnn_filters,
                "cnn_kernel_size": args.cnn_kernel_size,
                "max_user_words": args.max_user_words,
                "max_item_words": args.max_item_words,
                "review_weight": args.review_weight,
            },
        },
    )
    data = load_review_data(args)
    raw_path = args.results_dir / "per_seed.csv"
    rows: list[dict[str, object]] = list(read_csv(raw_path))
    completed = {(str(row["model"]), int(row["seed"])) for row in rows}
    for model_name in args.models:
        for seed in args.seeds:
            if (model_name, seed) in completed:
                print(
                    f"Skipping completed {model_name} seed={seed}",
                    flush=True,
                )
                continue
            row = train_and_evaluate(model_name, seed, data, args, device)
            rows.append(row)
            write_csv(raw_path, rows)
            summary = summarize(rows)
            write_csv(args.results_dir / "summary.csv", summary)
            atomic_save_json(
                args.results_dir / "summary.json",
                {"protocol": "review common-protocol", "results": summary},
            )
            print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
