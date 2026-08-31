#!/usr/bin/env python3
"""Faithful PyTorch port of the original HeteroEmbed/CFKG recommender.

CAFE calls the method from Ai et al. (Algorithms 2018) ``HeteroEmbed``.
The authors' released repository calls it CFKG/KBE4ER and implements it in
TensorFlow 1.x.  This runner ports the original multi-relation NCE objective
to current PyTorch while preserving the CAFE Amazon ids and splits.

Original source:
https://github.com/evison/KBE4ExplainableRecommendation
Pinned locally at commit 6b460fe5436def931800ed291453c431b0fb427f.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
ATOMIC_ROOT = ROOT / "baselines" / "data"
SOURCE_ROOT = ROOT / "baselines" / "source" / "PGPR-datasets"
RESULT_ROOT = ROOT / "results" / "paper_baselines"
ORIGINAL_ROOT = ROOT / "baselines" / "vendor" / "HeteroEmbed-original"
ORIGINAL_COMMIT = "6b460fe5436def931800ed291453c431b0fb427f"
PROTOCOL_VERSION = 1

DATASETS = {
    "beauty": ("beauty_paper", "Amazon_Beauty"),
    "cell": ("cell_paper", "Amazon_Cellphones"),
    "clothing": ("clothing_paper", "Amazon_Clothing"),
    "cd": ("cds_paper", "Amazon_CDs"),
}
KG_RELATIONS = {
    "also_bought": ("also_bought_p_p.txt.gz", "related_product"),
    "also_viewed": ("also_viewed_p_p.txt.gz", "related_product"),
    "bought_together": ("bought_together_p_p.txt.gz", "related_product"),
    "brand": ("brand_p_b.txt.gz", "brand"),
    "categories": ("category_p_c.txt.gz", "categories"),
}
ENTITY_FILES = {
    "user": "users.txt.gz",
    "product": "product.txt.gz",
    "word": "vocab.txt.gz",
    "related_product": "related_product.txt.gz",
    "brand": "brand.txt.gz",
    "categories": "category.txt.gz",
}
METRICS = ("ndcg", "recall", "hr", "precision")


@dataclass
class SplitBundle:
    dataset: str
    split: str
    token: str
    source: Path
    train_path: Path
    evaluation_path: Path
    train_pairs: np.ndarray
    train: sp.csr_matrix
    targets: list[np.ndarray]
    n_users: int
    n_items: int


@dataclass
class ReviewCorpus:
    users: np.ndarray
    products: np.ndarray
    offsets: np.ndarray
    words: np.ndarray
    word_frequency: np.ndarray

    @property
    def reviews(self) -> int:
        return len(self.users)

    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass
class RelationData:
    name: str
    tail_entity: str
    offsets: np.ndarray
    tails: np.ndarray
    frequency: np.ndarray


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def count_gzip_lines(path: Path) -> int:
    with gzip.open(path, "rb") as handle:
        return sum(1 for _ in handle)


def read_pairs(path: Path) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            user, item = line.rstrip("\n").split("\t")
            pairs.append((int(user), int(item)))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def group_targets(pairs: np.ndarray, n_users: int) -> list[np.ndarray]:
    values: list[list[int]] = [[] for _ in range(n_users)]
    for user, item in pairs:
        values[int(user)].append(int(item))
    return [np.asarray(items, dtype=np.int64) for items in values]


def load_split(dataset: str, split: str) -> SplitBundle:
    outer_token, source_name = DATASETS[dataset]
    token = outer_token if split == "outer" else f"{outer_token}_tune"
    directory = ATOMIC_ROOT / token
    suffix = "test" if split == "outer" else "valid"
    train_path = directory / f"{token}.train.inter"
    evaluation_path = directory / f"{token}.{suffix}.inter"
    metadata_path = directory / "metadata.json"
    for path in (train_path, evaluation_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    n_items = int(metadata["items"])
    source = SOURCE_ROOT / source_name
    n_users = count_gzip_lines(source / ENTITY_FILES["user"])
    train_pairs = read_pairs(train_path)
    evaluation_pairs = read_pairs(evaluation_path)
    train = sp.csr_matrix(
        (
            np.ones(len(train_pairs), dtype=np.float32),
            (train_pairs[:, 0], train_pairs[:, 1]),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    train.sum_duplicates()
    train.data[:] = 1.0
    return SplitBundle(
        dataset=dataset,
        split=split,
        token=token,
        source=source,
        train_path=train_path,
        evaluation_path=evaluation_path,
        train_pairs=train_pairs,
        train=train,
        targets=group_targets(evaluation_pairs, n_users),
        n_users=n_users,
        n_items=n_items,
    )


def entity_sizes(source: Path) -> dict[str, int]:
    return {
        name: count_gzip_lines(source / filename)
        for name, filename in ENTITY_FILES.items()
    }


def load_reviews(
    bundle: SplitBundle,
    n_words: int,
    max_reviews: int,
) -> ReviewCorpus:
    allowed = {
        int(user) * bundle.n_items + int(item)
        for user, item in bundle.train_pairs
    }
    users = array("I")
    products = array("I")
    offsets = array("Q", [0])
    words = array("I")
    with gzip.open(
        bundle.source / "train.txt.gz", "rt", encoding="utf-8"
    ) as handle:
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            user, product = int(cells[0]), int(cells[1])
            if user * bundle.n_items + product not in allowed:
                continue
            review_words = (
                [int(value) for value in cells[2].split()]
                if cells[2]
                else []
            )
            if not review_words:
                continue
            users.append(user)
            products.append(product)
            words.extend(review_words)
            offsets.append(len(words))
            if max_reviews and len(users) >= max_reviews:
                break
    user_values = np.frombuffer(users, dtype=np.uint32).astype(
        np.int64, copy=False
    )
    product_values = np.frombuffer(products, dtype=np.uint32).astype(
        np.int64, copy=False
    )
    offset_values = np.frombuffer(offsets, dtype=np.uint64).astype(
        np.int64, copy=False
    )
    word_values = np.frombuffer(words, dtype=np.uint32).astype(
        np.int64, copy=False
    )
    if len(user_values) == 0 or len(word_values) == 0:
        raise RuntimeError("No training reviews/words matched the fit split")
    frequency = np.bincount(word_values, minlength=n_words).astype(np.float64)
    return ReviewCorpus(
        users=user_values,
        products=product_values,
        offsets=offset_values,
        words=word_values,
        word_frequency=frequency,
    )


def load_relation(
    source: Path,
    name: str,
    filename: str,
    tail_entity: str,
    n_items: int,
    tail_size: int,
) -> RelationData:
    offsets = array("Q", [0])
    tails = array("I")
    with gzip.open(source / filename, "rt", encoding="utf-8") as handle:
        lines = 0
        for line in handle:
            tails.extend(int(value) for value in line.split())
            offsets.append(len(tails))
            lines += 1
    if lines != n_items:
        raise RuntimeError(
            f"{filename}: expected {n_items} product rows, found {lines}"
        )
    tail_values = np.frombuffer(tails, dtype=np.uint32).astype(
        np.int64, copy=False
    )
    frequency = np.bincount(
        tail_values, minlength=tail_size
    ).astype(np.float64)
    return RelationData(
        name=name,
        tail_entity=tail_entity,
        offsets=np.frombuffer(offsets, dtype=np.uint64).astype(
            np.int64, copy=False
        ),
        tails=tail_values,
        frequency=frequency,
    )


def load_relations(
    bundle: SplitBundle,
    sizes: dict[str, int],
    selected: tuple[str, ...],
) -> dict[str, RelationData]:
    return {
        name: load_relation(
            bundle.source,
            name,
            KG_RELATIONS[name][0],
            KG_RELATIONS[name][1],
            bundle.n_items,
            sizes[KG_RELATIONS[name][1]],
        )
        for name in selected
    }


def subsampling_rates(
    frequency: np.ndarray, threshold: float
) -> np.ndarray:
    rates = np.ones(len(frequency), dtype=np.float64)
    if threshold <= 0:
        return rates
    scaled = float(frequency.sum()) * threshold
    present = frequency > 0
    rates[present] = np.minimum(
        (
            np.sqrt(frequency[present] / scaled) + 1.0
        )
        * scaled
        / frequency[present],
        1.0,
    )
    return rates


def noise_distribution(frequency: np.ndarray) -> torch.Tensor:
    weights = np.asarray(frequency, dtype=np.float64) ** 0.75
    if not np.any(weights > 0):
        weights[:] = 1.0
    weights /= weights.sum()
    return torch.from_numpy(weights.astype(np.float32))


def epoch_batches(
    corpus: ReviewCorpus,
    order: np.ndarray,
    keep_rates: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    raw_start: int,
):
    user_buffer = np.empty(0, dtype=np.int64)
    product_buffer = np.empty(0, dtype=np.int64)
    word_buffer = np.empty(0, dtype=np.int64)
    raw_seen = raw_start
    for review in order:
        start, end = corpus.offsets[review : review + 2]
        review_words = corpus.words[start:end]
        raw_seen += len(review_words)
        keep = rng.random(len(review_words)) < keep_rates[review_words]
        selected = review_words[keep]
        if len(selected) == 0:
            continue
        user_buffer = np.concatenate(
            [
                user_buffer,
                np.full(len(selected), corpus.users[review], dtype=np.int64),
            ]
        )
        product_buffer = np.concatenate(
            [
                product_buffer,
                np.full(
                    len(selected), corpus.products[review], dtype=np.int64
                ),
            ]
        )
        word_buffer = np.concatenate([word_buffer, selected])
        while len(word_buffer) >= batch_size:
            yield (
                user_buffer[:batch_size],
                product_buffer[:batch_size],
                word_buffer[:batch_size],
                raw_seen,
            )
            user_buffer = user_buffer[batch_size:]
            product_buffer = product_buffer[batch_size:]
            word_buffer = word_buffer[batch_size:]
    if len(word_buffer):
        yield user_buffer, product_buffer, word_buffer, raw_seen


def sample_relation_tails(
    relation: RelationData,
    products: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    starts = relation.offsets[products]
    lengths = relation.offsets[products + 1] - starts
    valid = lengths > 0
    positions = np.zeros(len(products), dtype=np.int64)
    positions[valid] = starts[valid] + np.floor(
        rng.random(int(valid.sum())) * lengths[valid]
    ).astype(np.int64)
    tails = np.zeros(len(products), dtype=np.int64)
    tails[valid] = relation.tails[positions[valid]]
    return tails, valid.astype(np.float32)


class HeteroEmbed(nn.Module):
    """Original multi-relation translation model with sampled NCE."""

    def __init__(
        self,
        sizes: dict[str, int],
        embedding_size: int,
        relations: tuple[str, ...],
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.entities = nn.ModuleDict(
            {
                name: nn.Embedding(size, embedding_size)
                for name, size in sizes.items()
            }
        )
        relation_names = ("purchase", "write", *relations)
        self.relation_vectors = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(embedding_size))
                for name in relation_names
            }
        )
        tail_sizes = {
            "purchase": sizes["product"],
            "write": sizes["word"],
            **{
                name: sizes[KG_RELATIONS[name][1]]
                for name in relations
            },
        }
        self.tail_bias = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(size))
                for name, size in tail_sizes.items()
            }
        )
        width = 0.5 / embedding_size
        for embedding in self.entities.values():
            nn.init.uniform_(embedding.weight, -width, width)
        for vector in self.relation_vectors.values():
            nn.init.uniform_(vector, -width, width)

    def relation_loss(
        self,
        head_entity: str,
        tail_entity: str,
        relation: str,
        heads: torch.Tensor,
        tails: torch.Tensor,
        weights: torch.Tensor,
        distribution: torch.Tensor,
        negative_samples: int,
        l2_lambda: float,
    ) -> torch.Tensor:
        head = self.entities[head_entity](heads)
        translated = head + self.relation_vectors[relation]
        true_tail = self.entities[tail_entity](tails)
        true_bias = self.tail_bias[relation][tails]
        true_logits = (translated * true_tail).sum(dim=1) + true_bias

        negative_ids = torch.multinomial(
            distribution,
            negative_samples,
            replacement=True,
        )
        negative_tail = self.entities[tail_entity](negative_ids)
        negative_bias = self.tail_bias[relation][negative_ids]
        negative_logits = translated @ negative_tail.T + negative_bias
        per_example = F.softplus(-true_logits) + F.softplus(
            negative_logits
        ).sum(dim=1)
        loss = (weights * per_example).sum()
        if l2_lambda > 0:
            regularization = 0.5 * (
                translated.square().sum()
                + true_tail.square().sum()
                + negative_tail.square().sum()
            )
            loss = loss + l2_lambda * regularization
        return loss

    def recommendation_scores(
        self, users: torch.Tensor, use_bias: bool
    ) -> torch.Tensor:
        translated = (
            self.entities["user"](users)
            + self.relation_vectors["purchase"]
        )
        scores = translated @ self.entities["product"].weight.T
        if use_bias:
            scores = scores + self.tail_bias["purchase"]
        return scores


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_seed(
    args: argparse.Namespace,
    bundle: SplitBundle,
    corpus: ReviewCorpus,
    sizes: dict[str, int],
    relations: dict[str, RelationData],
    seed: int,
    checkpoint_path: Path,
) -> tuple[HeteroEmbed, dict[str, object]]:
    seed_everything(seed)
    device = args.device
    selected_relations = tuple(relations)
    model = HeteroEmbed(
        sizes, args.embedding_size, selected_relations
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    keep_rates = subsampling_rates(
        corpus.word_frequency, args.subsampling_rate
    )
    distributions = {
        "purchase": noise_distribution(
            np.ones(sizes["product"], dtype=np.float64)
        ).to(device),
        "write": noise_distribution(corpus.word_frequency).to(device),
        **{
            name: noise_distribution(relation.frequency).to(device)
            for name, relation in relations.items()
        },
    }
    config = {
        "dataset": bundle.dataset,
        "split": bundle.split,
        "seed": seed,
        "epochs": args.epochs,
        "embedding_size": args.embedding_size,
        "learning_rate": args.learning_rate,
        "batch_size_flag": args.batch_size,
        "effective_token_batch": (
            args.batch_size * (args.negative_samples + 1)
        ),
        "negative_samples": args.negative_samples,
        "subsampling_rate": args.subsampling_rate,
        "l2_lambda": args.l2_lambda,
        "relations": selected_relations,
        "max_reviews": args.max_reviews,
        "train_sha256": sha256(bundle.train_path),
    }
    config_id = digest(config)
    rng = np.random.default_rng(seed)
    start_epoch = 0
    raw_seen = 0
    total_steps = 0
    if checkpoint_path.exists() and not args.restart:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint.get("config_id") != config_id:
            raise RuntimeError(
                f"Checkpoint configuration mismatch: {checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        raw_seen = int(checkpoint["raw_seen"])
        total_steps = int(checkpoint["steps"])
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        print(
            f"Resuming seed={seed} from epoch={start_epoch}",
            flush=True,
        )

    effective_batch = args.batch_size * (args.negative_samples + 1)
    total_raw_words = max(1, corpus.word_count * args.epochs)
    started = time.monotonic()
    stop_early = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        order = rng.permutation(corpus.reviews)
        for users, products, words, current_raw in epoch_batches(
            corpus,
            order,
            keep_rates,
            effective_batch,
            rng,
            raw_seen,
        ):
            raw_seen = current_raw
            factor = max(0.0001, 1.0 - raw_seen / total_raw_words)
            learning_rate = args.learning_rate * factor
            optimizer.param_groups[0]["lr"] = learning_rate
            user_tensor = torch.from_numpy(users).to(device)
            product_tensor = torch.from_numpy(products).to(device)
            word_tensor = torch.from_numpy(words).to(device)
            weights = torch.ones(len(users), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.relation_loss(
                "user",
                "product",
                "purchase",
                user_tensor,
                product_tensor,
                weights,
                distributions["purchase"],
                args.negative_samples,
                args.l2_lambda,
            )
            loss = loss + model.relation_loss(
                "user",
                "word",
                "write",
                user_tensor,
                word_tensor,
                weights,
                distributions["write"],
                args.negative_samples,
                args.l2_lambda,
            )
            loss = loss + model.relation_loss(
                "product",
                "word",
                "write",
                product_tensor,
                word_tensor,
                weights,
                distributions["write"],
                args.negative_samples,
                args.l2_lambda,
            )
            for name, relation in relations.items():
                tails, relation_weights = sample_relation_tails(
                    relation, products, rng
                )
                tail_tensor = torch.from_numpy(tails).to(device)
                weight_tensor = torch.from_numpy(
                    relation_weights
                ).to(device)
                loss = loss + model.relation_loss(
                    "product",
                    relation.tail_entity,
                    name,
                    product_tensor,
                    tail_tensor,
                    weight_tensor,
                    distributions[name],
                    args.negative_samples,
                    args.l2_lambda,
                )
            loss = loss / len(users)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_gradient_norm
            )
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_steps += 1
            total_steps += 1
            if total_steps % args.log_every == 0:
                print(
                    f"seed={seed} epoch={epoch + 1}/{args.epochs} "
                    f"step={total_steps} lr={learning_rate:.6f} "
                    f"loss={epoch_loss / epoch_steps:.5f}",
                    flush=True,
                )
            if args.max_steps and total_steps >= args.max_steps:
                stop_early = True
                break
        completed_epoch = epoch + 1
        atomic_torch(
            checkpoint_path,
            {
                "config_id": config_id,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": completed_epoch,
                "raw_seen": raw_seen,
                "steps": total_steps,
                "numpy_rng_state": rng.bit_generator.state,
            },
        )
        print(
            f"seed={seed} completed epoch {completed_epoch}/{args.epochs} "
            f"mean_loss={epoch_loss / max(epoch_steps, 1):.5f}",
            flush=True,
        )
        if stop_early:
            break
    return model, {
        "config": config,
        "config_id": config_id,
        "steps": total_steps,
        "raw_words_seen": raw_seen,
        "seconds_training": time.monotonic() - started,
        "partial_run": stop_early,
    }


def evaluate(
    model: HeteroEmbed,
    bundle: SplitBundle,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    users = np.asarray(
        [
            user
            for user, targets in enumerate(bundle.targets)
            if len(targets)
        ],
        dtype=np.int64,
    )
    if args.max_eval_users:
        users = users[: args.max_eval_users]
    if len(users) == 0:
        raise RuntimeError("No users have evaluation targets")
    values = {metric: [] for metric in METRICS}
    with torch.no_grad():
        for start in range(0, len(users), args.eval_batch_size):
            batch_users = users[start : start + args.eval_batch_size]
            user_tensor = torch.from_numpy(batch_users).to(args.device)
            scores = model.recommendation_scores(
                user_tensor, args.similarity == "bias_product"
            )
            for row, user in enumerate(batch_users):
                seen_start, seen_end = bundle.train.indptr[
                    user : user + 2
                ]
                seen = bundle.train.indices[seen_start:seen_end]
                scores[row, torch.as_tensor(seen, device=args.device)] = (
                    -torch.inf
                )
            recommendations = (
                torch.topk(scores, k=args.topk, dim=1)
                .indices.cpu()
                .numpy()
            )
            for user, predicted in zip(batch_users, recommendations):
                relevant = set(bundle.targets[int(user)].tolist())
                positions = [
                    rank
                    for rank, item in enumerate(predicted)
                    if int(item) in relevant
                ]
                hits = len(positions)
                dcg = sum(
                    1.0 / math.log2(position + 2)
                    for position in positions
                )
                ideal = min(len(relevant), args.topk)
                idcg = sum(
                    1.0 / math.log2(position + 2)
                    for position in range(ideal)
                )
                values["ndcg"].append(dcg / idcg if idcg else 0.0)
                values["recall"].append(hits / len(relevant))
                values["hr"].append(float(hits > 0))
                values["precision"].append(hits / args.topk)
            print(
                f"Evaluated {min(start + args.eval_batch_size, len(users))}"
                f"/{len(users)} users",
                flush=True,
            )
    return {
        metric: 100.0 * statistics.fmean(metric_values)
        for metric, metric_values in values.items()
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    usable = [row for row in rows if str(row["partial_run"]) == "False"]
    if not usable:
        return []
    summary: dict[str, object] = {
        "model": "HeteroEmbed",
        "seeds": len(usable),
    }
    for metric in METRICS:
        metric_values = [float(row[metric]) for row in usable]
        summary[f"{metric}_mean"] = statistics.fmean(metric_values)
        summary[f"{metric}_std"] = (
            statistics.stdev(metric_values)
            if len(metric_values) > 1
            else 0.0
        )
    return [summary]


def protocol(args: argparse.Namespace) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "paper_name_in_CAFE": "HeteroEmbed",
        "authors_name": "CFKG / KBE4ER",
        "implementation": (
            "PyTorch port of the authors' TensorFlow 1.x released code"
        ),
        "original_repository": (
            "https://github.com/evison/KBE4ExplainableRecommendation"
        ),
        "original_commit": ORIGINAL_COMMIT,
        "objective": (
            "Sampled NCE for purchase, user-write, product-write, and "
            "selected product-knowledge relations"
        ),
        "negative_distribution": (
            "uniform for Purchase; frequency^0.75 for other relations"
        ),
        "optimizer": "SGD with linear learning-rate decay",
        "metrics": ["NDCG@10", "Recall@10", "HR@10", "Precision@10"],
        "evaluation": "full ranking; fit interactions masked",
        "inner_split_policy": (
            "Only review pairs in the inner fit split enter training"
        ),
        "paper_defaults": {
            "learning_rate": 0.5,
            "epochs": 20,
            "batch_size_flag": 64,
            "negative_samples": 5,
            "max_gradient_norm": 5.0,
            "embedding_size_search": [10, 50, 100, 200, 300, 400, 500],
        },
        "run_arguments": dict(vars(args)),
    }


def verify_original_source() -> None:
    if not (ORIGINAL_ROOT / "Recommendation" / "MultiViewEmbedding.py").exists():
        raise RuntimeError(
            "Original HeteroEmbed source is missing from "
            "baselines/vendor/HeteroEmbed-original"
        )
    try:
        import subprocess

        commit = subprocess.check_output(
            ["git", "-C", str(ORIGINAL_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception as error:
        raise RuntimeError("Cannot verify original source commit") from error
    if commit != ORIGINAL_COMMIT:
        raise RuntimeError(
            f"Expected original commit {ORIGINAL_COMMIT}, found {commit}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=["beauty"]
    )
    parser.add_argument("--split", choices=("outer", "inner"), default="outer")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[123, 124, 125, 126, 127]
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--embedding-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--subsampling-rate", type=float, default=1e-4)
    parser.add_argument("--l2-lambda", type=float, default=0.0)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    parser.add_argument(
        "--relations",
        nargs="+",
        choices=tuple(KG_RELATIONS),
        default=list(KG_RELATIONS),
    )
    parser.add_argument(
        "--similarity",
        choices=("product", "bias_product"),
        default="bias_product",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-eval-users", type=int, default=0)
    parser.add_argument("--max-reviews", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--results-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.embedding_size < 1 or args.batch_size < 1:
        raise ValueError("epochs, embedding-size, and batch-size must be positive")
    if args.negative_samples < 1 or args.topk < 1:
        raise ValueError("negative-samples and topk must be positive")
    args.relations = tuple(dict.fromkeys(args.relations))
    args.results_root = args.results_root.resolve()
    args.device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else f"cuda:{args.gpu}"
    )
    verify_original_source()
    plan = protocol(args)
    plan["run_arguments"]["results_root"] = str(args.results_root)
    plan["run_arguments"]["device"] = str(args.device)
    if args.plan_only:
        print(json.dumps(plan, indent=2, default=str))
        return

    for dataset in args.datasets:
        bundle = load_split(dataset, args.split)
        sizes = entity_sizes(bundle.source)
        if sizes["user"] != bundle.n_users or sizes["product"] != bundle.n_items:
            raise RuntimeError(
                f"Source/atomic id mismatch for {dataset}: {sizes}"
            )
        print(
            f"Loading {dataset} {args.split} review corpus",
            flush=True,
        )
        corpus = load_reviews(bundle, sizes["word"], args.max_reviews)
        print(
            f"Loaded {corpus.reviews} reviews and "
            f"{corpus.word_count} word tokens",
            flush=True,
        )
        relations = load_relations(
            bundle, sizes, tuple(args.relations)
        )
        result_dir = (
            args.results_root / dataset / "heteroembed"
        )
        checkpoint_dir = result_dir / "checkpoints"
        atomic_json(result_dir / "protocol.json", plan)
        raw_path = result_dir / "per_seed.csv"
        rows: list[dict[str, object]] = list(read_csv(raw_path))
        completed = {
            int(row["seed"])
            for row in rows
            if str(row["partial_run"]) == "False"
        }
        for seed in args.seeds:
            if seed in completed and not args.restart:
                print(f"Skipping completed seed={seed}", flush=True)
                continue
            model, training = train_seed(
                args,
                bundle,
                corpus,
                sizes,
                relations,
                seed,
                checkpoint_dir / f"seed_{seed}.pt",
            )
            metrics = evaluate(model, bundle, args)
            row: dict[str, object] = {
                "dataset": dataset,
                "model": "HeteroEmbed",
                "implementation": "authors-code-pytorch-port",
                "seed": seed,
                "split": args.split,
                "epochs": args.epochs,
                "embedding_size": args.embedding_size,
                "learning_rate": args.learning_rate,
                "negative_samples": args.negative_samples,
                "relations": "+".join(args.relations),
                "similarity": args.similarity,
                "partial_run": training["partial_run"],
                "ndcg": metrics["ndcg"],
                "recall": metrics["recall"],
                "hr": metrics["hr"],
                "precision": metrics["precision"],
                "seconds_training": training["seconds_training"],
            }
            rows = [item for item in rows if int(item["seed"]) != seed]
            rows.append(row)
            write_csv(raw_path, rows)
            summary = summarize(rows)
            if summary:
                write_csv(result_dir / "summary.csv", summary)
                atomic_json(result_dir / "summary.json", summary)
            atomic_json(
                result_dir / f"seed_{seed}.json",
                {
                    "row": row,
                    "training": training,
                    "metrics": metrics,
                },
            )
            print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
