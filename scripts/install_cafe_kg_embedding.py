#!/usr/bin/env python3
"""Install completed PGPR TransE checkpoints where CAFE expects them."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PGPR_TMP = ROOT / "baselines" / "models" / "pgpr" / "tmp"
DATASETS = {
    "cell": {
        "source_dir": "Amazon_Cellphones",
        "destination": ROOT / "data" / "Cellphones_Accessories" / "embedding_des_epoch_30.ckpt",
    },
    "clothing": {
        "source_dir": "Amazon_Clothing",
        "destination": ROOT / "data" / "Clothing" / "embedding_des_epoch_29.ckpt",
    },
    "cd": {
        "source_dir": "Amazon_CDs",
        "destination": ROOT / "data" / "CDs_Vinyl" / "embedding_des_epoch_30.ckpt",
    },
}
REQUIRED_KEYS = {
    "purchase", "mentions", "describe_as", "produced_by", "belongs_to",
    "also_bought", "also_viewed", "bought_together",
}


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a state dictionary: {path}")
    missing = sorted(REQUIRED_KEYS.difference(checkpoint))
    if missing:
        raise RuntimeError(f"Checkpoint {path} is missing keys: {missing}")
    return checkpoint


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def install(dataset: str) -> Path:
    specification = DATASETS[dataset]
    source = (
        PGPR_TMP / str(specification["source_dir"])
        / "paper_transe_e30_seed_123" / "transe_model_sd_epoch_30.ckpt"
    )
    destination = specification["destination"]
    assert isinstance(destination, Path)
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Completed PGPR TransE checkpoint not found: {source}")

    load_checkpoint(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size != source.stat().st_size:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    load_checkpoint(destination)

    provenance = {
        "dataset": dataset,
        "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source.relative_to(ROOT)),
        "destination": str(destination.relative_to(ROOT)),
        "source_seed": 123,
        "source_epoch": 30,
        "size_bytes": source.stat().st_size,
        "note": (
            "Clothing retains the epoch-29 filename expected by the released CAFE configuration; its contents are the completed epoch-30 checkpoint."
            if dataset == "clothing"
            else "PGPR TransE state dictionary reused as CAFE KG initialization."
        ),
    }
    atomic_json(destination.with_suffix(destination.suffix + ".provenance.json"), provenance)
    print(f"CAFE KG embedding ready: {destination.relative_to(ROOT)}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    args = parser.parse_args()
    install(args.dataset)


if __name__ == "__main__":
    main()
