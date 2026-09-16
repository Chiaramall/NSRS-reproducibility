"""Download stable MovieLens releases from the official GroupLens host."""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional


URLS = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"unsafe path in archive: {member.filename}")
    archive.extractall(destination)


def download_dataset(name: str, root: Path, force: bool = False) -> Path:
    if name not in URLS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(URLS)}")
    root = Path(root)
    dataset_dir = root / name
    marker = dataset_dir / ("u.data" if name == "ml-100k" else "ratings.dat")
    if marker.exists() and not force:
        return dataset_dir

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f"{name}.zip"
    temporary = archive_path.with_suffix(".zip.part")
    with urllib.request.urlopen(URLS[name]) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        _safe_extract(archive, root)
    if not marker.exists():
        raise RuntimeError(f"download completed but expected file is missing: {marker}")
    return dataset_dir


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(URLS), default=sorted(URLS))
    parser.add_argument("--root", default="./data/movielens")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for dataset in args.datasets:
        path = download_dataset(dataset, Path(args.root), force=args.force)
        print(f"{dataset}: {path}")


if __name__ == "__main__":
    main()
