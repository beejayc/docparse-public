"""Walk a root, hash files in a thread pool (hashing releases the GIL), group duplicates."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from docparse.ids import doc_id_from_digest, hash_file
from docparse.inventory.detect import detect_type

HASH_WORKERS = 8


@dataclass(frozen=True)
class DocRecord:
    doc_id: str
    paths: tuple[str, ...]
    detected_type: str
    size_bytes: int


def iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        out.extend(Path(dirpath) / f for f in sorted(filenames) if not f.startswith("."))
    return out


def _hash_one(path: Path) -> tuple[Path, str, int]:
    return path, hash_file(path), path.stat().st_size


def walk(root: Path, workers: int = HASH_WORKERS) -> list[DocRecord]:
    """Identical bytes -> one doc_id with many paths. Sorted by doc_id."""
    files = iter_files(root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashed = list(pool.map(_hash_one, files))
    groups: dict[str, list[tuple[Path, int]]] = {}
    for path, digest, size in hashed:
        groups.setdefault(doc_id_from_digest(digest), []).append((path, size))
    records = []
    for doc_id, items in sorted(groups.items()):
        paths = tuple(str(p) for p, _ in sorted(items))
        first = Path(paths[0])
        records.append(DocRecord(doc_id, paths, detect_type(first), items[0][1]))
    return records
