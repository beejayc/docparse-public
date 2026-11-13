"""Identifier builders; nothing else builds ids by hand (ARCHITECTURE section 4)."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

CHUNK = 1 << 20  # 1 MiB


def hash_file(path: Path) -> str:
    """Full SHA-256 hex digest of the file bytes, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def doc_id_from_digest(digest: str) -> str:
    return digest[:16]


def doc_id_for(path: Path) -> str:
    return doc_id_from_digest(hash_file(path))


def page_id(doc_id: str, page_no: int) -> str:
    return f"{doc_id}:{page_no:05d}"


def block_id(page_id_: str, seq: int) -> str:
    return f"{page_id_}:{seq:03d}"


def run_id(profile: str, now: datetime | None = None) -> str:
    return f"{(now or datetime.now()):%Y%m%d-%H%M%S}-{profile}"
