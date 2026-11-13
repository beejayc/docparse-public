"""ZIP members -> files on disk (become child docs). Rejects traversal, caps size."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path, PurePosixPath

log = logging.getLogger(__name__)

MAX_DEPTH = 3
MAX_TOTAL_BYTES = 2 * 1024**3


def unsafe_member(name: str) -> bool:
    n = name.replace("\\", "/")
    p = PurePosixPath(n)
    return p.is_absolute() or ".." in p.parts or (len(n) > 1 and n[1] == ":") or "\x00" in n


def extract_zip(
    src: Path, dest: Path, max_total: int = MAX_TOTAL_BYTES
) -> tuple[list[Path], list[str]]:
    """Extract safe members. Returns (files, rejected names). Raises ValueError if over cap."""
    files: list[Path] = []
    rejected: list[str] = []
    with zipfile.ZipFile(src) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if sum(i.file_size for i in infos) > max_total:
            raise ValueError(f"archive exceeds {max_total} bytes uncompressed")
        written = 0
        for info in infos:
            name = info.filename
            if unsafe_member(name):
                log.warning("zip %s: rejected unsafe member %r", src, name)
                rejected.append(name)
                continue
            if name.startswith("__MACOSX/") or PurePosixPath(name).name.startswith("."):
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:  # symlink
                rejected.append(name)
                continue
            target = dest / PurePosixPath(name.replace("\\", "/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as fin, target.open("wb") as fout:
                while chunk := fin.read(1 << 20):
                    written += len(chunk)
                    if written > max_total:
                        raise ValueError(f"archive exceeds {max_total} bytes uncompressed")
                    fout.write(chunk)
            files.append(target)
    return files, rejected
