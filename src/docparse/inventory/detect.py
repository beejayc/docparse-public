"""File type detection: magic bytes first, extension fallback (DESIGN D11)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import puremagic

_EXT_TO_TYPE = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".webp": "image",
    ".gif": "image",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".csv": "csv",
    ".eml": "eml",
    ".msg": "msg",
    ".zip": "zip",
}

_OOXML_MARKERS = {"word/": "docx", "ppt/": "pptx", "xl/": "xlsx"}


def _ooxml_type(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return None
    if "[Content_Types].xml" not in names:
        return "zip"
    for prefix, kind in _OOXML_MARKERS.items():
        if any(n.startswith(prefix) for n in names):
            return kind
    return "zip"


def detect_type(path: Path) -> str:
    """Return one of pdf, image, docx, pptx, xlsx, csv, eml, msg, zip, unknown."""
    ext: str | None
    try:
        ext = puremagic.from_file(str(path)).lower()
    except (puremagic.PureError, OSError):
        ext = None
    if ext in (".zip", ".docx", ".pptx", ".xlsx"):
        return _ooxml_type(path) or "unknown"
    if ext in _EXT_TO_TYPE:
        return _EXT_TO_TYPE[ext]
    return _EXT_TO_TYPE.get(path.suffix.lower(), "unknown")
