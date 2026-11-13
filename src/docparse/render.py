"""Page rendering and text-layer extraction.

pypdfium2 is not thread-safe; only call from spawn processes.

Each worker process keeps a small LRU of open ``PdfDocument`` objects. Feed pages in document order
(manifest rows are sorted that way) so each worker opens each PDF about once.
"""

from __future__ import annotations

import base64
import io
import os
from collections import OrderedDict
from typing import Any

import pypdfium2 as pdfium
from PIL import Image

from docparse.osutil import pin_cores

PDF_SUFFIX = ".pdf"
SUPERSAMPLE_DPI = 400

_cache: OrderedDict[str, pdfium.PdfDocument] = OrderedDict()
_cache_size = 4
_cache_misses = 0
_mode = "direct"
_longest_side_px = 2048
_compress_level = 1


def init_render_worker(
    cores: list[int] | None,
    cache_size: int,
    *,
    mode: str = "direct",
    longest_side_px: int = 2048,
    png_compress_level: int = 1,
) -> None:
    """Pool initialiser: single-threaded math libs, CPU pinning, PDF cache, render settings."""
    global _cache_size, _mode, _longest_side_px, _compress_level, _cache_misses
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    pin_cores(cores)
    _cache_size = max(1, cache_size)
    _mode, _longest_side_px, _compress_level = mode, longest_side_px, png_compress_level
    close_cache()
    _cache_misses = 0


def close_cache() -> None:
    while _cache:
        _, doc = _cache.popitem()
        doc.close()


def cache_misses() -> int:
    """Number of PDF opens in this process since init (for tests)."""
    return _cache_misses


def _open_pdf(path: str) -> pdfium.PdfDocument:
    global _cache_misses
    doc = _cache.get(path)
    if doc is not None:
        _cache.move_to_end(path)
        return doc
    _cache_misses += 1
    doc = pdfium.PdfDocument(path)
    _cache[path] = doc
    while len(_cache) > _cache_size:
        _, old = _cache.popitem(last=False)
        old.close()
    return doc


def _fit_longest(img: Image.Image, longest_side_px: int) -> Image.Image:
    if max(img.size) == longest_side_px:
        return img
    k = longest_side_px / max(img.size)
    new = (max(1, round(img.width * k)), max(1, round(img.height * k)))
    # make the longest side exactly the target despite rounding
    new = (longest_side_px, new[1]) if img.width >= img.height else (new[0], longest_side_px)
    return img.resize(new, Image.Resampling.LANCZOS)


def _png(img: Image.Image, compress_level: int) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()


def _image_frame(path: str, page_no: int) -> Image.Image:
    with Image.open(path) as im:
        im.seek(page_no - 1)
        return im.copy()


def render_page_png(
    path: str,
    page_no: int,
    mode: str,
    longest_side_px: int,
    compress_level: int | None = None,
) -> bytes:
    """Render page ``page_no`` (1-based) so its longest side is exactly ``longest_side_px``.

    ``direct``: scale chosen so the render lands on the target. ``supersample``: render at 400 DPI,
    then Lanczos down.
    """
    level = _compress_level if compress_level is None else compress_level
    if not path.lower().endswith(PDF_SUFFIX):
        return _png(_fit_longest(_image_frame(path, page_no), longest_side_px), level)
    doc = _open_pdf(path)
    page = doc[page_no - 1]
    try:
        w_pt, h_pt = page.get_size()
        if mode == "supersample":
            scale = SUPERSAMPLE_DPI / 72
        elif mode == "direct":
            scale = longest_side_px / max(w_pt, h_pt)
        else:
            raise ValueError(f"unknown render mode {mode!r}")
        bitmap = page.render(scale=scale)
        try:
            img = bitmap.to_pil()
        finally:
            bitmap.close()
    finally:
        page.close()
    return _png(_fit_longest(img, longest_side_px), level)


def text_layer(path: str, page_no: int) -> tuple[str, float, float]:
    """Return (text, width_pt, height_pt). Images have no text layer; size is in pixels."""
    if not path.lower().endswith(PDF_SUFFIX):
        with Image.open(path) as im:
            return "", float(im.width), float(im.height)
    doc = _open_pdf(path)
    page = doc[page_no - 1]
    try:
        w_pt, h_pt = page.get_size()
        tp = page.get_textpage()
        try:
            text = tp.get_text_bounded()
        finally:
            tp.close()
    finally:
        page.close()
    return text, float(w_pt), float(h_pt)


def render_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add ``png_b64`` to a manifest page row."""
    png = render_page_png(row["render_path"], int(row["page_no"]), _mode, _longest_side_px)
    return {**row, "png_b64": base64.b64encode(png).decode("ascii")}


def render_row_with_stats(row: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """Test helper: the rendered row plus (pid, cache misses so far) of the worker."""
    return render_row(row), os.getpid(), _cache_misses
