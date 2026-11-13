"""Crop image-like blocks from a fresh render. Runs in a spawn process (pypdfium2)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image

from docparse.assemble.blocks import IMAGE_LIKE
from docparse.assemble.types import Block
from docparse.render import render_page_png

PAD_PX = 8


def crop_box(
    norm: tuple[float, float, float, float], size: tuple[int, int], pad: int = PAD_PX
) -> tuple[int, int, int, int]:
    w, h = size
    x1, y1, x2, y2 = norm
    return (
        max(0, int(x1 * w) - pad),
        max(0, int(y1 * h) - pad),
        min(w, int(x2 * w + 0.999) + pad),
        min(h, int(y2 * h + 0.999) + pad),
    )


def crop_assets(
    blocks: list[Block],
    page_rows: dict[int, dict[str, Any]],
    out_dir: Path,
    mode: str,
    longest_side_px: int,
) -> set[str]:
    """Write ``assets/{block_id}.png`` for image-like blocks; returns the ids that got a crop."""
    wanted: dict[int, list[Block]] = {}
    for b in blocks:
        if b.type in IMAGE_LIKE and b.bbox_norm is not None and b.page_no in page_rows:
            wanted.setdefault(b.page_no, []).append(b)
    done: set[str] = set()
    if not wanted:
        return done
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    for page_no, items in sorted(wanted.items()):
        png = render_page_png(page_rows[page_no]["render_path"], page_no, mode, longest_side_px)
        with Image.open(io.BytesIO(png)) as page:
            for b in items:
                assert b.bbox_norm is not None
                box = crop_box(b.bbox_norm, page.size)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                page.crop(box).save(out_dir / "assets" / f"{b.block_id}.png", compress_level=6)
                done.add(b.block_id)
    return done
