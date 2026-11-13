"""Raw grounding output -> ``Block`` records for one page. Pure, CPU only."""

from __future__ import annotations

import re
from typing import Any

from docparse.assemble.types import Block, BlockType, Selected
from docparse.ids import block_id
from docparse.vlm.grounding import RawBlock, parse_grounding

TABLE_LIKE = {"table", "chart"}
IMAGE_LIKE = {"image", "chart", "header_image", "footer_image"}
_TAGS = re.compile(r"<[^>]+>")
_EMPH = re.compile(r"(\*\*|__|\*|(?<!\w)_|_(?!\w))")
_LIST_MARK = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_HEADING = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_WS = re.compile(r"[ \t]+")


def strip_markup(label: str, content: str) -> str:
    """Plain text of a block: HTML tags, emphasis, list/heading markers and $$ wrappers removed."""
    text = content
    if label in TABLE_LIKE:
        text = re.sub(r"</t[dh]>", " | ", text)
        text = re.sub(r"</tr>", "\n", text)
    text = _TAGS.sub("", text)
    if label == "formula":
        text = text.replace("$$", "")
    text = _HEADING.sub("", text)
    text = _LIST_MARK.sub("", text)
    text = _EMPH.sub("", text)
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def bbox_valid(bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    return 0 <= x1 <= x2 <= 1000 and 0 <= y1 <= y2 <= 1000


def bbox_to_norm(bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """Model 0-1000 box -> 0-1. Malformed boxes are sorted and clamped so they stay on the page."""
    x1, x2 = sorted((bbox[0], bbox[2]))
    y1, y2 = sorted((bbox[1], bbox[3]))
    c = [min(1000, max(0, v)) / 1000 for v in (x1, y1, x2, y2)]
    return (c[0], c[1], c[2], c[3])


def norm_to_pt(
    norm: tuple[float, float, float, float], width_pt: float, height_pt: float
) -> tuple[float, float, float, float]:
    """Normalised top-left box -> PDF points [left, bottom, right, top] (bottom-left origin)."""
    x1, y1, x2, y2 = norm
    return (
        round(x1 * width_pt, 3),
        round(height_pt * (1 - y2), 3),
        round(x2 * width_pt, 3),
        round(height_pt * (1 - y1), 3),
    )


def build_blocks(sel: Selected, page_row: dict[str, Any]) -> list[Block]:
    """Blocks of one page in reading order. ``markdown`` is a placeholder until finalisation."""
    raw_blocks, _ = parse_grounding(sel.raw)
    out: list[Block] = []
    for seq, rb in enumerate(raw_blocks, start=1):
        flags = list(sel.qa_flags)  # per-block flags only: page-level warnings stay with QA (P6)
        if rb.bbox is None:
            flags.append("parse_error")
        elif not bbox_valid(rb.bbox):
            flags.append("bad_bbox")
        quality = "warn" if flags != list(sel.qa_flags) and sel.quality == "ok" else sel.quality
        out.append(_one(rb, seq, sel, page_row, flags, quality))
    return out


def _one(
    rb: RawBlock,
    seq: int,
    sel: Selected,
    page_row: dict[str, Any],
    flags: list[str],
    quality: str,
) -> Block:
    norm = bbox_to_norm(rb.bbox) if rb.bbox else None
    pt = None
    if norm is not None and page_row.get("width_pt") and page_row.get("height_pt"):
        pt = norm_to_pt(norm, float(page_row["width_pt"]), float(page_row["height_pt"]))
    label: BlockType = rb.label  # type: ignore[assignment]  # parser guarantees a known label
    text = strip_markup(rb.label, rb.content)
    return Block(
        block_id=block_id(page_row["page_id"], seq),
        doc_id=page_row["doc_id"],
        page_id=page_row["page_id"],
        page_no=int(page_row["page_no"]),
        seq=seq,
        type=label,
        bbox_norm=norm,
        bbox_pt=pt,
        text=text,
        markdown=rb.content,
        html=rb.content if rb.label in TABLE_LIKE and "<table" in rb.content else None,
        continues_prev=rb.cont,
        alt=text if rb.label in {"image", "header_image", "footer_image"} else None,
        source=sel.source,  # type: ignore[arg-type]
        quality=quality,  # type: ignore[arg-type]
        qa_flags=flags,
    )
