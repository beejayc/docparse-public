"""Parser for the LightOnOCR-3 ``grounding`` output (ARCHITECTURE section 7). Pure, CPU only."""

from __future__ import annotations

import re
from dataclasses import dataclass

LABELS = frozenset(
    "text title list header footer page_number footnote caption formula code table image chart "
    "header_image footer_image aside_text".split()
)

# `+` accepted either inside the brackets (![text+](...)) or right after the parenthesis.
_MARKER = re.compile(
    r"^!\[(?P<label>[A-Za-z_]+)(?P<cont>\+)?\]\("
    r"\s*(?P<x1>\d+)\s*,\s*(?P<y1>\d+)\s*,\s*(?P<x2>\d+)\s*,\s*(?P<y2>\d+)\s*\)(?P<cont2>\+)?",
    re.MULTILINE,
)


@dataclass(frozen=True)
class RawBlock:
    label: str
    cont: bool
    bbox: tuple[int, int, int, int] | None
    content: str


def parse_grounding(raw: str) -> tuple[list[RawBlock], list[str]]:
    """Split raw output into blocks; warnings: parse_error, unknown_label:<x>, bad_bbox."""
    blocks: list[RawBlock] = []
    warnings: list[str] = []
    matches = list(_MARKER.finditer(raw))
    head = raw[: matches[0].start()] if matches else raw
    if head.strip():
        blocks.append(RawBlock("text", False, None, head.strip()))
        warnings.append("parse_error")
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        label = m["label"]
        if label not in LABELS:
            warnings.append(f"unknown_label:{label}")
            label = "text"
        x1, y1, x2, y2 = (int(m[g]) for g in ("x1", "y1", "x2", "y2"))
        if not (0 <= x1 <= x2 <= 1000 and 0 <= y1 <= y2 <= 1000):
            warnings.append("bad_bbox")
        blocks.append(
            RawBlock(
                label=label,
                cont=bool(m["cont"] or m["cont2"]),
                bbox=(x1, y1, x2, y2),
                content=raw[m.end() : end].strip(),
            )
        )
    return blocks, warnings
