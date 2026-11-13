"""Assembly data models (ARCHITECTURE 6.5) and the pass-1-only page selector.

Coordinate convention (ADR-022): ``bbox_norm`` is the model's 0-1000 box divided by 1000, origin
top-left, y down. ``bbox_pt`` is in PDF user space, origin bottom-left, as
[left, bottom, right, top].
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BlockType = Literal[
    "text",
    "title",
    "list",
    "header",
    "footer",
    "page_number",
    "footnote",
    "caption",
    "formula",
    "code",
    "table",
    "image",
    "chart",
    "header_image",
    "footer_image",
    "aside_text",
    "sheet_table",
    "email_body",
]
Source = Literal["vlm_fast", "vlm_accurate", "textlayer", "native"]
Quality = Literal["ok", "warn", "low"]

Box = tuple[float, float, float, float]


class Block(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    doc_id: str
    page_id: str
    page_no: int
    seq: int
    type: BlockType
    bbox_norm: Box | None = None
    bbox_pt: Box | None = None
    text: str = ""
    markdown: str = ""
    html: str | None = None
    continues_prev: bool = False
    merged_from: list[str] = Field(default_factory=list)
    asset_path: str | None = None
    alt: str | None = None
    section_path: list[str] = Field(default_factory=list)
    native_ref: str | None = None
    source: Source = "vlm_fast"
    quality: Quality = "ok"
    qa_flags: list[str] = Field(default_factory=list)


class Cell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    origin: tuple[int, int] | None = None  # set on spanned filler cells: [row, col] of the origin


class TableGrid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_rows: int
    n_cols: int
    header_rows: int
    cells: list[list[Cell]]


@dataclass(frozen=True)
class Selected:
    page_id: str
    raw: str
    source: str
    quality: str
    qa_flags: tuple[str, ...]


def select_pass1_only(
    raw_dir: Path, page_ids: Collection[str] | None = None
) -> dict[str, Selected]:
    """Stand-in selector (P6 replaces it): every successful pass-1 record is ``vlm_fast``/``ok``."""
    from saturate import read_output

    if not raw_dir.exists():
        return {}
    wanted = set(page_ids) if page_ids is not None else None
    out: dict[str, Selected] = {}
    for pid, row in read_output(str(raw_dir)):
        if wanted is not None and pid not in wanted:
            continue
        out[pid] = Selected(pid, row.get("raw") or "", "vlm_fast", "ok", ())
    return out
