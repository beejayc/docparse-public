"""Block finalisation (sections, markdown) and document markdown rendering (ARCHITECTURE 6.7)."""

from __future__ import annotations

import re
from collections.abc import Iterable

from docparse.assemble.blocks import IMAGE_LIKE
from docparse.assemble.tables import grid_text, grid_to_pipe, is_nested, is_simple
from docparse.assemble.types import Block, TableGrid

OMITTED = {"header", "footer", "page_number"}
_LEADING_HASH = re.compile(r"^\s*#{1,6}\s*")


def _heading_text(block: Block) -> str:
    """Plain single-line title: markup-stripped ``text``, else the raw content minus '#'."""
    return " ".join(block.text.split()) or " ".join(_LEADING_HASH.sub("", block.markdown).split())


def table_markdown(block: Block, grid: TableGrid | None) -> str:
    if grid is None or not block.html:
        return block.markdown
    if is_simple(grid, is_nested(block.html)):
        return grid_to_pipe(grid)
    return block.html


def finalize_blocks(blocks: list[Block], grids: dict[str, TableGrid]) -> list[Block]:
    """Set ``section_path``, ``markdown`` and table ``text`` on blocks in document order."""
    out: list[Block] = []
    path: list[str] = []
    first_title_done = False
    for b in blocks:
        update: dict[str, object] = {}
        if b.type == "title":
            heading = _heading_text(b)
            level = 1 if not first_title_done else 2
            first_title_done = True
            path = [heading] if level == 1 else [*path[:1], heading]
            update["markdown"] = f"{'#' * level} {heading}"
        elif b.type in {"table", "chart"}:
            grid = grids.get(b.block_id)
            md = table_markdown(b, grid)
            if grid is not None:
                update["text"] = grid_text(grid)
            if b.type == "chart":
                img = f"![](assets/{b.block_id}.png)" if b.asset_path else ""
                md = f"{img}\n\n{md}".strip()
            update["markdown"] = md
        elif b.type in IMAGE_LIKE:
            update["markdown"] = f"![](assets/{b.block_id}.png)" if b.asset_path else ""
        update["section_path"] = list(path)
        out.append(b.model_copy(update=update))
    return out


def render_markdown(
    blocks: Iterable[Block],
    pages: Iterable[int] | None = None,
    attachments: list[tuple[str, str]] | None = None,
) -> str:
    """Deterministic document markdown: page marker, then blocks in (page, seq) order."""
    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        by_page.setdefault(b.page_no, []).append(b)
    page_nos = sorted(set(by_page) | set(pages or ()))
    parts: list[str] = []
    for no in page_nos:
        parts.append(f"<!-- page: {no} -->")
        for b in sorted(by_page.get(no, []), key=lambda x: x.seq):
            if b.type in OMITTED or not b.markdown.strip():
                continue
            parts.append(b.markdown.strip())
    if attachments:
        parts.append(
            "**Attachments**\n\n"
            + "\n".join(f"- [{name}](../{doc}/document.md)" for doc, name in attachments)
        )
    return "\n\n".join(parts) + "\n"
