"""Cross-page table merge (ARCHITECTURE 6.7, ADR-022)."""

from __future__ import annotations

from docparse.assemble.tables import grid_to_html, shift_origins, table_grid
from docparse.assemble.types import Block, TableGrid

TOP_SKIP = {"header", "header_image", "page_number"}
BOTTOM_SKIP = {"footer", "footer_image", "page_number"}
TABLE_TYPES = {"table", "chart"}


def _same_header(prev: TableGrid, cont: TableGrid) -> int:
    """Number of leading rows of ``cont`` that repeat ``prev``'s header (0 if none)."""
    h = prev.header_rows
    if h == 0 or cont.n_rows < h:
        return 0
    a = [[c.text for c in row] for row in prev.cells[:h]]
    b = [[c.text for c in row] for row in cont.cells[:h]]
    return h if a == b else 0


def merge_grids(prev: TableGrid, cont: TableGrid) -> TableGrid | None:
    if prev.n_cols != cont.n_cols or prev.n_cols == 0:
        return None
    drop = _same_header(prev, cont)
    rest = cont.cells[drop:]
    if any(c.origin is not None and c.origin[0] < drop for row in rest for c in row):
        return None  # a span reaches into the dropped header
    rest = [
        [
            c.model_copy(update={"origin": (c.origin[0] - drop, c.origin[1])}) if c.origin else c
            for c in row
        ]
        for row in rest
    ]
    offset = prev.n_rows
    return TableGrid(
        n_rows=prev.n_rows + len(rest),
        n_cols=prev.n_cols,
        header_rows=prev.header_rows,
        cells=prev.cells + shift_origins(rest, offset),
    )


def grids_for(blocks: list[Block]) -> dict[str, TableGrid]:
    out: dict[str, TableGrid] = {}
    for b in blocks:
        if b.type in TABLE_TYPES and b.html:
            try:
                out[b.block_id] = table_grid(b.html)
            except ValueError:
                continue
    return out


def merge_tables(
    pages: list[list[Block]], grids: dict[str, TableGrid]
) -> tuple[list[Block], dict[str, TableGrid]]:
    """Flatten ``pages`` (one block list per page, in page order) merging continued tables.

    The first block of a merged table survives (html regenerated, ``merged_from`` filled); the
    continuation block is dropped. ``grids`` is updated in place and returned.
    """
    out: list[Block] = []
    for blocks in pages:
        blocks = list(blocks)
        first = next((i for i, b in enumerate(blocks) if b.type not in TOP_SKIP), None)
        if first is not None:
            cont = blocks[first]
            prev_i = next(
                (i for i in range(len(out) - 1, -1, -1) if out[i].type not in BOTTOM_SKIP), None
            )
            if (
                cont.type == "table"
                and cont.continues_prev
                and prev_i is not None
                and out[prev_i].type == "table"
                and out[prev_i].page_no == cont.page_no - 1
                and out[prev_i].block_id in grids
                and cont.block_id in grids
            ):
                prev = out[prev_i]
                merged = merge_grids(grids[prev.block_id], grids[cont.block_id])
                if merged is not None:
                    grids[prev.block_id] = merged
                    out[prev_i] = prev.model_copy(
                        update={
                            "html": grid_to_html(merged),
                            "merged_from": [*prev.merged_from, cont.block_id],
                        }
                    )
                    del blocks[first]
        out.extend(blocks)
    return out, grids
