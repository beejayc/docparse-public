"""HTML table -> expanded cell grid, back to HTML, and pipe-table rendering. Pure, CPU only."""

from __future__ import annotations

import re
from html import escape
from typing import Any

import lxml.html

from docparse.assemble.types import Cell, TableGrid

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _span(value: str | None, limit: int) -> int:
    try:
        n = int(value or 1)
    except ValueError:
        n = 1
    return max(1, min(n, limit))


def find_table(html: str) -> Any:
    root = lxml.html.fromstring(html)
    if root.tag == "table":
        return root
    tbl = root.find(".//table")
    if tbl is None:
        raise ValueError("no <table> in html")
    return tbl


def is_nested(html: str) -> bool:
    return bool(find_table(html).xpath(".//table"))


def table_grid(html: str) -> TableGrid:
    """Parse ``html`` into a grid with row/col spans expanded (ADR-022).

    Spanned positions repeat the origin cell's text and carry ``origin=(row, col)``.
    """
    tbl = find_table(html)
    trs = tbl.xpath("./tr | ./thead/tr | ./tbody/tr | ./tfoot/tr")
    n_rows = len(trs)
    thead_rows = sum(1 for tr in trs if tr.getparent().tag == "thead")
    placed: dict[tuple[int, int], Cell] = {}
    for r, tr in enumerate(trs):
        c = 0
        for el in tr.xpath("./td | ./th"):
            while (r, c) in placed:
                c += 1
            rs = _span(el.get("rowspan"), n_rows - r)
            cs = _span(el.get("colspan"), 10_000)
            is_h = el.tag == "th"
            text = _clean(el.text_content())
            for dr in range(rs):
                for dc in range(cs):
                    if (r + dr, c + dc) in placed:
                        continue
                    if dr == 0 and dc == 0:
                        placed[(r, c)] = Cell(text=text, row_span=rs, col_span=cs, is_header=is_h)
                    else:
                        placed[(r + dr, c + dc)] = Cell(text=text, is_header=is_h, origin=(r, c))
            c += cs
    n_cols = max((c for _, c in placed), default=-1) + 1
    cells = [[placed.get((r, c)) or Cell(text="") for c in range(n_cols)] for r in range(n_rows)]
    header_rows = thead_rows
    if not header_rows:
        for row in cells:
            if row and all(cell.is_header for cell in row):
                header_rows += 1
            else:
                break
    return TableGrid(n_rows=n_rows, n_cols=n_cols, header_rows=header_rows, cells=cells)


def is_simple(grid: TableGrid, nested: bool = False) -> bool:
    """True when the table can be a pipe table (no spans, no nesting)."""
    if nested:
        return False
    return all(
        cell.row_span == 1 and cell.col_span == 1 and cell.origin is None
        for row in grid.cells
        for cell in row
    )


def grid_to_html(grid: TableGrid) -> str:
    def tr(r: int) -> str:
        parts = []
        for cell in grid.cells[r]:
            if cell.origin is not None:
                continue
            tag = "th" if cell.is_header else "td"
            attrs = ""
            if cell.row_span > 1:
                attrs += f' rowspan="{cell.row_span}"'
            if cell.col_span > 1:
                attrs += f' colspan="{cell.col_span}"'
            parts.append(f"<{tag}{attrs}>{escape(cell.text)}</{tag}>")
        return "<tr>" + "".join(parts) + "</tr>"

    head = "".join(tr(r) for r in range(grid.header_rows))
    body = "".join(tr(r) for r in range(grid.header_rows, grid.n_rows))
    out = "<table>"
    if head:
        out += f"<thead>{head}</thead>"
    return out + f"<tbody>{body}</tbody></table>"


def _esc(text: str) -> str:
    return text.replace("|", "\\|")


def grid_to_pipe(grid: TableGrid) -> str:
    """Pipe table; the first row is the header (also when the source had no header row)."""
    if grid.n_rows == 0 or grid.n_cols == 0:
        return ""
    rows = [[_esc(c.text) for c in row] for row in grid.cells]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * grid.n_cols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def grid_text(grid: TableGrid) -> str:
    return "\n".join(" | ".join(c.text for c in row) for row in grid.cells)


def shift_origins(cells: list[list[Cell]], row_offset: int) -> list[list[Cell]]:
    return [
        [
            c.model_copy(update={"origin": (c.origin[0] + row_offset, c.origin[1])})
            if c.origin is not None
            else c
            for c in row
        ]
        for row in cells
    ]
