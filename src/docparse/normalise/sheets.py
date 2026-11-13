"""XLSX and CSV -> one sheet_table block per contiguous row range (native, no OCR)."""

from __future__ import annotations

import csv
import html
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from docparse.normalise.native import make_block

Row = tuple[Any, ...]


def _cell(v: Any) -> str:
    return "" if v is None else str(v)


def _md_cell(v: str) -> str:
    return v.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def table_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(_md_cell(c) for c in padded[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in padded[1:]]
    return "\n".join(lines)


def table_html(rows: list[list[str]]) -> str:
    def tr(r: list[str], tag: str) -> str:
        return "<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in r) + "</tr>"

    return "<table>" + tr(rows[0], "th") + "".join(tr(r, "td") for r in rows[1:]) + "</table>"


def _ranges(rows: Iterable[Row]) -> Iterator[tuple[int, int, int, int, list[list[str]]]]:
    """Yield (r1, c1, r2, c2, cells) for runs of non-empty rows (1-based, inclusive)."""
    start = 0
    block: list[Row] = []

    def flush() -> tuple[int, int, int, int, list[list[str]]] | None:
        if not block:
            return None
        cols = [i for r in block for i, v in enumerate(r) if _cell(v).strip()]
        c1, c2 = min(cols), max(cols)
        cells = [[_cell(v) for v in (list(r) + [None] * (c2 + 1))[c1 : c2 + 1]] for r in block]
        return start, c1 + 1, start + len(block) - 1, c2 + 1, cells

    for idx, row in enumerate(rows, start=1):
        if any(_cell(v).strip() for v in row):
            if not block:
                start = idx
            block.append(row)
        else:
            out = flush()
            block = []
            if out:
                yield out
    out = flush()
    if out:
        yield out


def sheet_blocks(
    doc_id: str, sheet_no: int, name: str, rows: Iterable[Row]
) -> list[dict[str, Any]]:
    blocks = []
    for seq, (r1, c1, r2, c2, cells) in enumerate(_ranges(rows), start=1):
        ref = f"{get_column_letter(c1)}{r1}:{get_column_letter(c2)}{r2}"
        md = table_markdown(cells)
        text = "\n".join("\t".join(r) for r in cells)
        blocks.append(
            make_block(
                doc_id, sheet_no, seq, "sheet_table", text, md, f"{name}!{ref}", table_html(cells)
            )
        )
    return blocks


def xlsx_blocks(doc_id: str, path: Path) -> tuple[list[dict[str, Any]], int]:
    """Blocks for every sheet; page_no is the 1-based sheet index. Returns (blocks, n_sheets)."""
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        blocks: list[dict[str, Any]] = []
        for no, ws in enumerate(wb.worksheets, start=1):
            blocks += sheet_blocks(doc_id, no, ws.title, ws.iter_rows(values_only=True))
        return blocks, len(wb.worksheets)
    finally:
        wb.close()


def csv_blocks(doc_id: str, path: Path) -> tuple[list[dict[str, Any]], int]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        dialect: Any = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = csv.reader(text.splitlines(), dialect)
    return sheet_blocks(doc_id, 1, path.stem, (tuple(r) for r in rows)), 1
