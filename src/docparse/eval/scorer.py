"""Local re-implementation of the olmOCR-Bench unit tests (ADR-024).

The official scorer (``olmocr.bench``) pulls in torch/KaTeX/playwright; this one is pure Python.
``present``/``absent``/``order``/``table``/``baseline`` follow the official semantics (fuzzy match with
``max_diffs`` edits). ``math`` is a **proxy**: the target LaTeX must appear, after normalisation, in
the document's math spans (the official test renders both sides with KaTeX). Math scores therefore
differ from the published numbers and are reported separately.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from dataclasses import dataclass
from typing import Any

import lxml.html
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import partial_ratio_alignment

_TAGS = re.compile(r"<[^>]+>")
_MD = re.compile(r"[*_`#>]+|!\[[^\]]*\]\([^)]*\)")
_WS = re.compile(r"\s+")
_MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)
_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_PIPE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_LATEX_NOISE = re.compile(
    r"\\(left|right|big|Big|bigg|Bigg|displaystyle|,|;|!|:|quad|qquad)|[\s{}]"
)
_LATEX_ALIAS = {
    r"\mathrm": "",
    r"\mathit": "",
    r"\text": "",
    r"\operatorname": "",
    r"\dfrac": r"\frac",
    r"\tfrac": r"\frac",
    r"\le": r"\leq",
    r"\ge": r"\geq",
}


def normalise(text: str) -> str:
    """NFKC, markup stripped, quotes/dashes unified, whitespace collapsed."""
    t = unicodedata.normalize("NFKC", text)
    t = _TAGS.sub(" ", t)
    t = _MD.sub("", t)
    t = (
        t.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return _WS.sub(" ", t).strip()


def fuzzy_find(needle: str, hay: str, max_diffs: int) -> tuple[int, int] | None:
    """(start, end) of the best match within ``max_diffs`` edits, else None."""
    if not needle:
        return None
    i = hay.find(needle)
    if i >= 0:
        return (i, i + len(needle))
    if max_diffs <= 0 or not hay:
        return None
    al = partial_ratio_alignment(needle, hay)
    if al is None:
        return None
    lo, hi = max(0, al.dest_start - max_diffs), min(len(hay), al.dest_end + max_diffs)
    window = hay[lo:hi]
    best: tuple[int, int, int] | None = None
    n = len(needle)
    for width in range(max(1, n - max_diffs), n + max_diffs + 1):
        for s in range(0, max(1, len(window) - width + 1)):
            d = Levenshtein.distance(needle, window[s : s + width], score_cutoff=max_diffs)
            if d <= max_diffs and (best is None or d < best[0]):
                best = (d, lo + s, lo + s + width)
    return None if best is None else (best[1], best[2])


def _prep(text: str, case_sensitive: bool) -> str:
    t = normalise(text)
    return t if case_sensitive else t.casefold()


def check_present(t: dict[str, Any], doc: str) -> bool:
    cs = bool(t.get("case_sensitive", True))
    hay = _prep(doc, cs)
    if t.get("first_n"):
        hay = hay[: int(t["first_n"])]
    if t.get("last_n"):
        hay = hay[-int(t["last_n"]) :]
    return fuzzy_find(_prep(t["text"], cs), hay, int(t.get("max_diffs") or 0)) is not None


def check_absent(t: dict[str, Any], doc: str) -> bool:
    return not check_present({**t, "case_sensitive": t.get("case_sensitive", False)}, doc)


def check_order(t: dict[str, Any], doc: str) -> bool:
    hay = normalise(doc)
    md = int(t.get("max_diffs") or 0)
    a = fuzzy_find(normalise(t["before"]), hay, md)
    b = fuzzy_find(normalise(t["after"]), hay, md)
    return a is not None and b is not None and a[0] < b[0]


def tables_of(doc: str) -> list[list[list[str]]]:
    """All tables (HTML and pipe) as grids of cell text. Spans are repeated into covered cells."""
    out: list[list[list[str]]] = []
    for m in re.finditer(r"<table.*?</table>", doc, re.DOTALL | re.IGNORECASE):
        try:
            root = lxml.html.fromstring(m.group(0))
        except (ValueError, lxml.etree.ParserError):
            continue
        tbl = root if root.tag == "table" else root.find(".//table")
        if tbl is None:
            continue
        grid = _html_grid(tbl)
        if grid:
            out.append(grid)
    stripped = re.sub(r"<table.*?</table>", "\n", doc, flags=re.DOTALL | re.IGNORECASE)
    rows: list[list[str]] = []
    for line in stripped.splitlines() + [""]:
        if _PIPE_ROW.match(line):
            if not _PIPE_SEP.match(line):
                rows.append([normalise(c) for c in line.strip().strip("|").split("|")])
        elif rows:
            out.append(rows)
            rows = []
    return out


def _html_grid(tbl: lxml.html.HtmlElement) -> list[list[str]]:
    grid: list[list[str]] = []
    carry: dict[int, tuple[int, str]] = {}  # column -> (rows still covered, text)
    for tr in tbl.xpath(".//tr"):
        row: list[str] = []
        for cell in [*tr.xpath("./td | ./th"), None]:
            while len(row) in carry and carry[len(row)][0] > 0:
                n, txt = carry[len(row)]
                carry[len(row)] = (n - 1, txt)
                row.append(txt)
            if cell is None:
                break
            txt = normalise(cell.text_content())
            rs = _span(cell.get("rowspan"))
            for _ in range(_span(cell.get("colspan"))):
                if rs > 1:
                    carry[len(row)] = (rs - 1, txt)
                row.append(txt)
        grid.append(row)
    return grid


def _span(v: str | None) -> int:
    try:
        return max(1, int(v or 1))
    except ValueError:
        return 1


def _same(a: str, b: str, max_diffs: int) -> bool:
    a, b = a.casefold(), b.casefold()
    return a == b or (
        max_diffs > 0 and Levenshtein.distance(a, b, score_cutoff=max_diffs) <= max_diffs
    )


def check_table(t: dict[str, Any], doc: str) -> bool:
    md = int(t.get("max_diffs") or 0)
    cell = normalise(t["cell"])
    for grid in tables_of(doc):
        for r, row in enumerate(grid):
            for c, txt in enumerate(row):
                if not _same(txt, cell, md):
                    continue
                if _relations_hold(t, grid, r, c, md):
                    return True
    return False


def _relations_hold(t: dict[str, Any], grid: list[list[str]], r: int, c: int, md: int) -> bool:
    def at(rr: int, cc: int) -> str | None:
        if 0 <= rr < len(grid) and 0 <= cc < len(grid[rr]):
            return grid[rr][cc]
        return None

    for key, (dr, dc) in {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}.items():
        want = t.get(key)
        if want:
            got = at(r + dr, c + dc)
            if got is None or not _same(got, normalise(want), md):
                return False
    if t.get("top_heading") and not _heading(grid, r, c, normalise(t["top_heading"]), md, True):
        return False
    return not (
        t.get("left_heading") and not _heading(grid, r, c, normalise(t["left_heading"]), md, False)
    )


def _heading(grid: list[list[str]], r: int, c: int, want: str, md: int, top: bool) -> bool:
    cells = [grid[i][c] for i in range(r) if c < len(grid[i])] if top else grid[r][:c]
    return any(_same(x, want, md) for x in cells)


def norm_latex(s: str) -> str:
    for k, v in _LATEX_ALIAS.items():
        s = re.sub(re.escape(k) + r"(?![A-Za-z])", v.replace("\\", "\\\\"), s)
    return _LATEX_NOISE.sub("", s)


def math_spans(doc: str) -> list[str]:
    return [norm_latex(next(g for g in m.groups() if g is not None)) for m in _MATH.finditer(doc)]


def check_math(t: dict[str, Any], doc: str) -> bool:
    target = norm_latex(t["math"])
    if not target:
        return False
    return any(target in span for span in math_spans(doc))


def check_baseline(t: dict[str, Any], doc: str) -> bool:
    """Non-empty, and no decoder loop (zlib ratio < 0.12 on long text, as the official check)."""
    text = doc.strip()
    if not text:
        return False
    data = text.encode("utf-8")
    return not (len(data) > 200 and len(zlib.compress(data)) / len(data) < 0.12)


TESTS = {
    "present": check_present,
    "absent": check_absent,
    "order": check_order,
    "table": check_table,
    "math": check_math,
    "baseline": check_baseline,
}


@dataclass(frozen=True)
class CaseResult:
    id: str
    pdf: str
    type: str
    passed: bool


def run_test(t: dict[str, Any], doc: str) -> bool:
    fn = TESTS.get(t["type"])
    return bool(fn(t, doc)) if fn else False
