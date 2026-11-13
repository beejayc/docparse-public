"""Label-free QA flags (DESIGN section 4): one pure function per flag, combined by check_page."""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import lxml.html

from config.settings import Settings
from docparse.qa.agreement import strip_math, token_agreement
from docparse.vlm.grounding import RawBlock, parse_grounding

AGREEMENT_EXCLUDED = {"header", "footer", "page_number"}
ESCALATING = (
    "truncated",
    "repetition",
    "empty",
    "parse_error",
    "bad_table",
    "low_recall",
    "low_precision",
)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
MIN_DISTINCT_CHARS = 4  # ADR-023


@dataclass(frozen=True)
class QAConfig:
    zlib_ratio_min: float = 0.12
    repeat_len: int = 40
    repeat_max: int = 20
    empty_chars_max: int = 20
    empty_text_layer_chars: int = 200
    empty_image_std: float = 8.0
    recall_min: float = 0.40
    precision_min: float = 0.40
    agreement_warn_max: float = 0.75
    token_min_len: int = 2
    letters_only: bool = True
    letters_min_len: int = 3

    @classmethod
    def from_settings(cls, s: Settings) -> QAConfig:
        return cls(
            zlib_ratio_min=s.qa_zlib_ratio_min,
            repeat_len=s.qa_repeat_substring_len,
            repeat_max=s.qa_repeat_substring_max,
            empty_chars_max=s.qa_empty_chars_max,
            empty_text_layer_chars=s.qa_empty_text_layer_chars,
            empty_image_std=s.qa_empty_image_std,
            recall_min=s.qa_recall_min,
            precision_min=s.qa_precision_min,
            agreement_warn_max=s.qa_agreement_warn_max,
            token_min_len=s.qa_token_min_len,
            letters_only=s.qa_letters_only,
            letters_min_len=s.qa_letters_min_len,
        )


@dataclass
class QAResult:
    page_id: str
    pass_no: int
    flags: list[str] = field(default_factory=list)
    recall: float | None = None
    precision: float | None = None

    @property
    def escalate(self) -> bool:
        return any(f in ESCALATING for f in self.flags)


def plain_text(blocks: list[RawBlock], exclude: set[str] | None = None) -> str:
    """Markup-stripped text of the blocks, minus excluded labels (ADR-023)."""
    skip = exclude or set()
    parts = []
    for b in blocks:
        if b.label in skip:
            continue
        parts.append(_TAGS.sub(" ", b.content))
    return _WS.sub(" ", " ".join(parts)).strip()


def agreement_text(blocks: list[RawBlock], cfg: QAConfig) -> str:
    """VLM text for agreement. Letters-only mode (ADR-025) keeps header/footer text too, because
    the text layer contains it, and drops formula blocks and ``$..$`` spans."""
    if not cfg.letters_only:
        return plain_text(blocks, AGREEMENT_EXCLUDED)
    return strip_math(plain_text(blocks, {"formula"}))


def check_truncated(finish_reason: str) -> bool:
    return finish_reason == "length"


def check_repetition(text: str, cfg: QAConfig) -> bool:
    """zlib ratio below the minimum, or one 40-char substring repeated more than 6 times."""
    if not text:
        return False
    data = text.encode("utf-8")
    if (
        len(data) >= 2 * cfg.repeat_len
        and len(zlib.compress(data)) / len(data) < cfg.zlib_ratio_min
    ):
        return True
    n = cfg.repeat_len
    if len(text) < n * (cfg.repeat_max + 1):
        return False
    counts = Counter(text[i : i + n] for i in range(len(text) - n + 1))
    return any(c > cfg.repeat_max and len(set(s)) >= MIN_DISTINCT_CHARS for s, c in counts.items())


def check_empty(text: str, layer_chars: int, image_std: float | None, cfg: QAConfig) -> bool:
    """Under 20 chars although the page has a long text layer or visible content."""
    if len(text) >= cfg.empty_chars_max:
        return False
    if layer_chars > cfg.empty_text_layer_chars:
        return True
    return image_std is not None and image_std > cfg.empty_image_std


def table_is_bad(html: str) -> bool:
    """No parseable table, or rows with different widths once row/col spans are expanded."""
    try:
        root = lxml.html.fromstring(html)
    except (ValueError, lxml.etree.ParserError):
        return True
    tbl = root if root.tag == "table" else root.find(".//table")
    if tbl is None:
        return True
    trs = tbl.xpath("./tr | ./thead/tr | ./tbody/tr | ./tfoot/tr")
    if not trs:
        return True
    carry: dict[int, int] = {}  # column -> rows still covered by a rowspan from above
    widths = set()
    for tr in trs:
        col = 0
        width = 0
        for cell in tr.xpath("./td | ./th"):
            while carry.get(col, 0) > 0:
                carry[col] -= 1
                col += 1
                width += 1
            cs = _int(cell.get("colspan"))
            rs = _int(cell.get("rowspan"))
            for k in range(cs):
                if rs > 1:
                    carry[col + k] = rs - 1
            col += cs
            width += cs
        while carry.get(col, 0) > 0:
            carry[col] -= 1
            col += 1
            width += 1
        widths.add(width)
    return len(widths) > 1


def _int(v: str | None) -> int:
    try:
        return max(1, int(v or 1))
    except ValueError:
        return 1


def check_bad_table(blocks: list[RawBlock]) -> bool:
    return any(
        b.label in {"table", "chart"} and "<table" in b.content and table_is_bad(b.content)
        for b in blocks
    ) or any(b.label == "table" and "<table" not in b.content for b in blocks)


def check_page(
    raw_row: dict[str, Any], page_row: dict[str, Any], cfg: QAConfig, pass_no: int = 1
) -> QAResult:
    """All flags for one page. ``raw_row`` has ``raw``/``finish_reason``; ``page_row`` the layer."""
    page_id = str(page_row.get("page_id") or raw_row.get("id") or "")
    res = QAResult(page_id, pass_no)
    raw = raw_row.get("raw") or ""
    blocks, warnings = parse_grounding(raw)
    text = plain_text(blocks, AGREEMENT_EXCLUDED)  # emptiness: any visible text counts
    layer = page_row.get("text_layer") or ""
    layer_chars = int(page_row.get("text_layer_chars") or 0)

    if check_truncated(str(raw_row.get("finish_reason") or "")):
        res.flags.append("truncated")
    if check_repetition(plain_text(blocks), cfg):
        res.flags.append("repetition")
    if check_empty(text, layer_chars, page_row.get("image_std"), cfg):
        res.flags.append("empty")
    if "parse_error" in warnings:
        res.flags.append("parse_error")
    if check_bad_table(blocks):
        res.flags.append("bad_table")
    if not page_row.get("scanned", layer_chars < 20) and layer.strip():
        res.recall, res.precision = token_agreement(
            agreement_text(blocks, cfg),
            layer,
            cfg.letters_min_len if cfg.letters_only else cfg.token_min_len,
            cfg.letters_only,
        )
        if res.recall < cfg.recall_min:
            res.flags.append("low_recall")
        if res.precision < cfg.precision_min:
            res.flags.append("low_precision")
        worst = min(res.recall, res.precision)
        if (
            cfg.recall_min <= res.recall
            and cfg.precision_min <= res.precision
            and (worst < cfg.agreement_warn_max)
        ):
            res.flags.append("low_agreement_warn")
    return res
