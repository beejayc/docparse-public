"""QA threshold calibration against olmOCR-Bench outcomes (P7 task 5). Pure functions + one loader."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import polars as pl

from docparse.qa.agreement import token_agreement
from docparse.qa.checks import QAConfig, agreement_text
from docparse.vlm.grounding import parse_grounding

MATH_CATEGORIES = {"arxiv_math", "old_scans_math"}
BAD_PASS_RATE = 0.5  # a page "fails" when at most half its benchmark tests pass


@dataclass(frozen=True)
class PageFeatures:
    page_id: str
    category: str
    pass_rate: float
    recall: float | None  # None for scans / no text layer
    precision: float | None
    zlib_ratio: float
    truncated: bool


def features(
    raw_rows: dict[str, dict[str, Any]],
    pages: list[dict[str, Any]],
    pass_rates: dict[str, tuple[float, str]],
    cfg: QAConfig,
) -> list[PageFeatures]:
    out = []
    for p in pages:
        pid = p["page_id"]
        if pid not in raw_rows or pid not in pass_rates:
            continue
        raw = raw_rows[pid].get("raw") or ""
        blocks, _ = parse_grounding(raw)
        layer = p.get("text_layer") or ""
        rec = prec = None
        if not p.get("scanned") and layer.strip():
            rec, prec = token_agreement(
                agreement_text(blocks, cfg),
                layer,
                cfg.letters_min_len if cfg.letters_only else cfg.token_min_len,
                cfg.letters_only,
            )
        data = raw.encode()
        ratio = len(zlib.compress(data)) / len(data) if len(data) > 80 else 1.0
        rate, cat = pass_rates[pid]
        out.append(
            PageFeatures(
                pid, cat, rate, rec, prec, ratio, raw_rows[pid].get("finish_reason") == "length"
            )
        )
    return out


def sweep(
    feats: list[PageFeatures],
    recall_grid: list[float],
    precision_grid: list[float],
    exclude_math: bool = True,
) -> pl.DataFrame:
    """For each (recall_min, precision_min): flagged share of all pages and share of failing pages
    caught. Failing = pass rate <= 0.5 (math categories excluded from "failing" when asked, since
    the math scorer is a proxy)."""
    n = len(feats)
    failing = [
        f
        for f in feats
        if f.pass_rate <= BAD_PASS_RATE and not (exclude_math and f.category in MATH_CATEGORIES)
    ]
    rows = []
    for rm, pm in product(recall_grid, precision_grid):
        flagged = {
            f.page_id
            for f in feats
            if f.recall is not None
            and f.precision is not None
            and (f.recall < rm or f.precision < pm)
        }
        caught = sum(1 for f in failing if f.page_id in flagged)
        rows.append(
            {
                "recall_min": rm,
                "precision_min": pm,
                "flagged_pct": 100 * len(flagged) / n if n else 0.0,
                "failing_caught_pct": 100 * caught / len(failing) if failing else 0.0,
                "precision_of_flags_pct": (100 * caught / len(flagged) if flagged else 0.0),
            }
        )
    return pl.DataFrame(rows)


def choose(df: pl.DataFrame, max_flagged_pct: float = 5.0) -> dict[str, float]:
    """Best catch rate within the flag budget; ties -> higher flag precision, then looser limits."""
    ok = df.filter(pl.col("flagged_pct") <= max_flagged_pct)
    if ok.is_empty():
        ok = df.sort("flagged_pct").head(1)
    best = ok.sort(
        ["failing_caught_pct", "precision_of_flags_pct", "recall_min", "precision_min"],
        descending=[True, True, False, False],
    ).row(0, named=True)
    return {k: float(v) for k, v in best.items()}


def load_pages(data_root: Path) -> list[dict[str, Any]]:
    return pl.read_parquet(data_root / "manifest" / "pages.parquet").to_dicts()
