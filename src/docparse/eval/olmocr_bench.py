"""olmOCR-Bench runner: score ``document.md`` (or raw pass output) against the benchmark tests."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from docparse.eval.scorer import run_test
from docparse.vlm.grounding import parse_grounding

log = logging.getLogger(__name__)

OMITTED = {"header", "footer", "page_number"}
CATEGORIES = [
    "arxiv_math",
    "old_scans_math",
    "table_tests",
    "old_scans",
    "headers_footers",
    "multi_column",
    "long_tiny_text",
]
PUBLISHED_OVERALL = 85.5


def load_tests(bench_dir: Path) -> list[dict[str, Any]]:
    """All unit tests, each tagged with its ``category`` (the jsonl file stem)."""
    out: list[dict[str, Any]] = []
    for f in sorted(bench_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append({**json.loads(line), "category": f.stem})
    return out


def pdf_to_page_id(data_root: Path, bench_dir: Path) -> dict[str, str]:
    """``arxiv_math/x.pdf`` (jsonl ``pdf`` field) -> page_id of its first page."""
    docs = pl.read_parquet(data_root / "manifest" / "docs.parquet").select(["doc_id", "paths"])
    out: dict[str, str] = {}
    marker = f"{bench_dir.name}/pdfs/"
    for doc_id, paths in docs.iter_rows():
        for p in paths:
            if "pdfs/" in p:
                out[p.split("pdfs/", 1)[1]] = f"{doc_id}:00001"
            elif marker in p:
                out[p.split(marker, 1)[1]] = f"{doc_id}:00001"
    return out


def raw_to_markdown(raw: str) -> str:
    """Page text straight from a raw grounding output (no assembly); headers/footers omitted."""
    blocks, _ = parse_grounding(raw)
    return "\n\n".join(b.content for b in blocks if b.label not in OMITTED)


def read_raw_pages(out_root: Path, pass_no: int) -> dict[str, str]:
    from saturate import read_output

    d = out_root / "raw" / f"pass{pass_no}"
    if not d.exists():
        return {}
    return {pid: (row.get("raw") or "") for pid, row in read_output(str(d))}


def document_texts(out_root: Path, page_ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pid in page_ids:
        p = out_root / "docs" / pid.split(":")[0] / "document.md"
        if p.exists():
            out[pid] = p.read_text(encoding="utf-8")
    return out


def score(
    tests: list[dict[str, Any]], pdf_pages: dict[str, str], texts: dict[str, str]
) -> pl.DataFrame:
    """One row per test: id, category, type, page_id, passed. Pages without text fail."""
    rows = []
    for t in tests:
        pid = pdf_pages.get(t["pdf"])
        doc = texts.get(pid or "")
        ok = bool(doc is not None and run_test(t, doc))
        rows.append(
            {
                "id": t["id"],
                "category": t["category"],
                "type": t["type"],
                "page_id": pid,
                "has_text": doc is not None,
                "passed": ok,
            }
        )
    return pl.DataFrame(rows)


def _rate(df: pl.DataFrame) -> float:
    return float(df.select(pl.col("passed").cast(pl.Float64).mean()).item())


def summarise(df: pl.DataFrame, only_present: bool = True) -> dict[str, Any]:
    """Per-category pass rate (%), micro and category-macro overall, math split from the rest."""
    if only_present:
        df = df.filter(pl.col("has_text"))
    per = {c: 100 * _rate(g) for (c,), g in df.group_by("category") if len(g)}
    counts = {c: len(g) for (c,), g in df.group_by("category")}
    macro = sum(per.values()) / len(per) if per else 0.0
    non_math = [v for k, v in per.items() if k not in {"arxiv_math", "old_scans_math"}]
    return {
        "per_category": per,
        "counts": counts,
        "macro": macro,
        "micro": 100 * _rate(df) if len(df) else 0.0,
        "macro_without_math": sum(non_math) / len(non_math) if non_math else 0.0,
    }


def page_pass_rates(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("page_id").is_not_null())
        .group_by("page_id")
        .agg(
            pass_rate=pl.col("passed").mean(), n_tests=pl.len(), category=pl.col("category").first()
        )
    )


def report_markdown(label: str, s: dict[str, Any], extra: str = "") -> str:
    lines = [
        f"# olmOCR-Bench — {label}",
        "",
        f"Published overall: {PUBLISHED_OVERALL}. Math here is a **proxy** (ADR-024), so compare "
        "`macro without math` with the published non-math categories.",
        "",
        "| category | tests | pass % |",
        "| --- | ---: | ---: |",
    ]
    by_cat: dict[str, float] = defaultdict(float, s["per_category"])
    for c in CATEGORIES:
        if c in by_cat:
            lines.append(f"| {c} | {s['counts'][c]} | {by_cat[c]:.1f} |")
    lines += [
        "",
        f"- macro (all categories): **{s['macro']:.1f}**",
        f"- macro without math: **{s['macro_without_math']:.1f}**",
        f"- micro (all tests): {s['micro']:.1f}",
        "",
        extra,
    ]
    return "\n".join(lines)
