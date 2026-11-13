"""Evaluation entry points used by ``docparse eval``."""

from __future__ import annotations

import statistics
from pathlib import Path

import polars as pl

from config.settings import Settings
from docparse.eval import calibrate as cal
from docparse.eval import olmocr_bench as ob
from docparse.ids import run_id as make_run_id
from docparse.qa.checks import QAConfig
from docparse.qa.store import read_qa

GERMAN_GATE = 0.95
RECALL_GRID = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 .. 0.95
PRECISION_GRID = RECALL_GRID


def _reports(data_root: Path) -> Path:
    d = data_root / "out" / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _texts(data_root: Path, source: str, page_ids: set[str]) -> dict[str, str]:
    out = data_root / "out"
    if source == "document":
        return ob.document_texts(out, page_ids)
    n = 1 if source == "pass1" else 2
    return {pid: ob.raw_to_markdown(r) for pid, r in ob.read_raw_pages(out, n).items()}


def run_suite(
    suite: str, data_root: Path, bench_dir: Path, source: str, pdf_dir: Path | None
) -> str:
    if suite == "olmocr":
        return _olmocr(data_root, bench_dir, source)
    if suite == "omnidoc":
        return _omnidoc(data_root, bench_dir.parent.parent / "OmniDocBench" / "OmniDocBench.json")
    if suite == "calibrate":
        return _calibrate(data_root, bench_dir)
    if suite == "german":
        return _german(data_root)
    raise SystemExit(f"unknown suite {suite!r} (olmocr | omnidoc | calibrate | german)")


def _olmocr(data_root: Path, bench_dir: Path, source: str) -> str:
    tests = ob.load_tests(bench_dir)
    pdf_pages = ob.pdf_to_page_id(data_root, bench_dir)
    texts = _texts(data_root, source, set(pdf_pages.values()))
    df = ob.score(tests, pdf_pages, texts)
    s = ob.summarise(df)
    missing = int((~df["has_text"]).sum())
    extra = f"- source: `{source}`; tests without output (not scored): {missing} of {len(df)}"
    rid = make_run_id("eval")
    path = _reports(data_root) / f"eval-olmocr-{rid}.md"
    path.write_text(ob.report_markdown(source, s, extra), encoding="utf-8")
    return f"{path}  macro={s['macro']:.1f} macro_no_math={s['macro_without_math']:.1f}"


def _calibrate(data_root: Path, bench_dir: Path) -> str:
    tests = ob.load_tests(bench_dir)
    pdf_pages = ob.pdf_to_page_id(data_root, bench_dir)
    raw_rows = _raw_rows(data_root)
    texts = {pid: ob.raw_to_markdown(str(r.get("raw") or "")) for pid, r in raw_rows.items()}
    df = ob.score(tests, pdf_pages, texts)
    pr = ob.page_pass_rates(df.filter(pl.col("has_text")))
    rates = {
        p: (r, c) for p, r, c in zip(pr["page_id"], pr["pass_rate"], pr["category"], strict=True)
    }
    cfg = QAConfig.from_settings(Settings())
    feats = cal.features(raw_rows, cal.load_pages(data_root), rates, cfg)
    sw = cal.sweep(feats, RECALL_GRID, PRECISION_GRID)
    best = cal.choose(sw)
    current = cal.sweep(feats, [0.85], [0.85]).row(0, named=True)
    top = sw.filter(pl.col("flagged_pct") <= 5).sort("failing_caught_pct", descending=True).head(10)
    n_dig = sum(1 for f in feats if f.recall is not None)
    lines = [
        "# QA threshold calibration",
        "",
        f"pages {len(feats)} (digital {n_dig}); failing = benchmark pass rate <= {cal.BAD_PASS_RATE}, "
        "math categories excluded from 'failing' (proxy scorer).",
        "",
        f"Current 0.85/0.85: flagged {current['flagged_pct']:.1f}%, failing caught "
        f"{current['failing_caught_pct']:.1f}%, flag precision {current['precision_of_flags_pct']:.1f}%.",
        f"Chosen (<=5% flagged): recall_min={best['recall_min']}, precision_min={best['precision_min']}: "
        f"flagged {best['flagged_pct']:.1f}%, caught {best['failing_caught_pct']:.1f}%, "
        f"flag precision {best['precision_of_flags_pct']:.1f}%.",
        "",
        "Top grid points within budget:",
        "",
        "| recall_min | precision_min | flagged % | failing caught % | flag precision % |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in top.iter_rows(named=True):
        lines.append(
            f"| {r['recall_min']} | {r['precision_min']} | {r['flagged_pct']:.1f} | "
            f"{r['failing_caught_pct']:.1f} | {r['precision_of_flags_pct']:.1f} |"
        )
    zl = sorted(f.zlib_ratio for f in feats)
    lines += [
        "",
        f"zlib ratio of raw output: min {zl[0]:.3f}, p1 {zl[len(zl) // 100]:.3f}, median {statistics.median(zl):.3f}",
    ]
    path = _reports(data_root) / "calibration.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sw.write_csv(_reports(data_root) / "calibration-grid.csv")
    return f"{path}\n" + "\n".join(lines[4:7])


def _raw_rows(data_root: Path) -> dict[str, dict[str, object]]:
    from saturate import read_output

    d = data_root / "out" / "raw" / "pass1"
    return {pid: dict(row) for pid, row in read_output(str(d))} if d.exists() else {}


def _german(data_root: Path) -> str:
    q = read_qa(data_root / "out", 1)
    rec = [r["recall"] for r in q.values() if r.get("recall") is not None]
    prec = [r["precision"] for r in q.values() if r.get("precision") is not None]
    if not rec:
        return "no digital pages with agreement scores in out/qa/pass1.parquet"
    mr, mp = statistics.median(rec), statistics.median(prec)
    verdict = "PASS" if min(mr, mp) >= GERMAN_GATE else "FAIL"
    text = (
        f"# German sample gate\n\npages {len(rec)}; median recall {mr:.3f}, median precision {mp:.3f}"
        f"; gate >= {GERMAN_GATE}: **{verdict}**\n"
    )
    path = _reports(data_root) / "eval-german.md"
    path.write_text(text, encoding="utf-8")
    return f"{path}  recall={mr:.3f} precision={mp:.3f} {verdict}"


def _omnidoc(data_root: Path, json_path: Path) -> str:
    from docparse.eval import omnidocbench as od

    docs = pl.read_parquet(data_root / "manifest" / "docs.parquet").select(["doc_id", "paths"])
    by_name = {Path(p).name: d for d, paths in docs.iter_rows() for p in paths}
    rows = []
    for page in od.load_pages(json_path):
        doc_id = by_name.get(page["page_info"]["image_path"])
        md_path = data_root / "out" / "docs" / str(doc_id) / "document.md"
        if doc_id is None or not md_path.exists():
            continue
        rows.append(
            od.score_page(page, md_path.read_text(encoding="utf-8"))
            | {"lang": page["page_info"]["page_attribute"].get("language", "?")}
        )
    if not rows:
        return "no OmniDocBench pages with assembled output"
    te = [float(r["text_edit"]) for r in rows if r["text_edit"] is not None]
    tf = [float(r["table_f1"]) for r in rows if r["table_f1"] is not None]
    lines = [
        "# OmniDocBench (proxy metrics, ADR-024)",
        "",
        f"- pages scored: {len(rows)}",
        f"- text edit distance (lower is better): mean {statistics.mean(te):.3f}, median {statistics.median(te):.3f}",
        f"- table cell-text F1 (TEDS proxy, {len(tf)} pages with tables): mean {statistics.mean(tf) if tf else 0:.3f}",
    ]
    for lang in sorted({str(r["lang"]) for r in rows}):
        v = [
            float(r["text_edit"]) for r in rows if r["lang"] == lang and r["text_edit"] is not None
        ]
        lines.append(f"- text edit, {lang}: mean {statistics.mean(v):.3f} ({len(v)} pages)")
    path = _reports(data_root) / "eval-omnidoc.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"{path}  text_edit={statistics.mean(te):.3f} table_f1={statistics.mean(tf) if tf else 0:.3f}"
