"""Per-run report (DESIGN section 8): flags by folder and type, escalation/fix rate, agreement
distribution, 20 worst pages. Reads ``manifest/`` and ``out/qa/``; pure CPU."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from docparse.qa.store import read_qa


def build_report(data_root: Path, run_id: str) -> str:
    out = data_root / "out"
    pages = pl.read_parquet(data_root / "manifest" / "pages.parquet").select(["page_id", "doc_id"])
    docs = pl.read_parquet(data_root / "manifest" / "docs.parquet").select(
        ["doc_id", "paths", "detected_type"]
    )
    q1 = read_qa(out, 1)
    q2 = read_qa(out, 2)
    rows = []
    for pid, r in q1.items():
        rows.append(
            {
                "page_id": pid,
                "flags": list(r["flags"] or []),
                "recall": r.get("recall"),
                "precision": r.get("precision"),
                "escalate": bool(r["escalate"]),
                "fixed": bool(pid in q2 and not q2[pid]["escalate"]),
                "pass2": pid in q2,
            }
        )
    if not rows:
        return f"# Run report {run_id}\n\nNo QA rows found in {out / 'qa'}.\n"
    df = (
        pl.DataFrame(rows)
        .join(pages, on="page_id", how="left")
        .join(docs, on="doc_id", how="left")
        .with_columns(
            folder=pl.col("paths")
            .list.first()
            .str.replace(r"/[^/]*$", "")
            .str.split("/")
            .list.last()
        )
    )
    n = len(df)
    esc = df.filter(pl.col("escalate"))
    lines = [
        f"# Run report {run_id}",
        "",
        f"- pages with pass-1 QA: **{n}**",
        f"- escalated: **{len(esc)}** ({100 * len(esc) / n:.1f}%)",
        f"- re-run in pass 2: {int(df['pass2'].sum())}; fixed (pass 2 clean): "
        f"{int(df['fixed'].sum())} ({100 * int(df['fixed'].sum()) / max(1, len(esc)):.1f}% of escalated)",
        "",
        "## Flag share",
        "",
        "| flag | pages | % |",
        "| --- | ---: | ---: |",
    ]
    flat = df.select(pl.col("flags").explode().alias("flag")).drop_nulls()
    for flag, cnt in flat.group_by("flag").len().sort("len", descending=True).iter_rows():
        lines.append(f"| {flag} | {cnt} | {100 * cnt / n:.1f} |")
    for title, col in (("folder", "folder"), ("type", "detected_type")):
        lines += [
            "",
            f"## Escalation by {title}",
            "",
            f"| {title} | pages | escalated % |",
            "| --- | ---: | ---: |",
        ]
        g = df.group_by(col).agg(n=pl.len(), e=pl.col("escalate").sum()).sort("n", descending=True)
        for k, cnt, e in g.iter_rows():
            lines.append(f"| {k} | {cnt} | {100 * e / cnt:.1f} |")
    for name in ("recall", "precision"):
        s = df[name].drop_nulls()
        if len(s):
            qs = [s.quantile(q) for q in (0.05, 0.25, 0.5, 0.75)]
            lines += [
                "",
                f"## {name} distribution (digital pages: {len(s)})",
                "",
                f"p5 {qs[0]:.2f} · p25 {qs[1]:.2f} · median {qs[2]:.2f} · p75 {qs[3]:.2f}",
            ]
    worst = (
        df.with_columns(
            score=pl.min_horizontal(
                pl.col("recall").fill_null(1.0), pl.col("precision").fill_null(1.0)
            )
        )
        .sort("score")
        .head(20)
    )
    lines += [
        "",
        "## 20 worst pages",
        "",
        "| page | score | flags | doc |",
        "| --- | ---: | --- | --- |",
    ]
    for r in worst.iter_rows(named=True):
        doc = r["doc_id"]
        lines.append(
            f"| {r['page_id']} | {r['score']:.2f} | {', '.join(r['flags'])} | [md](../docs/{doc}/document.md) |"
        )
    return "\n".join(lines) + "\n"
