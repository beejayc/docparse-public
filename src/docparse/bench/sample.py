"""Fixed benchmark sample: the same pages, in the same shuffled order, for every profile."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import polars as pl

DEFAULT_MIX = {"digital": 0.70, "scanned": 0.15, "image": 0.15}  # share of pages per class


def classify(detected_type: str, scanned: bool) -> str:
    if detected_type != "pdf":
        return "image"
    return "scanned" if scanned else "digital"


def build_sample(
    roots: list[Path],
    out_root: Path,
    n: int = 2000,
    seed: int = 7,
    mix: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Stratified sample of ``n`` pages over the manifests in ``roots`` -> ``out_root/manifest``.

    Pages are shuffled with ``seed``, so any prefix is itself a random sample. Classes short of
    their quota are topped up from the others. The result is cached: an existing sample with the
    same ``n``/``seed``/``mix`` is reused unchanged.
    """
    mix = mix or DEFAULT_MIX
    meta_path = out_root / "sample.json"
    want = {"n": n, "seed": seed, "mix": mix, "roots": [str(r) for r in roots]}
    if meta_path.exists() and json.loads(meta_path.read_text())["request"] == want:
        return dict(json.loads(meta_path.read_text()))
    docs = pl.concat([pl.read_parquet(r / "manifest" / "docs.parquet") for r in roots])
    pages = pl.concat([pl.read_parquet(r / "manifest" / "pages.parquet") for r in roots])
    kinds = docs.select(["doc_id", "detected_type"])
    pages = pages.filter(pl.col("status") == "pending").join(kinds, on="doc_id", how="left")
    pages = pages.with_columns(
        cls=pl.struct(["detected_type", "scanned"]).map_elements(
            lambda s: classify(s["detected_type"], bool(s["scanned"])), return_dtype=pl.String
        )
    )
    rng = random.Random(seed)
    by_cls = {c: sorted(g["page_id"].to_list()) for (c,), g in pages.group_by("cls")}
    for c in sorted(by_cls):
        rng.shuffle(by_cls[c])
    take: dict[str, list[str]] = {}
    for c, share in mix.items():
        take[c] = by_cls.get(c, [])[: round(n * share)]
    chosen = {p for ids in take.values() for p in ids}
    short = n - len(chosen)
    if short > 0:  # top up from whatever is left, in a fixed order
        rest = sorted(set(pages["page_id"].to_list()) - chosen)
        rng.shuffle(rest)
        chosen |= set(rest[:short])
    order = sorted(chosen)
    rng.shuffle(order)
    rank = {p: i for i, p in enumerate(order)}
    sel = (
        pages.filter(pl.col("page_id").is_in(pl.Series(order).implode()))
        .with_columns(_r=pl.col("page_id").replace_strict(rank, return_dtype=pl.Int64))
        .sort("_r")
    )
    out = out_root / "manifest"
    out.mkdir(parents=True, exist_ok=True)
    keep = sel.drop(["_r", "detected_type", "cls"])
    keep.write_parquet(out / "pages.parquet")
    docs.filter(pl.col("doc_id").is_in(keep["doc_id"].unique().implode())).write_parquet(
        out / "docs.parquet"
    )
    comp = {c: int(len(g)) for (c,), g in sel.group_by("cls")}
    meta = {"request": want, "pages": len(sel), "composition": comp}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
