"""Render seconds per page on one pinned core, per render mode."""

from __future__ import annotations

import multiprocessing
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl

from docparse import render


def _init(core: int, mode: str, side: int) -> None:
    render.init_render_worker([core], 1, mode=mode, longest_side_px=side, png_compress_level=1)


def _time_rows(rows: list[dict[str, Any]]) -> list[float]:
    out = []
    for row in rows:
        t0 = time.perf_counter()
        render.render_row(row)
        out.append(time.perf_counter() - t0)
    return out


def render_s_per_page(
    manifest_root: Path, core: int, mode: str, side: int = 2048, n: int = 100
) -> dict[str, float]:
    """Mean/median/p95 seconds per page for the first ``n`` pages of the sample on ``core``."""
    cols = ["page_id", "doc_id", "render_path", "page_no", "width_pt", "height_pt"]
    rows = pl.read_parquet(manifest_root / "manifest" / "pages.parquet").head(n).select(cols)
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init,
        initargs=(core, mode, side),
    ) as pool:
        times = pool.submit(_time_rows, rows.to_dicts()).result()
    times.sort()
    return {
        "mean_s": round(statistics.mean(times), 4),
        "median_s": round(statistics.median(times), 4),
        "p95_s": round(times[int(0.95 * (len(times) - 1))], 4),
        "pages": float(len(times)),
    }
