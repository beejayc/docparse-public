"""One measured pass-1 run: pages/s, bound_by, tokens/page, GPU utilisation, cost."""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from config.settings import Profile
from docparse.batch.rows import iter_pending_pages
from docparse.batch.saturate_runner import SaturateRunner


@dataclass
class BenchResult:
    tag: str
    pages: int
    workers: int
    cpuset: str
    pages_per_s: float
    bound_by: dict[str, float]
    out_tokens_per_page: float
    gpu_util_pct: float | None
    elapsed_s: float
    errors: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def engine_share(self) -> float:
        return self.bound_by.get("engine", 0.0)


def workers_for(pages_per_s: float, render_s_per_page: float) -> int:
    """DESIGN D/P8 task 3: ceil(pages/s x render s/page x 1.3), at least 1."""
    return max(1, math.ceil(pages_per_s * render_s_per_page * 1.3))


def cost_per_1000(pages_per_s: float, hourly_cost: float) -> float | None:
    return round(hourly_cost / 3600 / pages_per_s * 1000, 4) if pages_per_s > 0 else None


def parse_dmon(text: str) -> float | None:
    """Mean SM utilisation (%) from ``nvidia-smi dmon -s u`` output (the ``sm`` column)."""
    vals: list[float] = []
    col: int | None = None
    for line in text.splitlines():
        parts = line.replace("#", " ").split()
        if not parts:
            continue
        if "sm" in parts:
            col = parts.index("sm")
            continue
        if col is not None and len(parts) > col and parts[col].replace(".", "").isdigit():
            vals.append(float(parts[col]))
    return round(sum(vals) / len(vals), 1) if vals else None


def tokens_per_page(output: Path) -> float:
    total = n = 0
    for f in output.glob("part-*.parquet"):
        t = pq.read_table(f, columns=["completion_tokens"])
        total += int(sum(v for v in t.column(0).to_pylist() if v))
        n += t.num_rows
    return round(total / n, 1) if n else 0.0


def run_bench(
    profile: Profile,
    sample_root: Path,
    work_root: Path,
    tag: str,
    limit: int,
    workers: int | None = None,
    cpuset: str | None = None,
    dmon_dir: Path | None = None,
) -> BenchResult:
    """Pass 1 over the first ``limit`` sample pages into a fresh output dir under ``work_root``."""
    p = profile.model_copy(deep=True)
    if workers is not None:
        p.render.workers = workers
    if cpuset is not None:
        p.render.cpuset = cpuset
    out = work_root / tag
    if out.exists():
        shutil.rmtree(out)
    rows = iter_pending_pages(sample_root / "manifest", 1)
    import itertools

    rows = itertools.islice(rows, limit)
    dmon = None
    log = (dmon_dir or work_root) / f"dmon-{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("nvidia-smi"):
        dmon = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-d", "2"],
            stdout=log.open("w"),
            stderr=subprocess.DEVNULL,
        )
    t0 = time.perf_counter()
    try:
        stats = SaturateRunner(p).run(rows, out / "raw", 1)
    finally:
        if dmon is not None:
            dmon.terminate()
            dmon.wait()
    util = parse_dmon(log.read_text()) if dmon is not None and log.exists() else None
    return BenchResult(
        tag=tag,
        pages=stats.rows_processed,
        workers=p.render.workers,
        cpuset=p.render.cpuset,
        pages_per_s=stats.pages_per_s,
        bound_by=stats.bound_by,
        out_tokens_per_page=tokens_per_page(out / "raw"),
        gpu_util_pct=util,
        elapsed_s=round(time.perf_counter() - t0, 1),
        errors=stats.errors,
    )


def choose_workers(results: list[BenchResult], threshold: float = 0.95) -> int:
    """Smallest worker count whose run was engine-bound >= 95%; else the best pages/s."""
    ok = sorted((r for r in results if r.engine_share >= threshold), key=lambda r: r.workers)
    if ok:
        return ok[0].workers
    return max(results, key=lambda r: r.pages_per_s).workers
