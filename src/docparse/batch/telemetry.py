"""Read saturate ``telemetry-*.jsonl`` back into pages/s and ``bound_by`` shares."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from saturate import bound_by_counts

WARMUP_S = 120.0


def telemetry_files(output: Path) -> set[Path]:
    return set(output.glob("telemetry-*.jsonl"))


def read_ticks(output: Path, exclude: set[Path] | None = None) -> list[dict[str, Any]]:
    """Ticks from every telemetry file in ``output`` except ``exclude`` (files of earlier runs)."""
    ticks: list[dict[str, Any]] = []
    for f in sorted(telemetry_files(output) - (exclude or set())):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ticks.append(json.loads(line))
    return ticks


def bound_by_shares(ticks: list[dict[str, Any]]) -> dict[str, float]:
    counts = bound_by_counts(ticks)
    total = sum(counts.values())
    return {k: round(v / total, 4) for k, v in counts.items()} if total else {}


def pages_per_s(rows_done: int, elapsed_s: float) -> float:
    return round(rows_done / elapsed_s, 3) if elapsed_s > 0 else 0.0


def steady_pages_per_s(ticks: list[dict[str, Any]], warmup_s: float = WARMUP_S) -> float:
    """Pages/s after the first ``warmup_s`` seconds (``ok`` = completions per tick)."""
    if not ticks:
        return 0.0
    t_end = max(float(t["t"]) for t in ticks)
    if t_end <= warmup_s:
        return 0.0
    done = sum(int(t.get("ok") or 0) for t in ticks if float(t["t"]) > warmup_s)
    return round(done / (t_end - warmup_s), 3)
