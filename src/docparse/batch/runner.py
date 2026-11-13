"""Batch runner contract (ARCHITECTURE section 8)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class RunStats:
    rows_processed: int
    errors: int
    pages_per_s: float
    bound_by: dict[str, float] = field(default_factory=dict)
    rendered: int = 0  # pages that reached the render stage in this run (resume test)
    elapsed_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class BatchRunner(Protocol):
    def run(self, rows: Iterable[dict[str, Any]], output: Path, pass_no: int) -> RunStats: ...


def parse_cpuset(spec: str | None) -> list[int] | None:
    """'4-8,10' -> [4, 5, 6, 7, 8, 10]. Empty/None -> None (no pinning)."""
    if not spec:
        return None
    cores: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        cores.extend(range(int(lo), int(hi or lo) + 1))
    return cores
