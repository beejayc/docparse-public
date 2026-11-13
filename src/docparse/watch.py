"""Incremental assembly while the GPU pass runs: assemble docs whose pages have all returned."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from config.settings import Profile
from docparse.assemble import assemble_all
from docparse.qa.run import run_qa

log = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 300.0


def watch(
    data_root: Path,
    profile: Profile,
    interval_s: float = DEFAULT_INTERVAL_S,
    once: bool = False,
    max_cycles: int | None = None,
) -> list[dict[str, int]]:
    """Every ``interval_s`` assemble newly complete documents. Returns per-cycle counts."""
    history: list[dict[str, int]] = []
    while True:
        run_qa(data_root, 1, workers=profile.cpu_other.workers, nice=profile.cpu_other.nice)
        counts = assemble_all(data_root, profile)
        history.append(counts)
        log.info("watch cycle: %s", counts)
        if once or (max_cycles is not None and len(history) >= max_cycles):
            return history
        if counts["ready"] == 0 and counts["waiting"] == 0:
            return history  # nothing left to do
        time.sleep(interval_s)
