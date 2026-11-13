"""Primary runner: stream -> skip_done -> prepare_ahead (render pool) -> pump (saturate)."""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from saturate import prepare_ahead as _prepare_ahead
from saturate import pump, skip_done, stream

from config.settings import Profile
from docparse.batch.runner import RunStats, parse_cpuset
from docparse.batch.telemetry import (
    bound_by_shares,
    read_ticks,
    steady_pages_per_s,
    telemetry_files,
)
from docparse.render import init_render_worker, render_row
from docparse.vlm.request import parse, to_request

log = logging.getLogger(__name__)


def api_base(endpoint: str) -> str:
    """saturate wants the OpenAI base URL including /v1; profiles store the bare host."""
    base = endpoint.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def request_builder(profile: Profile, pass_no: int, model: str) -> Any:
    max_tokens = (
        profile.sampling.max_tokens.pass1 if pass_no == 1 else profile.sampling.max_tokens.pass2
    )
    return partial(
        to_request,
        model=model,
        temperature=profile.sampling.temperature,
        top_p=profile.sampling.top_p,
        max_tokens=max_tokens,
    )


class _Counter:
    def __init__(self) -> None:
        self.n = 0

    def wrap(
        self, it: Iterable[tuple[str, dict[str, Any]]]
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        for item in it:
            self.n += 1
            yield item


class SaturateRunner:
    def __init__(self, profile: Profile, model: str | None = None) -> None:
        self.profile = profile
        self.model = model

    def run(self, rows: Iterable[dict[str, Any]], output: Path, pass_no: int) -> RunStats:
        p = self.profile
        model = self.model or (p.vllm.models.fast.id if pass_no == 1 else p.vllm.models.accurate.id)
        output.mkdir(parents=True, exist_ok=True)
        counter = _Counter()
        before = telemetry_files(output)
        # skip_done BEFORE rendering: a resumed run must not render finished pages.
        pending = skip_done(stream(rows, id_key="page_id"), str(output), retry_errors=True)
        r = p.render
        with ProcessPoolExecutor(
            max_workers=r.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init,
            initargs=(
                parse_cpuset(r.cpuset),
                r.pdf_cache_size,
                r.mode,
                r.longest_side_px,
                r.png_compress_level,
            ),
        ) as pool:
            prepared = _prepare_ahead(
                counter.wrap(pending), render_row, workers=r.workers, executor=pool
            )
            stats = pump(
                prepared,
                to_request=request_builder(p, pass_no, model),
                parse=parse,
                endpoint=api_base(p.endpoint),
                output=str(output),
                retry_errors=True,
            )
        ticks = read_ticks(output, exclude=before)
        steady = steady_pages_per_s(ticks)
        overall = round(stats.rows_processed / stats.elapsed_s, 3) if stats.elapsed_s else 0.0
        return RunStats(
            rows_processed=stats.rows_processed,
            errors=stats.rows_failed,
            pages_per_s=steady or overall,  # steady state excludes the first 2 min
            bound_by=bound_by_shares(ticks),
            rendered=counter.n,
            elapsed_s=round(stats.elapsed_s, 2),
        )


def _init(
    cores: list[int] | None, cache_size: int, mode: str, longest_side_px: int, compress: int
) -> None:
    init_render_worker(
        cores,
        cache_size,
        mode=mode,
        longest_side_px=longest_side_px,
        png_compress_level=compress,
    )
