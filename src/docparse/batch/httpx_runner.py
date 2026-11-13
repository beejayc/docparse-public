"""Fallback runner: fixed-semaphore httpx client; same output schema as saturate's parquet parts."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import time
import uuid
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import Profile
from docparse.batch.runner import RunStats, parse_cpuset
from docparse.batch.saturate_runner import _init, request_builder
from docparse.render import render_row
from docparse.vlm.request import parse

log = logging.getLogger(__name__)

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("error", pa.string()),
        ("raw", pa.string()),
        ("finish_reason", pa.string()),
        ("prompt_tokens", pa.int64()),
        ("completion_tokens", pa.int64()),
        ("model", pa.string()),
    ]
)
FLUSH_EVERY = 10


def done_ids(output: Path) -> set[str]:
    """Ids with a successful (error IS NULL) record in any part file: exact resume."""
    ids: set[str] = set()
    for f in sorted(output.glob("part-*.parquet")):
        t = pq.read_table(f, columns=["id", "error"])
        ids.update(
            i for i, e in zip(t["id"].to_pylist(), t["error"].to_pylist(), strict=True) if e is None
        )
    return ids


class HttpxRunner:
    def __init__(
        self,
        profile: Profile,
        concurrency: int = 64,
        model: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.transport = transport
        self.profile = profile
        self.concurrency = concurrency
        self.model = model

    def run(self, rows: Iterable[dict[str, Any]], output: Path, pass_no: int) -> RunStats:
        return asyncio.run(self._run(rows, output, pass_no))

    async def _run(self, rows: Iterable[dict[str, Any]], output: Path, pass_no: int) -> RunStats:
        p, r = self.profile, self.profile.render
        model = self.model or (p.vllm.models.fast.id if pass_no == 1 else p.vllm.models.accurate.id)
        output.mkdir(parents=True, exist_ok=True)
        build = request_builder(p, pass_no, model)
        done = done_ids(output)  # skip BEFORE rendering
        todo = (row for row in rows if row["page_id"] not in done)
        sem = asyncio.Semaphore(self.concurrency)
        buf: list[dict[str, Any]] = []
        rendered = failed = ok = 0
        part_tag = uuid.uuid4().hex[:8]
        part_no = 0

        def flush() -> None:
            nonlocal buf, part_no
            if buf:
                pq.write_table(
                    pa.Table.from_pylist(buf, schema=SCHEMA),
                    output / f"part-httpx-{part_tag}-{part_no:05d}.parquet",
                )
                part_no += 1
                buf = []

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        pool = ProcessPoolExecutor(
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
        )

        async def one(client: httpx.AsyncClient, row: dict[str, Any]) -> None:
            nonlocal ok, failed
            rec: dict[str, Any] = {
                "id": row["page_id"], "error": None, "raw": "", "finish_reason": "",
                "prompt_tokens": 0, "completion_tokens": 0, "model": "",
            }  # fmt: skip
            try:
                prepared = await loop.run_in_executor(pool, render_row, row)
                body = build(prepared)
                del prepared
                resp = await client.post("/chat/completions", json=body)
                resp.raise_for_status()
                rec.update(parse(row, resp.json()))
                ok += 1
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            finally:
                sem.release()
            buf.append(rec)
            if len(buf) >= FLUSH_EVERY:
                flush()

        base = self.profile.endpoint.rstrip("/") + "/v1"
        tasks: set[asyncio.Task[None]] = set()
        try:
            async with httpx.AsyncClient(
                base_url=base, timeout=1800, transport=self.transport
            ) as client:
                for row in todo:
                    await sem.acquire()  # bounds rendered-but-unsent payloads too
                    rendered += 1
                    t = asyncio.create_task(one(client, row))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)
                await asyncio.gather(*tasks)
        finally:
            flush()
            pool.shutdown(wait=True)
        elapsed = time.perf_counter() - t0
        return RunStats(
            rows_processed=ok,
            errors=failed,
            pages_per_s=round(ok / elapsed, 3) if elapsed else 0.0,
            bound_by={},
            rendered=rendered,
            elapsed_s=round(elapsed, 2),
        )
