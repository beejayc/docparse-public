"""Assembly: raw OCR output + native blocks -> ``out/docs/{doc_id}/`` (Phase 5)."""

from __future__ import annotations

import logging
import multiprocessing
import os
from collections.abc import Callable, Collection
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from config.settings import Profile
from docparse.assemble.document import DocTask, assemble_document
from docparse.assemble.types import Selected
from docparse.batch.runner import parse_cpuset
from docparse.osutil import lower_priority
from docparse.render import init_render_worker

log = logging.getLogger(__name__)

# Selector: (out_root, page_ids) -> {page_id: Selected}; the default is qa.select.select_results.
Selector = Callable[[Path, Collection[str] | None], dict[str, Selected]]
PAGE_COLUMNS = ["page_id", "doc_id", "render_path", "page_no", "width_pt", "height_pt", "status"]


def _init_worker(cores: list[int] | None, nice: int, mode: str, side: int) -> None:
    init_render_worker(cores, 2, mode=mode, longest_side_px=side)
    lower_priority(nice)


def _run_task(task: DocTask) -> dict[str, Any]:
    return assemble_document(task)


def is_done(out_root: Path, doc_id: str) -> bool:
    return (out_root / "docs" / doc_id / ".done").exists()


def plan(
    data_root: Path,
    profile: Profile,
    selector: Selector | None = None,
    only: Collection[str] | None = None,
    force: bool = False,
) -> tuple[list[DocTask], dict[str, int]]:
    """Tasks for documents that are complete (all pending pages OCR'd, or native) and not done."""
    if selector is None:
        from docparse.qa.select import select_results

        selector = select_results
    out_root = data_root / "out"
    docs = pq.read_table(data_root / "manifest" / "docs.parquet").to_pylist()
    pages_path = data_root / "manifest" / "pages.parquet"
    pages = (
        pq.read_table(pages_path, columns=PAGE_COLUMNS).to_pylist() if pages_path.exists() else []
    )
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for p in pages:
        if p["status"] == "pending":
            by_doc.setdefault(p["doc_id"], []).append(p)
    children: dict[str, list[dict[str, Any]]] = {}
    for d in docs:
        if d.get("parent_doc_id"):
            children.setdefault(d["parent_doc_id"], []).append(d)

    candidates = [
        d
        for d in docs
        if (only is None or d["doc_id"] in only)
        and d["status"] in ("normalised", "pending")
        and (force or not is_done(out_root, d["doc_id"]))
        and (by_doc.get(d["doc_id"]) or (out_root / "native" / f"{d['doc_id']}.jsonl").exists())
    ]
    need = {p["page_id"] for d in candidates for p in by_doc.get(d["doc_id"], [])}
    selected = selector(out_root, need) if need else {}

    tasks: list[DocTask] = []
    waiting = 0
    for d in candidates:
        dp = by_doc.get(d["doc_id"], [])
        if any(p["page_id"] not in selected for p in dp):
            waiting += 1
            continue
        kids = (
            [(c["doc_id"], Path(c["paths"][0]).name) for c in children.get(d["doc_id"], [])]
            if d["detected_type"] in ("eml", "msg")
            else []
        )
        tasks.append(
            DocTask(
                doc=d,
                pages=dp,
                selected={p["page_id"]: selected[p["page_id"]] for p in dp},
                out_root=out_root,
                profile=profile,
                attachments=sorted(kids),
            )
        )
    tasks.sort(key=lambda t: (-len(t.pages), t.doc["doc_id"]))  # largest first
    return tasks, {"ready": len(tasks), "waiting": waiting}


def assemble_all(
    data_root: Path,
    profile: Profile,
    selector: Selector | None = None,
    only: Collection[str] | None = None,
    force: bool = False,
    workers: int | None = None,
) -> dict[str, int]:
    tasks, counts = plan(data_root, profile, selector, only, force)
    counts = {**counts, "assembled": 0, "failed": 0}
    if not tasks:
        return counts
    n = max(1, min(workers or profile.cpu_other.workers, os.cpu_count() or 1, len(tasks)))
    r = profile.render
    with ProcessPoolExecutor(
        max_workers=n,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(
            parse_cpuset(profile.cpu_other.cpuset),
            profile.cpu_other.nice,
            r.mode,
            r.longest_side_px,
        ),
    ) as pool:
        futures = [(t.doc["doc_id"], pool.submit(_run_task, t)) for t in tasks]
        for doc_id, fut in futures:
            try:
                fut.result()
                counts["assembled"] += 1
            except Exception:
                log.exception("assembly failed for %s", doc_id)
                counts["failed"] += 1
    return counts
