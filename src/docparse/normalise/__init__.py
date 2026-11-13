"""Step 2 of the run lifecycle: Office/XLSX/CSV/email/ZIP -> PDFs, native blocks, child docs."""

from __future__ import annotations

import logging
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import Settings, load_profile
from docparse.ids import hash_file
from docparse.inventory.detect import detect_type
from docparse.inventory.manifest import (
    DOCS_SCHEMA,
    PAGES_SCHEMA,
    ScanResult,
    _init_scan_worker,
    _scan_task,
)
from docparse.normalise.archive import MAX_DEPTH, extract_zip
from docparse.normalise.email import (
    email_blocks,
    keep_attachment,
    parse_eml,
    parse_msg,
)
from docparse.normalise.native import write_native
from docparse.normalise.office import ConversionError, convert_doc
from docparse.normalise.sheets import csv_blocks, xlsx_blocks
from docparse.osutil import lower_priority

log = logging.getLogger(__name__)

OFFICE = ("docx", "pptx")
SCAN = ("pdf", "image")
HANDLED = (*OFFICE, *SCAN, "xlsx", "csv", "eml", "msg", "zip")

# (doc_id, detected_type, src_path, converted_dir, timeout) -> converted pdf path
Converter = Callable[[str, str, str, str, float], str]


def _init_convert_worker(nice: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    lower_priority(nice)


def _convert_task(args: tuple[str, str, str, str, float]) -> tuple[str | None, str | None]:
    try:
        return convert_doc(*args), None
    except ConversionError as exc:
        return None, str(exc)
    except Exception as exc:  # unreadable/corrupt input must not stop the run
        return None, f"convert failed: {exc}"


def _depth(doc: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> int:
    d, cur = 0, doc
    while cur.get("parent_doc_id") and d < 50:
        d += 1
        cur = by_id.get(cur["parent_doc_id"], {})
    return d


class _Manifest:
    def __init__(self, data_root: Path) -> None:
        self.dir = data_root / "manifest"
        self.docs: list[dict[str, Any]] = pq.read_table(self.dir / "docs.parquet").to_pylist()
        self.pages: list[dict[str, Any]] = pq.read_table(self.dir / "pages.parquet").to_pylist()
        self.by_id = {d["doc_id"]: d for d in self.docs}

    def add_child(self, parent: dict[str, Any], path: Path) -> dict[str, Any] | None:
        doc_id = hash_file(path)[:16]
        if doc_id in self.by_id:
            log.info("child %s duplicates existing doc %s", path, doc_id)
            return None
        kind = detect_type(path)
        row: dict[str, Any] = {
            "doc_id": doc_id,
            "paths": [str(path)],
            "detected_type": kind,
            "size_bytes": path.stat().st_size,
            "parent_doc_id": parent["doc_id"],
            "render_path": None,
            "n_pages": None,
            "status": "pending",
            "reason": None,
        }
        if kind == "unknown":
            row.update(status="skipped", reason="unsupported file type")
        elif _depth(row, self.by_id | {parent["doc_id"]: parent}) > MAX_DEPTH:
            row.update(status="skipped", reason=f"nesting deeper than {MAX_DEPTH}")
        self.docs.append(row)
        self.by_id[doc_id] = row
        return row

    def save(self) -> None:
        self.docs.sort(key=lambda d: d["doc_id"])
        self.pages.sort(key=lambda p: (p["doc_id"], p["page_no"]))
        pq.write_table(
            pa.Table.from_pylist(self.docs, schema=DOCS_SCHEMA), self.dir / "docs.parquet"
        )
        pq.write_table(
            pa.Table.from_pylist(self.pages, schema=PAGES_SCHEMA), self.dir / "pages.parquet"
        )


def _fail(doc: dict[str, Any], reason: str) -> None:
    log.warning("doc %s failed: %s", doc["doc_id"], reason)
    doc.update(status="failed", reason=reason)


def _convert_office(
    man: _Manifest,
    todo: list[dict[str, Any]],
    data_root: Path,
    workers: int,
    nice: int,
    timeout: float,
    converter: Converter | None,
) -> list[dict[str, Any]]:
    out_dir = str(data_root / "work" / "converted")
    tasks = [(d["doc_id"], d["detected_type"], d["paths"][0], out_dir, timeout) for d in todo]
    if converter is not None:
        results = [_safe(converter, t) for t in tasks]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, workers),
            mp_context=ctx,
            initializer=_init_convert_worker,
            initargs=(nice,),
        ) as pool:
            results = list(pool.map(_convert_task, tasks))
    ok = []
    for doc, (pdf, err) in zip(todo, results, strict=True):
        if pdf is None:
            _fail(doc, err or "conversion failed")
        else:
            doc["render_path"] = pdf
            ok.append(doc)
    return ok


def _safe(fn: Converter, task: tuple[str, str, str, str, float]) -> tuple[str | None, str | None]:
    try:
        return fn(*task), None
    except Exception as exc:
        return None, str(exc)


def _scan(man: _Manifest, docs: list[dict[str, Any]], workers: int) -> None:
    if not docs:
        return
    jobs = [(d["doc_id"], d["render_path"] or d["paths"][0]) for d in docs]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max(1, workers), mp_context=ctx, initializer=_init_scan_worker
    ) as pool:
        results: list[ScanResult] = list(pool.map(_scan_task, jobs))
    for doc, (_, path), res in zip(docs, jobs, results, strict=True):
        doc.update(
            status=res.status,
            reason=res.reason,
            n_pages=res.n_pages,
            render_path=path if res.status == "normalised" else None,
        )
        man.pages.extend(res.pages)


def _native(man: _Manifest, doc: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    """Handle sheet/email/zip docs inline. Returns newly registered child docs."""
    kind, doc_id, src = doc["detected_type"], doc["doc_id"], Path(doc["paths"][0])
    children: list[dict[str, Any]] = []
    try:
        if kind in ("xlsx", "csv"):
            blocks, n = (xlsx_blocks if kind == "xlsx" else csv_blocks)(doc_id, src)
            write_native(data_root, doc_id, blocks)
            doc.update(status="normalised", n_pages=n)
        elif kind in ("eml", "msg"):
            parsed = parse_eml(src) if kind == "eml" else parse_msg(src)
            write_native(data_root, doc_id, email_blocks(doc_id, parsed))
            dest = data_root / "work" / "extracted" / doc_id
            for i, (name, data, inline) in enumerate(parsed.attachments, start=1):
                if not keep_attachment(data, name, inline):
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                f = dest / f"{i:02d}_{name}"
                f.write_bytes(data)
                if (child := man.add_child(doc, f)) is not None:
                    children.append(child)
            doc.update(status="normalised", n_pages=1)
        elif kind == "zip":
            files, rejected = extract_zip(src, data_root / "work" / "extracted" / doc_id)
            for f in files:
                if (child := man.add_child(doc, f)) is not None:
                    children.append(child)
            reason = f"rejected members: {rejected}" if rejected else None
            doc.update(status="normalised", reason=reason)
    except Exception as exc:
        _fail(doc, f"{kind} failed: {exc}")
    return children


def normalise_all(
    settings: Settings, converter: Converter | None = None, timeout: float = 120
) -> dict[str, int]:
    """Process every pending doc; loop until no new child docs appear. Idempotent."""
    data_root = settings.data_root
    profile = load_profile(settings.profile)
    workers = max(1, min(profile.cpu_other.workers, os.cpu_count() or 1))
    man = _Manifest(data_root)
    stats = {"rounds": 0, "converted": 0, "native": 0, "children": 0, "failed": 0}
    while True:
        pending = [
            d for d in man.docs if d["status"] == "pending" and d["detected_type"] in HANDLED
        ]
        if not pending:
            break
        stats["rounds"] += 1
        office = [d for d in pending if d["detected_type"] in OFFICE]
        to_scan = [d for d in pending if d["detected_type"] in SCAN]
        if office:
            converted = _convert_office(
                man, office, data_root, workers, profile.cpu_other.nice, timeout, converter
            )
            stats["converted"] += len(converted)
            to_scan += converted
        _scan(man, to_scan, workers)
        for doc in [d for d in pending if d["detected_type"] not in (*OFFICE, *SCAN)]:
            kids = _native(man, doc, data_root)
            stats["native"] += 1
            stats["children"] += len(kids)
        man.save()
    stats["failed"] = sum(1 for d in man.docs if d["status"] == "failed")
    return stats
