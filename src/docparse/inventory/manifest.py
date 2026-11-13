"""Write manifest/docs.parquet and manifest/pages.parquet (ARCHITECTURE 6.1-6.2)."""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from docparse import render
from docparse.ids import page_id
from docparse.inventory.walk import DocRecord, walk

log = logging.getLogger(__name__)

MAX_PAGES = 2000
SCANNED_CHARS = 20
SCAN_TYPES = ("pdf", "image")

DOCS_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("paths", pa.list_(pa.string())),
        ("detected_type", pa.string()),
        ("size_bytes", pa.int64()),
        ("parent_doc_id", pa.string()),
        ("render_path", pa.string()),
        ("n_pages", pa.int32()),
        ("status", pa.string()),
        ("reason", pa.string()),
    ]
)

PAGES_SCHEMA = pa.schema(
    [
        ("page_id", pa.string()),
        ("doc_id", pa.string()),
        ("render_path", pa.string()),
        ("page_no", pa.int32()),
        ("width_pt", pa.float32()),
        ("height_pt", pa.float32()),
        ("text_layer", pa.string()),
        ("text_layer_chars", pa.int32()),
        ("scanned", pa.bool_()),
        ("lang_hint", pa.string()),
        ("status", pa.string()),
    ]
)

_detector: Any = None


def _init_scan_worker() -> None:
    """Pool initialiser: pinned-thread libs, tiny PDF cache, one English/German detector."""
    global _detector
    render.init_render_worker(None, 1)
    from lingua import Language, LanguageDetectorBuilder

    _detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.GERMAN).build()


def lang_hint(text: str, detector: Any | None = None) -> str | None:
    """'en' or 'de' for text with enough letters, else None."""
    det = detector if detector is not None else _detector
    if det is None or sum(c.isalpha() for c in text) < SCANNED_CHARS:
        return None
    lang = det.detect_language_of(text)
    if lang is None:
        return None
    return str(lang.iso_code_639_1.name).lower()


@dataclass
class ScanResult:
    n_pages: int | None
    status: str
    reason: str | None
    pages: list[dict[str, Any]]


def _page_count(path: str) -> int:
    if path.lower().endswith(render.PDF_SUFFIX):
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(path)
        try:
            return len(doc)
        finally:
            doc.close()
    with Image.open(path) as im:
        return int(getattr(im, "n_frames", 1))


def scan_doc(doc_id: str, path: str) -> ScanResult:
    """Count pages and extract text layers for one PDF/image. Runs in a worker process."""
    try:
        n = _page_count(path)
    except Exception as exc:
        text = str(exc).lower()
        if "password" in text or "encrypt" in text:
            return ScanResult(None, "skipped", "encrypted", [])
        return ScanResult(None, "failed", f"open failed: {exc}", [])
    if n > MAX_PAGES:
        return ScanResult(n, "skipped", f"too many pages ({n} > {MAX_PAGES})", [])
    pages: list[dict[str, Any]] = []
    try:
        for no in range(1, n + 1):
            try:
                text, w, h = render.text_layer(path, no)
                row_status = "pending"
            except Exception as exc:
                log.warning("text layer failed %s p%d: %s", path, no, exc)
                text, w, h, row_status = "", 0.0, 0.0, "failed"
            chars = len(text.strip())
            pages.append(
                {
                    "page_id": page_id(doc_id, no),
                    "doc_id": doc_id,
                    "render_path": path,
                    "page_no": no,
                    "width_pt": w,
                    "height_pt": h,
                    "text_layer": text,
                    "text_layer_chars": chars,
                    "scanned": chars < SCANNED_CHARS,
                    "lang_hint": lang_hint(text),
                    "status": row_status,
                }
            )
    finally:
        render.close_cache()
    return ScanResult(n, "normalised", None, pages)


def _scan_task(args: tuple[str, str]) -> ScanResult:
    return scan_doc(*args)


def build_tables(records: Iterable[DocRecord], workers: int) -> tuple[pa.Table, pa.Table]:
    recs = sorted(records, key=lambda r: r.doc_id)
    to_scan = [(r.doc_id, r.paths[0]) for r in recs if r.detected_type in SCAN_TYPES]
    results: dict[str, ScanResult] = {}
    if to_scan:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, workers), mp_context=ctx, initializer=_init_scan_worker
        ) as pool:
            for (doc_id, _), res in zip(to_scan, pool.map(_scan_task, to_scan), strict=True):
                results[doc_id] = res

    doc_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    for r in recs:
        row: dict[str, Any] = {
            "doc_id": r.doc_id,
            "paths": list(r.paths),
            "detected_type": r.detected_type,
            "size_bytes": r.size_bytes,
            "parent_doc_id": None,
            "render_path": None,
            "n_pages": None,
            "status": "pending",  # office/email/zip wait for P2 normalise
            "reason": None,
        }
        if r.detected_type == "unknown":
            row.update(status="skipped", reason="unsupported file type")
        elif r.detected_type in SCAN_TYPES:
            res = results[r.doc_id]
            row.update(
                status=res.status,
                reason=res.reason,
                n_pages=res.n_pages,
                render_path=r.paths[0] if res.status == "normalised" else None,
            )
            page_rows.extend(res.pages)
        doc_rows.append(row)

    page_rows.sort(key=lambda p: (p["doc_id"], p["page_no"]))
    docs = pa.Table.from_pylist(doc_rows, schema=DOCS_SCHEMA)
    pages = pa.Table.from_pylist(page_rows, schema=PAGES_SCHEMA)
    return docs, pages


def write_manifest(root: Path, data_root: Path, workers: int) -> tuple[int, int]:
    """Inventory ``root`` into ``<data_root>/manifest/``. Returns (n_docs, n_pages)."""
    records = walk(root)
    docs, pages = build_tables(records, workers)
    out = data_root / "manifest"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(docs, out / "docs.parquet")
    pq.write_table(pages, out / "pages.parquet")
    return docs.num_rows, pages.num_rows
