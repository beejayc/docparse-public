"""Run QA over raw output: parent reads via saturate, chunks fan out to a spawn pool."""

from __future__ import annotations

import io
import logging
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image, ImageStat

from config.settings import Settings
from docparse.osutil import lower_priority
from docparse.qa.checks import AGREEMENT_EXCLUDED, QAConfig, check_page, plain_text
from docparse.qa.store import read_qa, write_qa
from docparse.render import init_render_worker, render_page_png
from docparse.vlm.grounding import parse_grounding

log = logging.getLogger(__name__)

CHUNK = 500
STD_SIDE_PX = 64
PAGE_COLUMNS = ["page_id", "render_path", "page_no", "text_layer", "text_layer_chars", "scanned"]


def _init_worker(nice: int) -> None:
    init_render_worker(None, 2)
    lower_priority(nice)


def image_std(path: str, page_no: int) -> float:
    """Std dev of a tiny grayscale render: > 8 means the page has visible content."""
    png = render_page_png(path, page_no, "direct", STD_SIDE_PX, compress_level=0)
    with Image.open(io.BytesIO(png)) as im:
        return float(ImageStat.Stat(im.convert("L")).stddev[0])


def _qa_chunk(
    items: list[tuple[dict[str, Any], dict[str, Any]]], cfg: QAConfig, pass_no: int
) -> list[dict[str, Any]]:
    out = []
    for raw_row, page_row in items:
        blocks, _ = parse_grounding(raw_row.get("raw") or "")
        short = len(plain_text(blocks, AGREEMENT_EXCLUDED)) < cfg.empty_chars_max
        if short and int(page_row.get("text_layer_chars") or 0) <= cfg.empty_text_layer_chars:
            try:
                page_row = {
                    **page_row,
                    "image_std": image_std(page_row["render_path"], int(page_row["page_no"])),
                }
            except Exception:
                log.warning("image_std failed for %s", page_row.get("page_id"))
        res = check_page(raw_row, page_row, cfg, pass_no)
        out.append(
            {
                "page_id": res.page_id,
                "pass_no": pass_no,
                "flags": res.flags,
                "recall": res.recall,
                "precision": res.precision,
                "escalate": res.escalate,
            }
        )
    return out


def _page_info(manifest: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    pf = pq.ParquetFile(manifest / "pages.parquet")
    for batch in pf.iter_batches(batch_size=2048, columns=PAGE_COLUMNS):
        for row in batch.to_pylist():
            if row["page_id"] in ids:
                info[row["page_id"]] = row
    return info


def run_qa(
    data_root: Path,
    pass_no: int,
    settings: Settings | None = None,
    force: bool = False,
    workers: int = 4,
    nice: int = 10,
) -> dict[str, Any]:
    """QA new raw pages of ``pass_no`` (all of them with ``force``); invalidate changed docs."""
    from saturate import read_output

    out_root = data_root / "out"
    raw_dir = out_root / "raw" / f"pass{pass_no}"
    cfg = QAConfig.from_settings(settings or Settings())
    existing = read_qa(out_root, pass_no)
    raw = {pid: row for pid, row in read_output(str(raw_dir))} if raw_dir.exists() else {}
    todo = sorted(pid for pid in raw if force or pid not in existing)
    counts: dict[str, Any] = {"pass": pass_no, "pages_new": len(todo), "pages_total": len(existing)}
    if not todo:
        return {**counts, "escalated": sum(r["escalate"] for r in existing.values()), "flags": {}}

    info = _page_info(data_root / "manifest", set(todo))
    items = [(raw[pid] | {"id": pid}, info[pid]) for pid in todo if pid in info]
    chunks = [items[i : i + CHUNK] for i in range(0, len(items), CHUNK)]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(chunks))),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(nice,),
    ) as pool:
        for part in pool.map(_qa_chunk, chunks, [cfg] * len(chunks), [pass_no] * len(chunks)):
            results.extend(part)

    merged = {**existing, **{r["page_id"]: r for r in results}}
    write_qa(out_root, pass_no, list(merged.values()))
    changed = {
        r["page_id"].split(":")[0]
        for r in results
        if r["page_id"] not in existing or existing[r["page_id"]]["flags"] != r["flags"]
    }
    invalidated = 0
    for doc_id in changed:
        done = out_root / "docs" / doc_id / ".done"
        if done.exists():
            done.unlink()
            invalidated += 1
    flags = Counter(f for r in results for f in r["flags"])
    return {
        **counts,
        "pages_total": len(merged),
        "escalated": sum(r["escalate"] for r in merged.values()),
        "escalated_new": sum(r["escalate"] for r in results),
        "flags": dict(sorted(flags.items())),
        "docs_invalidated": invalidated,
    }
