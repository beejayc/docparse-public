"""Assemble one document folder: blocks.jsonl, document.md, tables/, assets/, meta.json."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import Profile
from docparse import __version__
from docparse.assemble.blocks import build_blocks
from docparse.assemble.crops import crop_assets
from docparse.assemble.markdown import finalize_blocks, render_markdown
from docparse.assemble.merge import grids_for, merge_tables
from docparse.assemble.types import Block, Selected

NATIVE_PAGE_ORDER = ("page_no", "seq")


@dataclass
class DocTask:
    """Everything one worker needs to assemble one document (picklable)."""

    doc: dict[str, Any]  # docs.parquet row
    pages: list[dict[str, Any]]  # pending pages.parquet rows (no text_layer)
    selected: dict[str, Selected]
    out_root: Path  # <data_root>/out
    profile: Profile
    attachments: list[tuple[str, str]] = field(default_factory=list)  # (child_doc_id, name)


def _read_native(path: Path) -> list[Block]:
    if not path.exists():
        return []
    return [
        Block.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8")


def assemble_document(task: DocTask) -> dict[str, Any]:
    doc_id = task.doc["doc_id"]
    out_dir = task.out_root / "docs" / doc_id
    native_path = task.out_root / "native" / f"{doc_id}.jsonl"
    pages = sorted(task.pages, key=lambda p: p["page_no"])
    page_rows = {int(p["page_no"]): p for p in pages}

    per_page = [build_blocks(task.selected[p["page_id"]], p) for p in pages]
    grids = grids_for([b for blocks in per_page for b in blocks])
    blocks, grids = merge_tables(per_page, grids)

    # fresh output folder (the path is built here from doc_id, never from outside input)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    r = task.profile.render
    crops = crop_assets(blocks, page_rows, out_dir, r.mode, r.longest_side_px)
    blocks = [
        b.model_copy(update={"asset_path": f"assets/{b.block_id}.png"})
        if b.block_id in crops
        else b
        for b in blocks
    ]
    blocks = finalize_blocks(blocks, grids)

    native = _read_native(native_path)
    all_blocks = sorted([*blocks, *native], key=lambda b: (b.page_no, b.seq))

    (out_dir / "tables").mkdir()
    for b in all_blocks:
        if b.block_id in grids:
            _write_json(
                out_dir / "tables" / f"{b.block_id}.json", grids[b.block_id].model_dump(mode="json")
            )
    if not any((out_dir / "tables").iterdir()):
        (out_dir / "tables").rmdir()

    with (out_dir / "blocks.jsonl").open("w", encoding="utf-8") as fh:
        for b in all_blocks:
            fh.write(
                json.dumps(b.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
            )
    page_nos = sorted({*page_rows, *(b.page_no for b in native)})
    (out_dir / "document.md").write_text(
        render_markdown(all_blocks, page_nos, task.attachments), encoding="utf-8"
    )

    models = task.profile.vllm.models
    used = {b.source for b in all_blocks}
    meta = {
        "doc_id": doc_id,
        "paths": task.doc["paths"],
        "detected_type": task.doc["detected_type"],
        "parent_doc_id": task.doc.get("parent_doc_id"),
        "n_pages": task.doc.get("n_pages"),
        "n_blocks": len(all_blocks),
        "profile": task.profile.name,
        "models": {
            k: {"id": getattr(models, k).id, "revision": getattr(models, k).revision}
            for k, src in (("fast", "vlm_fast"), ("accurate", "vlm_accurate"))
            if src in used
        },
        "image": task.profile.vllm.image,
        "qa": {
            "quality": dict(sorted(Counter(b.quality for b in all_blocks).items())),
            "flags": dict(sorted(Counter(f for b in all_blocks for f in b.qa_flags).items())),
        },
        "parser_version": __version__,
    }
    _write_json(out_dir / "meta.json", meta)
    (out_dir / ".done").write_text("", encoding="utf-8")  # written last: marks a complete folder
    return {"doc_id": doc_id, "n_blocks": len(all_blocks), "n_pages": len(pages)}
