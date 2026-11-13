"""Native (no-OCR) Block records, ARCHITECTURE 6.5. Written as plain dicts; P5 owns the model."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from docparse.ids import block_id, page_id


def make_block(
    doc_id: str,
    page_no: int,
    seq: int,
    type_: str,
    text: str,
    markdown: str,
    native_ref: str,
    html: str | None = None,
) -> dict[str, Any]:
    pid = page_id(doc_id, page_no)
    return {
        "block_id": block_id(pid, seq),
        "doc_id": doc_id,
        "page_id": pid,
        "page_no": page_no,
        "seq": seq,
        "type": type_,
        "bbox_norm": None,
        "bbox_pt": None,
        "text": text,
        "markdown": markdown,
        "html": html,
        "continues_prev": False,
        "merged_from": [],
        "asset_path": None,
        "alt": None,
        "section_path": [],
        "native_ref": native_ref,
        "source": "native",
        "quality": "ok",
        "qa_flags": [],
    }


def write_native(data_root: Path, doc_id: str, blocks: Iterable[dict[str, Any]]) -> Path:
    """Write out/native/{doc_id}.jsonl."""
    path = data_root / "out" / "native" / f"{doc_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for b in blocks:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    return path
