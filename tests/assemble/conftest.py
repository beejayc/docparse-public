"""Shared builders for assembly tests: a synthetic 5-page document and manifest tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from config.settings import Profile, load_profile
from docparse.assemble.document import DocTask
from docparse.assemble.types import Selected
from docparse.ids import page_id
from docparse.inventory.manifest import DOCS_SCHEMA, PAGES_SCHEMA

FIX = Path(__file__).parents[1] / "fixtures"
GOLD = FIX / "assemble" / "gold"
DIGITAL = str(FIX / "files" / "digital.pdf")
DOC_ID = "a0a0a0a0a0a0a0a0"
W, H = 595.2755737304688, 841.8897705078125


@pytest.fixture
def profile() -> Profile:
    p = load_profile("consumer24")
    return p.model_copy(
        update={"cpu_other": p.cpu_other.model_copy(update={"cpuset": "", "workers": 2})}
    )


def gold_pages(doc_id: str = DOC_ID) -> list[dict[str, Any]]:
    return [
        {
            "page_id": page_id(doc_id, n),
            "doc_id": doc_id,
            "render_path": DIGITAL,
            "page_no": n,
            "width_pt": W,
            "height_pt": H,
            "status": "pending",
        }
        for n in range(1, 6)
    ]


def gold_selected(doc_id: str = DOC_ID, upto: int = 5) -> dict[str, Selected]:
    return {
        page_id(doc_id, n): Selected(
            page_id(doc_id, n),
            (GOLD / f"p{n}.raw.txt").read_text(encoding="utf-8"),
            "vlm_fast",
            "ok",
            (),
        )
        for n in range(1, upto + 1)
    }


def gold_doc(doc_id: str = DOC_ID) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "paths": ["/in/report.pdf"],
        "detected_type": "pdf",
        "size_bytes": 1,
        "parent_doc_id": None,
        "render_path": DIGITAL,
        "n_pages": 5,
        "status": "normalised",
        "reason": None,
    }


def gold_task(out_root: Path, profile: Profile) -> DocTask:
    return DocTask(
        doc=gold_doc(),
        pages=gold_pages(),
        selected=gold_selected(),
        out_root=out_root,
        profile=profile,
    )


def write_manifest_tables(
    data_root: Path, docs: list[dict[str, Any]], pages: list[dict[str, Any]]
) -> None:
    m = data_root / "manifest"
    m.mkdir(parents=True, exist_ok=True)
    full_docs = [{k: d.get(k) for k in DOCS_SCHEMA.names} for d in docs]
    full_pages = [
        {k: p.get(k, "" if k == "text_layer" else None) for k in PAGES_SCHEMA.names} for p in pages
    ]
    pq.write_table(pa.Table.from_pylist(full_docs, schema=DOCS_SCHEMA), m / "docs.parquet")
    pq.write_table(pa.Table.from_pylist(full_pages, schema=PAGES_SCHEMA), m / "pages.parquet")
