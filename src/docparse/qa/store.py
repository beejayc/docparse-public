"""Read/write ``out/qa/pass{N}.parquet`` (ARCHITECTURE 6.4)."""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

QA_SCHEMA = pa.schema(
    [
        ("page_id", pa.string()),
        ("pass_no", pa.int8()),
        ("flags", pa.list_(pa.string())),
        ("recall", pa.float32()),
        ("precision", pa.float32()),
        ("escalate", pa.bool_()),
    ]
)


def qa_path(out_root: Path, pass_no: int) -> Path:
    return out_root / "qa" / f"pass{pass_no}.parquet"


def read_qa(out_root: Path, pass_no: int) -> dict[str, dict[str, Any]]:
    path = qa_path(out_root, pass_no)
    if not path.exists():
        return {}
    return {r["page_id"]: r for r in pq.read_table(path).to_pylist()}


def write_qa(out_root: Path, pass_no: int, rows: Collection[dict[str, Any]]) -> Path:
    """Atomic write (tmp + rename), sorted by page_id so the file is deterministic."""
    path = qa_path(out_root, pass_no)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    table = pa.Table.from_pylist(sorted(rows, key=lambda r: r["page_id"]), schema=QA_SCHEMA)
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return path


def escalated_ids(out_root: Path, pass_no: int = 1) -> set[str]:
    return {pid for pid, r in read_qa(out_root, pass_no).items() if r["escalate"]}
