"""Pending page rows from the manifest, in (doc_id, page_no) order."""

from __future__ import annotations

from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

# text_layer is large and not needed to render or request; QA reads it from the manifest itself.
ROW_COLUMNS = ["page_id", "doc_id", "render_path", "page_no", "width_pt", "height_pt"]


def iter_pending_pages(
    manifest: Path,
    pass_no: int,
    ids: Collection[str] | None = None,
    batch_size: int = 4096,
) -> Iterator[dict[str, Any]]:
    """Yield page rows with status ``pending``. Pass 2 passes an explicit ``ids`` set."""
    if pass_no == 2 and ids is None:
        raise ValueError("pass 2 needs an explicit set of page ids")
    wanted = set(ids) if ids is not None else None
    pf = pq.ParquetFile(manifest / "pages.parquet")
    for batch in pf.iter_batches(batch_size=batch_size, columns=[*ROW_COLUMNS, "status"]):
        mask = pc.equal(batch.column("status"), "pending")
        if wanted is not None:
            mask = pc.and_(mask, pc.is_in(batch.column("page_id"), value_set=_arr(wanted)))
        yield from batch.filter(mask).drop_columns(["status"]).to_pylist()


def _arr(values: set[str]) -> Any:
    import pyarrow as pa

    return pa.array(sorted(values), type=pa.string())
