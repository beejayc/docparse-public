"""Pick the best result per page (DESIGN section 4, step 5), including the text-layer fallback."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from docparse.assemble.types import Selected
from docparse.qa.store import read_qa

FULL_PAGE = "![text](0,0,1000,1000) "
MIN_LAYER_CHARS = 20
AGREEMENT_ONLY = {"low_recall", "low_precision", "low_agreement_warn"}  # ADR-026


def _read_raw(raw_dir: Path, ids: Collection[str] | None) -> dict[str, dict[str, Any]]:
    from saturate import read_output

    if not raw_dir.exists():
        return {}
    wanted = set(ids) if ids is not None else None
    return {pid: row for pid, row in read_output(str(raw_dir)) if wanted is None or pid in wanted}


def _layers(manifest: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not ids:
        return out
    pf = pq.ParquetFile(manifest / "pages.parquet")
    for batch in pf.iter_batches(
        batch_size=2048, columns=["page_id", "text_layer", "text_layer_chars"]
    ):
        for row in batch.to_pylist():
            if row["page_id"] in ids:
                out[row["page_id"]] = row
    return out


def _quality(flags: tuple[str, ...]) -> str:
    return "warn" if "low_agreement_warn" in flags else "ok"


def select_results(out_root: Path, page_ids: Collection[str] | None = None) -> dict[str, Selected]:
    """Pass 2 if it passed, else pass 1 if it passed, else the fallback.

    - pass-1 page flagged but no (QA'd) pass-2 result yet: provisional pass-1 raw, ``quality=low``,
      flag ``escalation_pending`` (ADR-023).
    - both passes failed: text-layer block when the page has a layer (``source=textlayer``),
      otherwise the pass-2 raw; ``quality=low``. Failures that are only agreement flags keep the
      pass-2 raw (ADR-026).
    """
    raw1 = _read_raw(out_root / "raw" / "pass1", page_ids)
    raw2 = _read_raw(out_root / "raw" / "pass2", set(raw1))
    qa1, qa2 = read_qa(out_root, 1), read_qa(out_root, 2)

    out: dict[str, Selected] = {}
    fallback: list[str] = []
    for pid, r1 in raw1.items():
        q1, q2, r2 = qa1.get(pid), qa2.get(pid), raw2.get(pid)
        if r2 is not None and q2 is not None and not q2["escalate"]:
            flags = tuple(q2["flags"])
            out[pid] = Selected(pid, r2["raw"], "vlm_accurate", _quality(flags), flags)
        elif q1 is None or not q1["escalate"]:
            flags = tuple(q1["flags"]) if q1 else ()
            out[pid] = Selected(pid, r1["raw"], "vlm_fast", _quality(flags), flags)
        elif r2 is None or q2 is None:
            flags = (*q1["flags"], "escalation_pending")
            out[pid] = Selected(pid, r1["raw"], "vlm_fast", "low", flags)
        else:
            fallback.append(pid)
    layers = _layers(out_root.parent / "manifest", set(fallback))
    for pid in fallback:
        flags = tuple(dict.fromkeys([*qa1[pid]["flags"], *qa2[pid]["flags"]]))
        layer = layers.get(pid)
        # ADR-026: the text layer only wins when the VLM output is structurally broken. If the
        # only complaint is disagreement with the layer, both models agree with each other, and the
        # layer (glyph soup on maths, OCR noise on scans) is the likelier culprit.
        if set(flags) <= AGREEMENT_ONLY:
            out[pid] = Selected(pid, raw2[pid]["raw"], "vlm_accurate", "low", flags)
        elif layer and int(layer["text_layer_chars"] or 0) >= MIN_LAYER_CHARS:
            raw = FULL_PAGE + str(layer["text_layer"]).strip()
            out[pid] = Selected(pid, raw, "textlayer", "low", flags)
        else:
            out[pid] = Selected(pid, raw2[pid]["raw"], "vlm_accurate", "low", flags)
    return out
