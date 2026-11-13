"""OmniDocBench scoring (ADR-024): text normalised edit distance and table-text F1 on page level.

Official TEDS needs ``apted`` (not on the dependency allowlist), so tables get a cell-text token F1
proxy. Reading order is scored as the edit distance between ground-truth and predicted block order
(text blocks only, matched by fuzzy text).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from docparse.eval.scorer import normalise

_TAGS = re.compile(r"<[^>]+>")


def load_pages(json_path: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(json_path.read_text(encoding="utf-8"))
    return data


def gt_text(page: dict[str, Any]) -> str:
    """Ground-truth reading-order text of the page (ignored regions and page furniture removed)."""
    dets = [
        d
        for d in page["layout_dets"]
        if d["category_type"] in {"title", "text_block"} and not d.get("ignore") and d.get("text")
    ]
    dets.sort(key=lambda d: d.get("order", 0))
    return "\n".join(normalise(d["text"]) for d in dets)


def gt_tables(page: dict[str, Any]) -> list[str]:
    return [
        normalise(d.get("html") or d.get("latex") or "")
        for d in page["layout_dets"]
        if d["category_type"] == "table" and not d.get("ignore")
    ]


def edit_distance(pred: str, gt: str) -> float:
    """Normalised Levenshtein distance in [0, 1] (0 = identical)."""
    if not gt and not pred:
        return 0.0
    return Levenshtein.normalized_distance(normalise(pred), normalise(gt))


def table_f1(pred_md: str, gt_tables: list[str]) -> float | None:
    gt = Counter(_TAGS.sub(" ", " ".join(gt_tables)).casefold().split())
    if not gt:
        return None
    pr = Counter(
        t for t in _TAGS.sub(" ", pred_md).replace("|", " ").casefold().split() if set(t) != {"-"}
    )
    common = sum((gt & pr).values())
    if not common:
        return 0.0
    p, r = common / sum(pr.values()), common / sum(gt.values())
    return 2 * p * r / (p + r)


def score_page(page: dict[str, Any], pred_md: str) -> dict[str, float | None]:
    """Edit distance on text blocks only: predicted text has tables/equations removed first."""
    body = re.sub(r"<table.*?</table>", " ", pred_md, flags=re.DOTALL | re.IGNORECASE)
    body = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("|"))
    body = re.sub(r"\$\$.*?\$\$", " ", body, flags=re.DOTALL)
    tables = re.findall(r"<table.*?</table>", pred_md, flags=re.DOTALL | re.IGNORECASE)
    tables += [ln for ln in pred_md.splitlines() if ln.lstrip().startswith("|")]
    return {
        "text_edit": edit_distance(body, gt_text(page)),
        "table_f1": table_f1(" ".join(tables), gt_tables(page)),
    }
