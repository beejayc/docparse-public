"""Smoke test: render varied pages, call the live server, save fixtures (png, raw, json)."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq

from config.settings import Profile
from docparse.render import render_page_png
from docparse.vlm.request import parse, to_request

log = logging.getLogger(__name__)
FIXTURE_DIR = Path("tests/fixtures/grounding")


def pick_pages(pages: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Varied selection: scanned and German pages first, rest evenly spaced over the manifest."""
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take(rows: list[dict[str, Any]], k: int) -> None:
        step = max(1, len(rows) // max(k, 1))
        for r in rows[::step][:k]:
            if r["page_id"] not in seen and len(chosen) < n:
                seen.add(r["page_id"])
                chosen.append(r)

    take([r for r in pages if r["scanned"]], max(1, n // 5))
    take([r for r in pages if r["lang_hint"] == "de"], max(1, n // 5))
    take(pages, n)
    return chosen


def run_smoke(
    profile: Profile,
    data_root: Path,
    n_pages: int,
    model: str,
    out_dir: Path = FIXTURE_DIR,
    timeout: float = 300.0,
) -> list[dict[str, Any]]:
    pages = pq.read_table(data_root / "manifest" / "pages.parquet").to_pylist()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with httpx.Client(base_url=profile.endpoint, timeout=timeout) as client:
        for row in pick_pages(pages, n_pages):
            name = f"{row['doc_id'][:10]}_p{row['page_no']}"
            if (out_dir / f"{name}.raw.txt").exists():
                log.warning("fixture %s exists; fixtures are final, skipping", name)
                continue
            png = render_page_png(
                row["render_path"],
                row["page_no"],
                profile.render.mode,
                profile.render.longest_side_px,
            )
            row["png_b64"] = base64.b64encode(png).decode()
            body = to_request(
                row,
                model=model,
                temperature=profile.sampling.temperature,
                top_p=profile.sampling.top_p,
                max_tokens=profile.sampling.max_tokens.pass1,
            )
            t0 = time.perf_counter()
            resp = client.post("/v1/chat/completions", json=body)
            latency = time.perf_counter() - t0
            resp.raise_for_status()
            rec = parse(row, resp.json())
            (out_dir / f"{name}.png").write_bytes(png)
            (out_dir / f"{name}.raw.txt").write_text(rec["raw"], encoding="utf-8")
            meta = {k: v for k, v in rec.items() if k != "raw"}
            meta |= {"latency_s": round(latency, 3), "page_id": row["page_id"], "name": name}
            (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            results.append(meta)
    return results
