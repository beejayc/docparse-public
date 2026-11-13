import base64
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from docparse import render

FILES = Path(__file__).parent / "fixtures" / "files"
DIGITAL = str(FILES / "digital.pdf")


@pytest.mark.parametrize("mode", ["direct", "supersample"])
def test_longest_side_exact(mode: str) -> None:
    render.init_render_worker(None, 2)
    for page in (1, 2, 3):
        png = render.render_page_png(DIGITAL, page, mode, 2048)
        assert max(Image.open(io.BytesIO(png)).size) == 2048


def test_image_input_resized() -> None:
    render.init_render_worker(None, 2)
    png = render.render_page_png(str(FILES / "blank.png"), 1, "direct", 2048)
    assert max(Image.open(io.BytesIO(png)).size) == 2048


def test_text_layer() -> None:
    render.init_render_worker(None, 2)
    text, w, h = render.text_layer(DIGITAL, 1)
    assert "Quarterly report" in text and w > 500 and h > 800
    assert render.text_layer(str(FILES / "scanned.pdf"), 1)[0].strip() == ""


def test_lru_closes_evicted() -> None:
    render.init_render_worker(None, 1)
    render.text_layer(DIGITAL, 1)
    render.text_layer(str(FILES / "scanned.pdf"), 1)
    render.text_layer(DIGITAL, 1)
    assert render.cache_misses() == 3
    render.close_cache()


def test_four_processes_open_each_pdf_once() -> None:
    rows = [
        {"render_path": p, "page_no": n}
        for p in (DIGITAL, str(FILES / "scanned.pdf"))
        for _ in range(7)
        for n in (1, 2)
    ]  # 28 rows over 2 PDFs, in document order
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        4, mp_context=ctx, initializer=render.init_render_worker, initargs=(None, 4)
    ) as pool:
        out = list(pool.map(render.render_row_with_stats, rows, chunksize=4))
    assert all(base64.b64decode(r["png_b64"])[:4] == b"\x89PNG" for r, _, _ in out)
    misses: dict[int, int] = {}
    for _, pid, m in out:
        misses[pid] = max(misses.get(pid, 0), m)
    assert all(m <= 2 for m in misses.values())  # <= once per PDF per worker
