import base64
import time
from pathlib import Path

import httpx
import pytest

from docparse.vlm.grounding import parse_grounding
from docparse.vlm.request import parse, to_request

pytestmark = pytest.mark.gpu


def test_one_page_under_10s() -> None:
    png = (Path(__file__).parents[1] / "fixtures/files").glob("*.png")
    img = next(iter(png))
    row = {"png_b64": base64.b64encode(img.read_bytes()).decode()}
    t0 = time.perf_counter()
    r = httpx.post("http://localhost:8000/v1/chat/completions", json=to_request(row), timeout=60)
    assert time.perf_counter() - t0 < 10
    rec = parse(row, r.json())
    _, warns = parse_grounding(rec["raw"])
    assert "parse_error" not in warns
