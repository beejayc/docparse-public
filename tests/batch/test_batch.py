import json
import shutil
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest

from config.settings import Profile, load_profile
from docparse.batch.httpx_runner import SCHEMA, HttpxRunner, done_ids
from docparse.batch.rows import iter_pending_pages
from docparse.batch.runner import parse_cpuset
from docparse.batch.saturate_runner import api_base
from docparse.batch.telemetry import bound_by_shares, steady_pages_per_s
from docparse.inventory.manifest import write_manifest

FILES = Path(__file__).parents[1] / "fixtures" / "files"


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    for n in ("digital.pdf", "scanned.pdf", "blank.png"):
        shutil.copy(FILES / n, root / n)
    write_manifest(root, tmp_path / "data", workers=2)
    return tmp_path / "data"


@pytest.fixture
def profile() -> Profile:
    p = load_profile("consumer24")
    return p.model_copy(update={"render": p.render.model_copy(update={"workers": 2, "cpuset": ""})})


def test_parse_cpuset() -> None:
    assert parse_cpuset("4-8") == [4, 5, 6, 7, 8]
    assert parse_cpuset("0-1,5") == [0, 1, 5]
    assert parse_cpuset("") is None and parse_cpuset(None) is None


def test_api_base() -> None:
    assert api_base("http://localhost:8000") == "http://localhost:8000/v1"
    assert api_base("http://h:8000/v1/") == "http://h:8000/v1"


def test_pending_rows_sorted_no_text_layer(manifest: Path) -> None:
    rows = list(iter_pending_pages(manifest / "manifest", 1))
    assert len(rows) == 6
    assert [(r["doc_id"], r["page_no"]) for r in rows] == sorted(
        (r["doc_id"], r["page_no"]) for r in rows
    )
    assert "text_layer" not in rows[0]


def test_pass2_needs_ids(manifest: Path) -> None:
    with pytest.raises(ValueError):
        list(iter_pending_pages(manifest / "manifest", 2))
    first = next(iter_pending_pages(manifest / "manifest", 1))["page_id"]
    got = list(iter_pending_pages(manifest / "manifest", 2, ids={first}))
    assert [r["page_id"] for r in got] == [first]


def test_telemetry_helpers() -> None:
    ticks = [
        {"t": 2.0, "ok": 0, "bound_by": "engine"},
        {"t": 100.0, "ok": 5, "bound_by": "engine"},
        {"t": 130.0, "ok": 10, "bound_by": "source"},
        {"t": 220.0, "ok": 20, "bound_by": "engine"},
    ]
    assert steady_pages_per_s(ticks) == pytest.approx(30 / 100, abs=0.001)
    assert steady_pages_per_s(ticks[:2]) == 0.0
    shares = bound_by_shares(ticks)
    assert shares["engine"] == 0.75 and shares["source"] == 0.25


def _handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1/chat/completions"
    body = json.loads(request.content)
    assert body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png")
    return httpx.Response(
        200,
        json={
            "model": body["model"],
            "choices": [{"message": {"content": "![text](1,2,3,4) hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        },
    )


def test_httpx_runner_schema_and_resume(manifest: Path, profile: Profile, tmp_path: Path) -> None:
    out = tmp_path / "raw"
    runner = HttpxRunner(profile, concurrency=4, transport=httpx.MockTransport(_handler))
    rows = list(iter_pending_pages(manifest / "manifest", 1))
    s1 = runner.run(rows[:4], out, 1)
    assert (s1.rows_processed, s1.errors, s1.rendered) == (4, 0, 4)
    s2 = runner.run(rows, out, 1)  # resume: only the 2 missing pages are rendered
    assert (s2.rows_processed, s2.rendered) == (2, 2)
    assert done_ids(out) == {r["page_id"] for r in rows}
    s3 = runner.run(rows, out, 1)
    assert (s3.rows_processed, s3.rendered) == (0, 0)
    for part in out.glob("part-*.parquet"):
        assert pq.read_schema(part).equals(SCHEMA)
    t = pq.read_table(next(out.glob("part-*.parquet"))).to_pylist()[0]
    assert t["raw"].startswith("![text]") and t["finish_reason"] == "stop"


def test_httpx_runner_records_errors_and_retries(
    manifest: Path, profile: Profile, tmp_path: Path
) -> None:
    out = tmp_path / "raw"
    rows = list(iter_pending_pages(manifest / "manifest", 1))[:2]

    def boom(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    bad = HttpxRunner(profile, transport=httpx.MockTransport(boom)).run(rows, out, 1)
    assert (bad.rows_processed, bad.errors) == (0, 2) and done_ids(out) == set()
    good = HttpxRunner(profile, transport=httpx.MockTransport(_handler)).run(rows, out, 1)
    assert good.rows_processed == 2 and len(done_ids(out)) == 2
