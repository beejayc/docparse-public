"""Live runner tests (need vLLM up on the profile endpoint): make gpu-test."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import saturate

from config.settings import load_profile
from docparse.batch.httpx_runner import HttpxRunner
from docparse.batch.rows import iter_pending_pages
from docparse.batch.saturate_runner import SaturateRunner
from docparse.inventory.manifest import write_manifest

BENCH = Path("dataset/olmOCR-bench/bench_data/pdfs")

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not BENCH.exists(), reason="bench dataset not downloaded")
def test_runners_same_schema_and_resume(tmp_path: Path) -> None:
    profile = load_profile("gb10")
    write_manifest(BENCH / "headers_footers", tmp_path / "data", workers=4)
    rows = list(iter_pending_pages(tmp_path / "data" / "manifest", 1))[:50]
    assert len(rows) == 50

    a_out, b_out = tmp_path / "sat", tmp_path / "httpx"
    s = SaturateRunner(profile).run(rows, a_out, 1)
    h = HttpxRunner(profile).run(rows, b_out, 1)
    assert s.errors == 0 and h.errors == 0 and s.rendered == h.rendered == 50

    a = pq.read_table(next(a_out.glob("part-*.parquet"))).schema
    b = pq.read_table(next(b_out.glob("part-*.parquet"))).schema
    assert {f.name: f.type for f in a} == {f.name: f.type for f in b}

    ids = [i for i, _ in saturate.read_output(str(a_out))]
    assert len(ids) == len(set(ids)) == 50
    again = SaturateRunner(profile).run(rows, a_out, 1)
    assert again.rendered == 0 and again.rows_processed == 0
