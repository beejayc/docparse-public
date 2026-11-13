from pathlib import Path

import polars as pl

from docparse.bench.full import sweep_values, write_measured
from docparse.bench.run import BenchResult, choose_workers, cost_per_1000, parse_dmon, workers_for
from docparse.bench.sample import build_sample, classify


def _root(tmp_path: Path, name: str, kinds: list[tuple[str, bool]]) -> Path:
    root = tmp_path / name
    (root / "manifest").mkdir(parents=True)
    docs = [{"doc_id": f"{name}{i}", "detected_type": k} for i, (k, _) in enumerate(kinds)]
    pages = [
        {"page_id": f"{name}{i}:00001", "doc_id": f"{name}{i}", "scanned": s, "status": "pending"}
        for i, (_, s) in enumerate(kinds)
    ]
    pl.DataFrame(docs).write_parquet(root / "manifest" / "docs.parquet")
    pl.DataFrame(pages).write_parquet(root / "manifest" / "pages.parquet")
    return root


def test_classify() -> None:
    assert classify("png", False) == "image"
    assert classify("pdf", True) == "scanned"
    assert classify("pdf", False) == "digital"


def test_sample_is_stratified_deterministic_and_cached(tmp_path: Path) -> None:
    a = _root(tmp_path, "a", [("pdf", False)] * 80 + [("pdf", True)] * 10)
    b = _root(tmp_path, "b", [("png", False)] * 30)
    out = tmp_path / "s"
    m1 = build_sample([a, b], out, n=40, seed=1)
    assert m1["pages"] == 40
    assert m1["composition"] == {"digital": 28, "scanned": 6, "image": 6}
    first = pl.read_parquet(out / "manifest" / "pages.parquet")["page_id"].to_list()
    m2 = build_sample([a, b], out, n=40, seed=1)  # cached
    assert m2 == m1
    out2 = tmp_path / "s2"
    build_sample([a, b], out2, n=40, seed=1)
    assert pl.read_parquet(out2 / "manifest" / "pages.parquet")["page_id"].to_list() == first
    docs = pl.read_parquet(out / "manifest" / "docs.parquet")["doc_id"].to_list()
    assert set(docs) == {p.split(":")[0] for p in first}


def test_sample_tops_up_when_a_class_is_short(tmp_path: Path) -> None:
    a = _root(tmp_path, "a", [("pdf", False)] * 30 + [("png", False)] * 2)
    m = build_sample([a], tmp_path / "s", n=20, seed=1)
    assert m["pages"] == 20


def test_worker_formula_and_sweep() -> None:
    assert workers_for(0.9, 0.2) == 1
    assert workers_for(7.7, 0.45) == 5  # ceil(7.7 x 0.45 x 1.3 = 4.5)
    assert sweep_values(1) == [1, 2, 3]
    assert sweep_values(5) == [3, 4, 5, 6, 7]


def _res(w: int, pps: float, engine: float) -> BenchResult:
    return BenchResult("t", 100, w, "0", pps, {"engine": engine}, 500.0, None, 10.0, 0)


def test_choose_workers_smallest_engine_bound() -> None:
    rs = [_res(1, 0.7, 0.80), _res(2, 0.9, 0.96), _res(3, 0.9, 0.99)]
    assert choose_workers(rs) == 2
    assert choose_workers([_res(1, 0.7, 0.5), _res(2, 0.8, 0.6)]) == 2  # none qualifies -> best pps


def test_cost_and_dmon() -> None:
    assert cost_per_1000(1.0, 3.6) == 1.0
    assert cost_per_1000(0.0, 3.6) is None
    text = "# gpu   sm  mem  enc  dec\n# Idx    %    %    %    %\n    0   90   40    0    0\n    0   70   30    0    0\n"
    assert parse_dmon(text) == 80.0
    assert parse_dmon("") is None


def test_write_measured_keeps_comments(tmp_path: Path) -> None:
    y = tmp_path / "p.yaml"
    y.write_text(
        "# c\nrender:\n  mode: direct\n  workers: 6\n  cpuset: x\ncpu_other:\n  workers: 10\nmeasured: {}  # fill\n"
    )
    write_measured(y, {"pages_per_s": 0.9}, 2)
    t = y.read_text()
    assert "# c" in t and "  workers: 2\n" in t and "  workers: 10" in t
    assert "measured: {pages_per_s: 0.9}" in t
