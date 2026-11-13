import multiprocessing
import shutil
import stat
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import docparse.normalise as norm
from config.settings import Settings
from docparse.inventory.manifest import DOCS_SCHEMA, write_manifest
from docparse.normalise.archive import extract_zip, unsafe_member
from docparse.normalise.email import parse_eml, parse_msg
from docparse.normalise.office import ConversionError, convert_to_pdf
from docparse.normalise.sheets import csv_blocks, xlsx_blocks

FILES = Path(__file__).parents[1] / "fixtures" / "files"
P2 = FILES / "p2"
HAVE_SOFFICE = shutil.which("soffice") is not None
needs_lo = pytest.mark.skipif(not HAVE_SOFFICE, reason="soffice not installed")


def _fake_converter(doc_id: str, kind: str, src: str, out: str, timeout: float) -> str:
    """Stands in for LibreOffice: 'converts' by copying a known PDF."""
    dest = Path(out) / f"{doc_id}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FILES / "digital.pdf", dest)
    return str(dest)


def _setup(tmp_path: Path, names: list[str]) -> Settings:
    root = tmp_path / "in"
    root.mkdir()
    for n in names:
        shutil.copy(P2 / n, root / n)
    data = tmp_path / "data"
    write_manifest(root, data, 1)
    return Settings(data_root=data)


def _docs(s: Settings) -> list[dict[str, Any]]:
    return pq.read_table(s.data_root / "manifest" / "docs.parquet").to_pylist()


def _pages(s: Settings) -> list[dict[str, Any]]:
    return pq.read_table(s.data_root / "manifest" / "pages.parquet").to_pylist()


def test_xlsx_ranges() -> None:
    blocks, n = xlsx_blocks("d" * 16, P2 / "sales.xlsx")
    assert n == 3
    refs = [(b["page_no"], b["native_ref"]) for b in blocks]
    assert refs == [(1, "Sales!A1:C6"), (1, "Sales!A9:D9"), (3, "Notes!B2:C3")]
    b = blocks[0]
    assert b["type"] == "sheet_table" and b["source"] == "native" and b["bbox_norm"] is None
    assert b["markdown"].startswith("| Region | Q1 | Q2 |\n| --- | --- | --- |")
    assert "<th>Region</th>" in b["html"]
    assert "\\|pipe\\|" in blocks[1]["markdown"]


def test_csv_semicolon() -> None:
    blocks, n = csv_blocks("e" * 16, P2 / "table.csv")
    assert n == 1 and blocks[0]["native_ref"] == "table!A1:B3"
    assert "| bolt | 10 |" in blocks[0]["markdown"]


def test_eml_parse() -> None:
    p = parse_eml(P2 / "mail.eml")
    assert p.headers["Subject"] == "Report attached" and "attached report" in p.body
    assert [a[0] for a in p.attachments] == ["report.pdf", "logo.gif"]
    (n0, d0, i0), (n1, d1, i1) = p.attachments
    assert norm.keep_attachment(d0, n0, i0) and not norm.keep_attachment(d1, n1, i1)


def test_zip_traversal_rejected(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert (
        unsafe_member("../evil.txt") and unsafe_member("/etc/passwd") and unsafe_member("a/../../b")
    )
    assert not unsafe_member("sub/ok.csv")
    with caplog.at_level("WARNING"):
        files, rejected = extract_zip(P2 / "evil.zip", tmp_path / "x")
    assert rejected == ["../evil.txt"] and [f.name for f in files] == ["ok.csv"]
    assert "rejected unsafe member" in caplog.text
    assert (
        not (tmp_path / "evil.txt").exists() and not (tmp_path / "x" / ".." / "evil.txt").exists()
    )


def test_zip_size_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        extract_zip(P2 / "nested.zip", tmp_path / "x", max_total=3)


def test_pipeline_sheets_email_zip(tmp_path: Path) -> None:
    s = _setup(tmp_path, ["sales.xlsx", "table.csv", "mail.eml", "nested.zip"])
    stats = norm.normalise_all(s)
    assert stats["failed"] == 0
    docs = _docs(s)
    by_type = {d["detected_type"]: d for d in docs if d["parent_doc_id"] is None}
    assert all(d["status"] == "normalised" for d in docs), docs
    for t in ("xlsx", "csv", "eml"):
        assert (s.data_root / "out" / "native" / f"{by_type[t]['doc_id']}.jsonl").is_file()
    eml = by_type["eml"]
    kids = [d for d in docs if d["parent_doc_id"] == eml["doc_id"]]
    assert [k["detected_type"] for k in kids] == ["pdf"]  # logo.gif dropped
    assert {p["doc_id"] for p in _pages(s)} == {kids[0]["doc_id"]} and kids[0]["n_pages"] == 3
    # nested zip: top.csv + sub/in.zip -> inner.csv (depth 2)
    zdoc = by_type["zip"]
    zkids = [d for d in docs if d["parent_doc_id"] == zdoc["doc_id"]]
    assert sorted(k["detected_type"] for k in zkids) == ["csv", "zip"]
    inner = [d for d in docs if d["parent_doc_id"] in {k["doc_id"] for k in zkids}]
    assert len(inner) == 1 and inner[0]["detected_type"] == "csv"
    again = norm.normalise_all(s)  # idempotent
    assert again["rounds"] == 0 and len(_docs(s)) == len(docs)


def test_zip_depth_limit(tmp_path: Path) -> None:
    import zipfile

    data = b"a,b\n1,2\n"
    cur = data
    name = "leaf.csv"
    for i in range(5):
        p = tmp_path / f"z{i}.zip"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr(name, cur)
        cur, name = p.read_bytes(), f"z{i}.zip"
    root = tmp_path / "in"
    root.mkdir()
    (root / "top.zip").write_bytes(cur)
    write_manifest(root, tmp_path / "data", 1)
    s = Settings(data_root=tmp_path / "data")
    norm.normalise_all(s)
    docs = _docs(s)
    assert any(d["status"] == "skipped" and "nesting" in (d["reason"] or "") for d in docs)
    assert not any(d["detected_type"] == "csv" for d in docs)


def test_office_with_mock_converter(tmp_path: Path) -> None:
    s = _setup(tmp_path, ["memo.docx", "slides.pptx"])
    stats = norm.normalise_all(s, converter=_fake_converter)
    docs = _docs(s)
    assert stats["converted"] == 2 and all(d["status"] == "normalised" for d in docs)
    assert all(d["render_path"].endswith(f"{d['doc_id']}.pdf") for d in docs)
    assert len(_pages(s)) == 6


def test_conversion_failure_continues(tmp_path: Path) -> None:
    s = _setup(tmp_path, ["memo.docx", "table.csv"])

    def boom(*a: Any) -> str:
        raise ConversionError("timeout after 1s")

    norm.normalise_all(s, converter=boom)
    st = {d["detected_type"]: (d["status"], d["reason"]) for d in _docs(s)}
    assert st["docx"] == ("failed", "timeout after 1s") and st["csv"][0] == "normalised"


def test_msg_with_pdf_attachment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No permissive MSG writer exists, so a stub oxmsg Message feeds the real code path."""
    pdf = (FILES / "digital.pdf").read_bytes()
    stub = SimpleNamespace(
        sender="Alice <a@x.org>",
        recipients=("bob@x.org",),
        sent_date=datetime(2026, 11, 12),
        subject="Hello",
        body="Body text",
        html_body=None,
        attachments=[SimpleNamespace(file_name="../r.pdf", file_bytes=pdf)],
    )
    p = parse_msg(stub)
    assert p.attachments[0][0] == "r.pdf" and p.headers["Subject"] == "Hello"
    root = tmp_path / "in"
    root.mkdir()
    (root / "m.msg").write_bytes(bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 600)
    write_manifest(root, tmp_path / "data", 1)
    s = Settings(data_root=tmp_path / "data")
    monkeypatch.setattr(norm, "parse_msg", lambda _path: parse_msg(stub))
    rows = _docs(s)
    for r in rows:  # puremagic cannot know a stub OLE file; force the type
        r.update(detected_type="msg", status="pending")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=DOCS_SCHEMA), s.data_root / "manifest" / "docs.parquet"
    )
    docs0 = _docs(s)
    norm.normalise_all(s)
    docs = _docs(s)
    child = next(d for d in docs if d["parent_doc_id"])
    assert child["detected_type"] == "pdf" and child["status"] == "normalised"
    assert child["parent_doc_id"] == docs0[0]["doc_id"]
    assert any(p["doc_id"] == child["doc_id"] for p in _pages(s))


def _fake_soffice(tmp_path: Path, body: str) -> str:
    sh = tmp_path / "fake_soffice"
    sh.write_text("#!/bin/sh\n" + body)
    sh.chmod(sh.stat().st_mode | stat.S_IXUSR)
    return str(sh)


def test_timeout_kills_and_raises(tmp_path: Path) -> None:
    src = tmp_path / "a.docx"
    src.write_bytes(b"x")
    with pytest.raises(ConversionError, match="timeout"):
        convert_to_pdf(
            src, tmp_path / "o", 1, timeout=1, soffice_bin=_fake_soffice(tmp_path, "sleep 30\n")
        )


def test_nonzero_exit_raises(tmp_path: Path) -> None:
    src = tmp_path / "a.docx"
    src.write_bytes(b"x")
    with pytest.raises(ConversionError, match="exit 3"):
        convert_to_pdf(src, tmp_path / "o", 1, soffice_bin=_fake_soffice(tmp_path, "exit 3\n"))


@needs_lo
def test_real_libreoffice_docx_pptx(tmp_path: Path) -> None:
    s = _setup(tmp_path, ["memo.docx", "slides.pptx"])
    norm.normalise_all(s, timeout=180)
    docs = _docs(s)
    assert all(d["status"] == "normalised" for d in docs), docs
    pages = _pages(s)
    assert {p["doc_id"] for p in pages} == {d["doc_id"] for d in docs}
    assert any("Quarterly memo" in p["text_layer"] for p in pages)


@needs_lo
@pytest.mark.slow
def test_parallel_speedup(tmp_path: Path) -> None:
    import time
    from concurrent.futures import ProcessPoolExecutor

    src = P2 / "memo.docx"
    jobs = []
    for i in range(40):
        jobs.append((f"{i:016d}", "docx", str(src), str(tmp_path / "out"), 120.0))

    def run(workers: int, chunk: list[Any]) -> float:
        t = time.perf_counter()
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            list(pool.map(norm._convert_task, chunk))
        return time.perf_counter() - t

    t1 = run(1, jobs[:8]) * 5  # extrapolate 40 jobs from 8
    t4 = run(4, jobs)
    assert len(list((tmp_path / "out").glob("*.pdf"))) == 40
    assert t1 / t4 >= 3, (t1, t4)
