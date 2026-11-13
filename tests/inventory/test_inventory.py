import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from docparse.inventory.detect import detect_type
from docparse.inventory.manifest import write_manifest
from docparse.inventory.walk import walk

FILES = Path(__file__).parents[1] / "fixtures" / "files"


def _corpus(tmp: Path) -> Path:
    root = tmp / "corpus"
    (root / "sub").mkdir(parents=True)
    shutil.copy(FILES / "digital.pdf", root / "a.pdf")
    shutil.copy(FILES / "digital.pdf", root / "sub" / "copy.pdf")
    shutil.copy(FILES / "scanned.pdf", root / "scan.pdf")
    shutil.copy(FILES / "blank.png", root / "blank.png")
    (root / "weird.bin").write_bytes(b"\x00\x01\x02 nothing known")
    return root


def test_detect(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    assert detect_type(root / "a.pdf") == "pdf"
    assert detect_type(root / "blank.png") == "image"
    assert detect_type(root / "weird.bin") == "unknown"
    mislabelled = tmp_path / "x.dat"
    shutil.copy(FILES / "digital.pdf", mislabelled)
    assert detect_type(mislabelled) == "pdf"


def test_ooxml_container(tmp_path: Path) -> None:
    import zipfile

    for kind, member in (
        ("docx", "word/document.xml"),
        ("xlsx", "xl/workbook.xml"),
        ("pptx", "ppt/presentation.xml"),
    ):
        p = tmp_path / f"t.{kind}"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("[Content_Types].xml", "<x/>")
            zf.writestr(member, "<x/>")
        assert detect_type(p) == kind
    z = tmp_path / "plain.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("a.txt", "hi")
    assert detect_type(z) == "zip"


def test_duplicates_share_doc_id(tmp_path: Path) -> None:
    recs = walk(_corpus(tmp_path))
    dup = [r for r in recs if len(r.paths) == 2]
    assert len(dup) == 1 and len(recs) == 4


def test_manifest(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    n_docs, n_pages = write_manifest(root, tmp_path / "data", workers=2)
    assert (n_docs, n_pages) == (4, 3 + 2 + 1)
    pages = pq.read_table(tmp_path / "data" / "manifest" / "pages.parquet").to_pylist()
    assert pages == sorted(pages, key=lambda r: (r["doc_id"], r["page_no"]))
    digital = [p for p in pages if p["text_layer_chars"] > 200]
    assert len(digital) == 3 and not any(p["scanned"] for p in digital)
    assert all(p["lang_hint"] in ("en", "de") for p in digital)
    scanned = [p for p in pages if p["scanned"]]
    assert len(scanned) == 3  # 2 scanned PDF pages + the blank image
    docs = {
        d["detected_type"]: d
        for d in pq.read_table(tmp_path / "data" / "manifest" / "docs.parquet").to_pylist()
    }
    assert docs["unknown"]["status"] == "skipped"
    assert docs["pdf"]["status"] == "normalised"


def test_guards_encrypted_and_too_many_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    from docparse.inventory import manifest

    root = tmp_path / "c"
    root.mkdir()
    enc = canvas.Canvas(str(root / "enc.pdf"), pagesize=A4, encrypt="secret")
    enc.drawString(72, 700, "hidden")
    enc.save()
    shutil.copy(FILES / "digital.pdf", root / "big.pdf")
    monkeypatch.setattr(manifest, "MAX_PAGES", 2)  # in-process only; see below
    res = manifest.scan_doc("x" * 16, str(root / "enc.pdf"))
    assert (res.status, res.reason) == ("skipped", "encrypted")
    res = manifest.scan_doc("y" * 16, str(root / "big.pdf"))
    assert res.status == "skipped" and "too many pages" in (res.reason or "")


@pytest.mark.slow
def test_inventory_1000_pdfs(tmp_path: Path) -> None:
    import time

    root = tmp_path / "many"
    root.mkdir()
    src = FILES / "digital.pdf"
    for i in range(1000):  # unique bytes so nothing dedups
        (root / f"{i}.pdf").write_bytes(src.read_bytes() + b"\n%" + str(i).encode())
    t0 = time.perf_counter()
    n_docs, _ = write_manifest(root, tmp_path / "data", workers=8)
    elapsed = time.perf_counter() - t0
    print(f"1000 PDFs inventoried in {elapsed:.1f}s")
    assert n_docs == 1000 and elapsed < 120
