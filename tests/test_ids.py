from datetime import datetime
from pathlib import Path

from docparse.ids import block_id, doc_id_for, page_id, run_id


def test_doc_id_is_sha256_prefix(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert doc_id_for(f) == "2cf24dba5fb0a30e"


def test_ids_format() -> None:
    assert page_id("3f2a9c1e5b7d4e8a", 12) == "3f2a9c1e5b7d4e8a:00012"
    assert block_id("3f2a9c1e5b7d4e8a:00012", 4) == "3f2a9c1e5b7d4e8a:00012:004"
    assert run_id("consumer24", datetime(2026, 11, 12, 10, 22)) == "20261112-102200-consumer24"
