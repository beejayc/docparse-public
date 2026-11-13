import json
from pathlib import Path
from typing import Any

import pytest

from config.settings import Profile
from docparse.assemble import assemble_all, is_done, plan
from docparse.assemble.blocks import build_blocks
from docparse.assemble.document import assemble_document
from docparse.assemble.types import Selected
from docparse.watch import watch
from tests.assemble.conftest import (
    DOC_ID,
    GOLD,
    H,
    W,
    gold_doc,
    gold_pages,
    gold_selected,
    gold_task,
    write_manifest_tables,
)

GROUNDING = Path(__file__).parents[1] / "fixtures" / "grounding"


def _blocks(out: Path) -> list[dict[str, Any]]:
    lines = (out / "docs" / DOC_ID / "blocks.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_golden_markdown_is_byte_identical(tmp_path: Path, profile: Profile) -> None:
    expected = (GOLD / "expected.document.md").read_bytes()
    for run in ("a", "b"):
        out = tmp_path / run
        assemble_document(gold_task(out, profile))
        assert (out / "docs" / DOC_ID / "document.md").read_bytes() == expected


def test_blocks_jsonl_deterministic(tmp_path: Path, profile: Profile) -> None:
    for run in ("a", "b"):
        assemble_document(gold_task(tmp_path / run, profile))
    a = (tmp_path / "a" / "docs" / DOC_ID / "blocks.jsonl").read_bytes()
    assert a == (tmp_path / "b" / "docs" / DOC_ID / "blocks.jsonl").read_bytes()


def test_cross_page_table_merged(tmp_path: Path, profile: Profile) -> None:
    assemble_document(gold_task(tmp_path, profile))
    blocks = _blocks(tmp_path)
    tables = [b for b in blocks if b["type"] == "table" and b["page_no"] <= 3]
    assert len(tables) == 1 and len(tables[0]["merged_from"]) == 1
    assert tables[0]["merged_from"][0].endswith(":00003:002")
    assert not any(b["page_no"] == 3 and b["type"] == "table" for b in blocks)
    grid = json.loads(
        (tmp_path / "docs" / DOC_ID / "tables" / f"{tables[0]['block_id']}.json").read_text()
    )
    assert grid["header_rows"] == 1 and grid["n_rows"] == 5  # 1 header + R1..R4
    assert [r[0]["text"] for r in grid["cells"]] == ["Region", "R1", "R2", "R3", "R4"]


def test_rowspan_html_simple_pipe_and_table_jsons(tmp_path: Path, profile: Profile) -> None:
    assemble_document(gold_task(tmp_path, profile))
    md = (tmp_path / "docs" / DOC_ID / "document.md").read_text()
    assert '<th rowspan="2">Group</th>' in md  # spans -> HTML verbatim
    assert "| Region | Revenue | Costs |" in md  # simple -> pipe
    table_blocks = [b for b in _blocks(tmp_path) if b["type"] in ("table", "chart")]
    assert len(table_blocks) == 3
    for b in table_blocks:
        assert (tmp_path / "docs" / DOC_ID / "tables" / f"{b['block_id']}.json").exists()


def test_bbox_pt_within_page_and_bottom_left_origin(tmp_path: Path, profile: Profile) -> None:
    assemble_document(gold_task(tmp_path, profile))
    blocks = _blocks(tmp_path)
    with_box = [b for b in blocks if b["bbox_pt"]]
    assert with_box
    for b in with_box:
        left, bottom, right, top = b["bbox_pt"]
        assert 0 <= left <= right <= W + 1e-6 and 0 <= bottom <= top <= H + 1e-6
    header = next(b for b in blocks if b["type"] == "header" and b["page_no"] == 1)
    assert header["bbox_pt"][1] > H * 0.9  # a page header sits near the TOP, i.e. high y


def test_sections_titles_and_omitted_types(tmp_path: Path, profile: Profile) -> None:
    assemble_document(gold_task(tmp_path, profile))
    md = (tmp_path / "docs" / DOC_ID / "document.md").read_text()
    assert md.startswith("<!-- page: 1 -->\n\n# Quarterly Report 2026\n")
    assert "## Regional table" in md and "\n# Regional" not in md
    assert "ACME Quarterly Report" not in md and "Confidential" not in md  # header/footer
    blocks = _blocks(tmp_path)
    assert any(b["type"] == "page_number" for b in blocks)  # kept in JSONL
    cap = next(b for b in blocks if b["type"] == "caption")
    assert cap["section_path"] == ["Quarterly Report 2026", "Detailed results"]


def test_crops_exist_with_asset_path(tmp_path: Path, profile: Profile) -> None:
    from PIL import Image

    assemble_document(gold_task(tmp_path, profile))
    d = tmp_path / "docs" / DOC_ID
    imgs = [b for b in _blocks(tmp_path) if b["type"] in ("image", "chart")]
    assert len(imgs) == 2
    for b in imgs:
        assert b["asset_path"] == f"assets/{b['block_id']}.png" and (d / b["asset_path"]).exists()
    with Image.open(d / imgs[0]["asset_path"]) as im:
        assert 700 < im.width < 1700  # 80% of 2048 px wide plus 8 px padding each side


def test_meta_and_done_marker_last(tmp_path: Path, profile: Profile) -> None:
    assemble_document(gold_task(tmp_path, profile))
    d = tmp_path / "docs" / DOC_ID
    meta = json.loads((d / "meta.json").read_text())
    assert meta["models"]["fast"]["id"] == "lightonai/LightOnOCR-3-0.8B"
    assert meta["n_pages"] == 5 and meta["parser_version"]
    assert (d / ".done").exists()


@pytest.mark.parametrize("raw", sorted(GROUNDING.glob("*.raw.txt")), ids=lambda p: p.name[:10])
def test_real_fixtures_build_clean_blocks(raw: Path) -> None:
    pid = f"{raw.name[:10]}:00001"
    row = {"page_id": pid, "doc_id": raw.name[:10], "page_no": 1, "width_pt": W, "height_pt": H}
    sel = Selected(pid, raw.read_text(encoding="utf-8"), "vlm_fast", "ok", ())
    blocks = build_blocks(sel, row)
    assert blocks and all(b.quality == "ok" and not b.qa_flags for b in blocks)
    for b in blocks:
        assert b.bbox_pt is not None and 0 <= b.bbox_pt[0] <= b.bbox_pt[2] <= W + 1e-6


def _data_root(tmp_path: Path, n_pages_in_manifest: int = 5) -> Path:
    root = tmp_path / "data"
    write_manifest_tables(root, [gold_doc()], gold_pages()[:n_pages_in_manifest])
    return root


def test_watch_waits_for_all_pages_then_assembles(tmp_path: Path, profile: Profile) -> None:
    root = _data_root(tmp_path)

    def partial(_: Path, __: Any) -> dict[str, Selected]:
        return gold_selected(upto=4)

    tasks, counts = plan(root, profile, partial)
    assert tasks == [] and counts == {"ready": 0, "waiting": 1}
    assert not is_done(root / "out", DOC_ID)

    def complete(_: Path, __: Any) -> dict[str, Selected]:
        return gold_selected()

    tasks, counts = plan(root, profile, complete)
    assert counts == {"ready": 1, "waiting": 0}
    res = assemble_all(root, profile, complete, workers=1)
    assert res["assembled"] == 1 and res["failed"] == 0 and is_done(root / "out", DOC_ID)
    # already done -> nothing to do; --force redoes it
    assert plan(root, profile, complete)[1] == {"ready": 0, "waiting": 0}
    assert plan(root, profile, complete, force=True)[1]["ready"] == 1


def test_watch_loop_picks_up_late_pages(
    tmp_path: Path, profile: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _data_root(tmp_path)
    state = {"upto": 3}

    def selector(_: Path, __: Any) -> dict[str, Selected]:
        sel = gold_selected(upto=state["upto"])
        state["upto"] = 5  # the GPU "finishes" between cycles
        return sel

    import docparse.assemble as asm
    import docparse.watch as w

    monkeypatch.setattr(
        w, "assemble_all", lambda dr, pr: asm.assemble_all(dr, pr, selector, workers=1)
    )
    monkeypatch.setattr(w.time, "sleep", lambda _: None)
    history = watch(root, profile, interval_s=0)
    assert history[0] == {"ready": 0, "waiting": 1, "assembled": 0, "failed": 0}
    assert history[1]["assembled"] == 1 and is_done(root / "out", DOC_ID)


def test_native_doc_and_email_attachment_links(tmp_path: Path, profile: Profile) -> None:
    root = tmp_path / "data"
    mail = {**gold_doc("b1b1b1b1b1b1b1b1"), "detected_type": "eml", "n_pages": None}
    child = {
        **gold_doc("c2c2c2c2c2c2c2c2"),
        "parent_doc_id": "b1b1b1b1b1b1b1b1",
        "paths": ["/in/inv.pdf"],
    }
    write_manifest_tables(root, [mail, child], [])
    native = root / "out" / "native"
    native.mkdir(parents=True)
    blk = {
        "block_id": "b1b1b1b1b1b1b1b1:00001:001", "doc_id": "b1b1b1b1b1b1b1b1",
        "page_id": "b1b1b1b1b1b1b1b1:00001", "page_no": 1, "seq": 1, "type": "email_body",
        "text": "Subject: hi", "markdown": "| Subject | hi |\n\nBody text", "source": "native",
    }  # fmt: skip
    (native / "b1b1b1b1b1b1b1b1.jsonl").write_text(json.dumps(blk) + "\n")
    res = assemble_all(root, profile, lambda *_: {}, workers=1)
    assert res["assembled"] == 1  # the child has no pages and no native file: not assemblable
    md = (root / "out" / "docs" / "b1b1b1b1b1b1b1b1" / "document.md").read_text()
    assert "Body text" in md and "- [inv.pdf](../c2c2c2c2c2c2c2c2/document.md)" in md


def test_cli_assemble_and_watch_once(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from docparse.cli import app

    root = tmp_path / "data"
    write_manifest_tables(root, [gold_doc()], gold_pages())
    runner = CliRunner()
    r = runner.invoke(app, ["--data-root", str(root), "assemble"])
    assert r.exit_code == 0 and '"assembled": 0' in r.output and '"waiting": 1' in r.output
    r = runner.invoke(app, ["--data-root", str(root), "watch", "--once"])
    assert r.exit_code == 0 and '"waiting": 1' in r.output
