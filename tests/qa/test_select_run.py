from pathlib import Path
from typing import Any

import pytest

from docparse.qa import select as sel
from docparse.qa.run import run_qa
from docparse.qa.store import escalated_ids, read_qa, write_qa
from tests.assemble.conftest import write_manifest_tables

DOC = "d0d0d0d0d0d0d0d0"
LAYER = " ".join(f"wort{i}x" for i in range(100))
GOOD = "![text](0,0,1000,1000) " + LAYER
BAD = "![text](0,0,1000,1000) tiny"  # empty vs a long text layer


def pid(n: int) -> str:
    return f"{DOC}:{n:05d}"


def qa_row(n: int, pass_no: int, flags: list[str]) -> dict[str, Any]:
    escalate = any(f != "low_agreement_warn" for f in flags)
    return {
        "page_id": pid(n),
        "pass_no": pass_no,
        "flags": flags,
        "recall": None,
        "precision": None,
        "escalate": escalate,
    }


def raw_row(text: str, finish: str = "stop") -> dict[str, Any]:
    return {"raw": text, "finish_reason": finish, "model": "m", "error": None}


def test_qa_store_roundtrip_and_escalated_ids(tmp_path: Path) -> None:
    write_qa(
        tmp_path,
        1,
        [qa_row(2, 1, ["truncated"]), qa_row(1, 1, []), qa_row(3, 1, ["low_agreement_warn"])],
    )
    got = read_qa(tmp_path, 1)
    assert list(got) == sorted(got) and got[pid(2)]["flags"] == ["truncated"]
    assert escalated_ids(tmp_path, 1) == {pid(2)}  # warn does not escalate


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """out_root with 6 pages covering every selection branch; raw output is injected."""
    out_root = tmp_path / "data" / "out"
    pages = [
        {"page_id": pid(n), "doc_id": DOC, "render_path": "x.pdf", "page_no": n, "width_pt": 100.0,
         "height_pt": 100.0, "status": "pending", "text_layer": LAYER if n != 6 else "",
         "text_layer_chars": len(LAYER) if n != 6 else 0, "scanned": n == 6}
        for n in range(1, 7)
    ]  # fmt: skip
    write_manifest_tables(tmp_path / "data", [], pages)
    raw1 = {pid(n): raw_row(GOOD if n in (1, 2) else BAD) for n in range(1, 7)}
    raw2 = {pid(3): raw_row(GOOD), pid(4): raw_row(BAD), pid(6): raw_row("![text](0,0,9,9) other")}
    # 1 ok pass1 | 2 warn pass1 | 3 flagged -> pass2 ok | 4 flagged, pass2 fails -> text layer
    # 5 flagged, no pass2 yet -> provisional | 6 scanned, both fail -> pass-2 raw, low
    write_qa(
        out_root,
        1,
        [qa_row(1, 1, []), qa_row(2, 1, ["low_agreement_warn"])]
        + [qa_row(n, 1, ["empty"]) for n in (3, 4, 5, 6)],
    )
    write_qa(out_root, 2, [qa_row(3, 2, []), qa_row(4, 2, ["empty"]), qa_row(6, 2, ["empty"])])

    def fake(raw_dir: Path, ids: Any) -> dict[str, dict[str, Any]]:
        src = raw2 if raw_dir.name == "pass2" else raw1
        return {k: v for k, v in src.items() if ids is None or k in ids}

    monkeypatch.setattr(
        sel, "_read_raw", lambda d, ids: fake(d, ids) if d.name in ("pass1", "pass2") else {}
    )
    (out_root / "raw" / "pass1").mkdir(parents=True)
    (out_root / "raw" / "pass2").mkdir(parents=True)
    return out_root


def test_select_every_branch(world: Path) -> None:
    got = sel.select_results(world)
    assert (got[pid(1)].source, got[pid(1)].quality) == ("vlm_fast", "ok")
    assert (got[pid(2)].source, got[pid(2)].quality) == ("vlm_fast", "warn")
    assert got[pid(3)].source == "vlm_accurate" and got[pid(3)].quality == "ok"
    assert got[pid(3)].raw == GOOD  # pass 2 wins where it passes
    fb = got[pid(4)]
    assert (fb.source, fb.quality) == ("textlayer", "low")
    assert fb.raw == "![text](0,0,1000,1000) " + LAYER and fb.qa_flags == ("empty",)
    pend = got[pid(5)]
    assert (pend.source, pend.quality) == (
        "vlm_fast",
        "low",
    ) and "escalation_pending" in pend.qa_flags
    scan = got[pid(6)]
    assert (scan.source, scan.quality) == ("vlm_accurate", "low") and scan.raw.endswith("other")


def test_select_filters_page_ids(world: Path) -> None:
    assert set(sel.select_results(world, {pid(1), pid(3)})) == {pid(1), pid(3)}


def test_assembly_uses_select_results_by_default(world: Path) -> None:
    """The text-layer fallback becomes a single full-page text block in the assembled doc."""
    from docparse.assemble.blocks import build_blocks

    s = sel.select_results(world, {pid(4)})[pid(4)]
    blocks = build_blocks(
        s, {"page_id": pid(4), "doc_id": DOC, "page_no": 4, "width_pt": 100.0, "height_pt": 100.0}
    )
    assert len(blocks) == 1 and blocks[0].type == "text" and blocks[0].source == "textlayer"
    assert blocks[0].quality == "low" and blocks[0].bbox_pt == (0.0, 0.0, 100.0, 100.0)


def test_run_qa_end_to_end_and_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import saturate

    root = tmp_path / "data"
    pages = [
        {"page_id": pid(n), "doc_id": DOC, "render_path": "x.pdf", "page_no": n, "width_pt": 100.0,
         "height_pt": 100.0, "status": "pending", "text_layer": LAYER,
         "text_layer_chars": len(LAYER), "scanned": False}
        for n in (1, 2)
    ]  # fmt: skip
    write_manifest_tables(root, [], pages)
    out = root / "out"
    (out / "raw" / "pass1").mkdir(parents=True)
    (out / "docs" / DOC).mkdir(parents=True)
    (out / "docs" / DOC / ".done").write_text("")
    rows = {pid(1): raw_row(GOOD), pid(2): raw_row(GOOD, "length")}
    monkeypatch.setattr(saturate, "read_output", lambda _: iter(rows.items()))

    res = run_qa(root, 1, workers=1)
    assert res["pages_new"] == 2 and res["escalated"] == 1 and res["flags"] == {"truncated": 1}
    assert res["docs_invalidated"] == 1 and not (out / "docs" / DOC / ".done").exists()
    assert read_qa(out, 1)[pid(2)]["escalate"] is True

    again = run_qa(root, 1, workers=1)  # idempotent: nothing new, nothing recomputed
    assert again["pages_new"] == 0 and again["escalated"] == 1


def test_agreement_only_failure_keeps_vlm_not_text_layer(world: Path) -> None:
    """ADR-026: page 4 fails both passes, but only on agreement -> keep pass 2, not the layer."""
    write_qa(world, 1, [qa_row(n, 1, ["low_recall"]) for n in (3, 4)])
    write_qa(world, 2, [qa_row(n, 2, ["low_recall", "low_precision"]) for n in (3, 4)])
    got = sel.select_results(world)
    assert got[pid(4)].source == "vlm_accurate" and got[pid(4)].quality == "low"
    assert got[pid(4)].raw == BAD
