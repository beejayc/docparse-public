from typing import Any

from docparse.qa.checks import QAConfig, check_page, table_is_bad

# DESIGN section 4 strict limits: flag logic is tested independent of the calibrated defaults.
CFG = QAConfig(repeat_max=6, recall_min=0.85, precision_min=0.85, agreement_warn_max=0.93)
WORDS = [f"wort{i}x" for i in range(100)]
LAYER = " ".join(WORDS)


def _raw(text: str, finish: str = "stop") -> dict[str, Any]:
    return {"id": "p:00001", "raw": text, "finish_reason": finish}


def _page(layer: str = LAYER, scanned: bool = False) -> dict[str, Any]:
    return {
        "page_id": "p:00001",
        "text_layer": layer,
        "text_layer_chars": len(layer),
        "scanned": scanned,
    }


def _good() -> str:
    return "![text](0,0,1000,1000) " + LAYER


def test_clean_page_has_no_flags() -> None:
    res = check_page(_raw(_good()), _page(), CFG)
    assert res.flags == [] and not res.escalate and res.recall == 1.0 and res.precision == 1.0


def test_truncated() -> None:
    res = check_page(_raw(_good(), finish="length"), _page(), CFG)
    assert "truncated" in res.flags and res.escalate


def test_repetition_by_zlib_ratio() -> None:
    loop = "![text](0,0,1000,1000) " + "the cat sat on the mat. " * 400
    assert "repetition" in check_page(_raw(loop), _page(scanned=True), CFG).flags


def test_repetition_by_repeated_substring() -> None:
    chunk = "alpha beta gamma delta epsilon zeta eta theta "  # > 40 chars, many distinct chars
    noise = " ".join(f"w{i}q" for i in range(300))
    text = "![text](0,0,1000,1000) " + (chunk + noise) * 1 + (" " + chunk) * 8
    assert "repetition" in check_page(_raw(text), _page(scanned=True), CFG).flags


def test_dot_leaders_are_not_a_loop() -> None:
    toc = (
        "![text](0,0,1000,1000) Chapter one "
        + "." * 400
        + " 12 "
        + " ".join(f"entry{i}z" for i in range(200))
    )
    assert "repetition" not in check_page(_raw(toc), _page(scanned=True), CFG).flags


def test_empty_with_long_text_layer() -> None:
    res = check_page(_raw("![text](0,0,10,10) hi"), _page(), CFG)
    assert "empty" in res.flags and res.escalate


def test_empty_by_image_std_on_scan() -> None:
    page = {**_page("", scanned=True), "image_std": 40.0}
    assert "empty" in check_page(_raw(""), page, CFG).flags
    blank = {**_page("", scanned=True), "image_std": 1.0}
    assert "empty" not in check_page(_raw(""), blank, CFG).flags


def test_header_footer_do_not_count_as_content() -> None:
    raw = "![header](0,0,1000,50) " + "x" * 80 + "\n\n![text](0,60,1000,100) ok"
    assert "empty" in check_page(_raw(raw), _page(), CFG).flags


def test_parse_error_without_markers() -> None:
    res = check_page(
        _raw("just some prose without any block marker " * 3), _page(scanned=True), CFG
    )
    assert "parse_error" in res.flags


def test_bad_table_ragged_and_unparseable() -> None:
    ragged = (
        "![table](0,0,1000,500) <table><tr><td>a</td><td>b</td></tr><tr><td>c</td></tr></table>"
    )
    assert "bad_table" in check_page(_raw(ragged), _page(scanned=True), CFG).flags
    assert (
        "bad_table"
        in check_page(_raw("![table](0,0,10,10) not html"), _page(scanned=True), CFG).flags
    )


def test_valid_spans_are_not_ragged() -> None:
    html = (
        '<table><tr><th rowspan="2">G</th><th colspan="2">S</th></tr>'
        "<tr><th>m</th><th>s</th></tr><tr><td>A</td><td>1</td><td>2</td></tr></table>"
    )
    assert not table_is_bad(html)
    assert table_is_bad("<table></table>")


def test_low_recall_when_text_layer_differs_by_30_percent() -> None:
    vlm = "![text](0,0,1000,1000) " + " ".join(WORDS[:70])
    res = check_page(_raw(vlm), _page(), CFG)
    assert "low_recall" in res.flags and "low_precision" not in res.flags
    assert res.recall is not None and abs(res.recall - 0.70) < 1e-9 and res.escalate


def test_low_precision() -> None:
    vlm = _good() + " " + " ".join(f"fake{i}y" for i in range(60))
    res = check_page(_raw(vlm), _page(), CFG)
    assert "low_precision" in res.flags and "low_recall" not in res.flags


def test_warn_band_logs_but_does_not_escalate() -> None:
    vlm = "![text](0,0,1000,1000) " + " ".join(WORDS[:90])  # recall 0.90: between 0.85 and 0.93
    res = check_page(_raw(vlm), _page(), CFG)
    assert res.flags == ["low_agreement_warn"] and not res.escalate


def test_scanned_pages_skip_agreement() -> None:
    res = check_page(_raw(_good()), _page("", scanned=True), CFG)
    assert res.recall is None and res.flags == []


def test_math_page_not_flagged_for_agreement() -> None:
    from docparse.qa.checks import check_page

    raw = (
        "![text](0,0,1000,300) We prove that the operator is bounded on every compact set.\n\n"
        "![formula](0,300,1000,500) $$\\mathcal{V}(\\psi_m)\\rightarrow +\\infty$$\n"
    )
    layer = "We prove that the operator is bounded on every compact set. V(ψm) → +∞ ˆxη"
    page = {"page_id": "d:00001", "text_layer": layer, "text_layer_chars": 250, "scanned": False}
    res = check_page({"raw": raw, "finish_reason": "stop"}, page, CFG, 1)
    assert "low_recall" not in res.flags and "low_precision" not in res.flags


def test_header_text_counts_toward_recall() -> None:
    from docparse.qa.checks import check_page

    raw = (
        "![header](0,0,1000,50) Journal of Applied Statistics volume twelve\n\n"
        "![text](0,100,1000,900) The estimator converges under mild regularity conditions.\n"
    )
    layer = (
        "Journal of Applied Statistics volume twelve The estimator converges under mild regularity "
        "conditions."
    )
    page = {"page_id": "d:00001", "text_layer": layer, "text_layer_chars": 250, "scanned": False}
    res = check_page({"raw": raw, "finish_reason": "stop"}, page, CFG, 1)
    assert res.recall == 1.0 and "low_recall" not in res.flags
