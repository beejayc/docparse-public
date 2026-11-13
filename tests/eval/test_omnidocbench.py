from docparse.eval.omnidocbench import edit_distance, gt_text, score_page, table_f1

PAGE = {
    "layout_dets": [
        {"category_type": "text_block", "order": 2, "text": "Second paragraph."},
        {"category_type": "title", "order": 1, "text": "A Title"},
        {"category_type": "text_block", "order": 3, "text": "ignored", "ignore": True},
        {"category_type": "table", "html": "<table><tr><td>a</td><td>b</td></tr></table>"},
    ]
}


def test_gt_text_is_reading_order_without_ignored() -> None:
    assert gt_text(PAGE) == "A Title\nSecond paragraph."


def test_edit_distance_bounds() -> None:
    assert edit_distance("abc", "abc") == 0.0
    assert edit_distance("", "abc") == 1.0
    assert 0 < edit_distance("abd", "abc") < 1


def test_table_f1_and_page_score() -> None:
    assert table_f1("| a | b |", [PAGE["layout_dets"][3]["html"]]) == 1.0
    assert table_f1("x", []) is None
    s = score_page(PAGE, "A Title\nSecond paragraph.\n\n| a | b |\n| --- | --- |\n")
    assert s["text_edit"] == 0.0 and s["table_f1"] == 1.0
