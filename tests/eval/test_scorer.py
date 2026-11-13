from docparse.eval.scorer import (
    check_absent,
    check_baseline,
    check_math,
    check_order,
    check_present,
    check_table,
    fuzzy_find,
    tables_of,
)

HTML = (
    "<table><tr><th>Name</th><th>Score</th></tr>"
    "<tr><td>Ann</td><td>0.569</td></tr><tr><td>Bo</td><td>0.7</td></tr></table>"
)
PIPE = "| Name | Score |\n| --- | --- |\n| Ann | 0.569 |\n| Bo | 0.7 |\n"


def test_present_exact_fuzzy_and_limits() -> None:
    doc = "The  quick brown fox\njumps over."
    assert check_present({"text": "quick brown fox", "max_diffs": 0}, doc)
    assert not check_present({"text": "quick brown fax", "max_diffs": 0}, doc)
    assert check_present({"text": "quick brown fax", "max_diffs": 1}, doc)
    assert not check_present({"text": "JUMPS", "case_sensitive": True}, doc)
    assert check_present({"text": "JUMPS", "case_sensitive": False}, doc)
    assert not check_present({"text": "jumps", "first_n": 10}, doc)


def test_absent_and_order() -> None:
    doc = "Header line\n\nBody text here.\n\nSecond part."
    assert check_absent({"text": "Footer", "max_diffs": 0}, doc)
    assert not check_absent({"text": "header line", "max_diffs": 0}, doc)
    assert check_order({"before": "Body text", "after": "Second part", "max_diffs": 0}, doc)
    assert not check_order({"before": "Second part", "after": "Body text", "max_diffs": 0}, doc)
    assert not check_order({"before": "Missing", "after": "Body text", "max_diffs": 0}, doc)


def test_fuzzy_find_reports_span() -> None:
    assert fuzzy_find("abc", "xxabcxx", 0) == (2, 5)
    assert fuzzy_find("abc", "xxabdxx", 0) is None


def test_tables_html_and_pipe_and_spans() -> None:
    assert tables_of(HTML)[0][1] == ["Ann", "0.569"]
    assert tables_of(PIPE)[0][2] == ["Bo", "0.7"]
    span = "<table><tr><td rowspan='2'>A</td><td>b</td></tr><tr><td>c</td></tr></table>"
    assert tables_of(span)[0] == [["A", "b"], ["A", "c"]]


def test_table_relations() -> None:
    for doc in (HTML, PIPE):
        assert check_table({"cell": "0.569", "left": "Ann", "top_heading": "Score"}, doc)
        assert check_table({"cell": "0.7", "up": "0.569", "left_heading": "Bo"}, doc)
        assert not check_table({"cell": "0.569", "left": "Bo"}, doc)
        assert not check_table({"cell": "nope"}, doc)


def test_math_proxy_normalises() -> None:
    doc = r"text $$ \left( x \leq 2^{N}-1 \right) $$ more"
    assert check_math({"math": r"(x \le 2^{N}-1)"}, doc)
    assert not check_math({"math": r"x \geq 3"}, doc)


def test_baseline() -> None:
    assert check_baseline({}, "some text")
    assert not check_baseline({}, "  ")
    assert not check_baseline({}, "ab " * 400)
