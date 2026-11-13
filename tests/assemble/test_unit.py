from docparse.assemble.blocks import bbox_to_norm, norm_to_pt, strip_markup
from docparse.assemble.crops import crop_box
from docparse.assemble.merge import merge_grids
from docparse.assemble.tables import (
    grid_to_html,
    grid_to_pipe,
    is_nested,
    is_simple,
    table_grid,
)

ROWSPAN = (
    '<table><thead><tr><th rowspan="2">G</th><th colspan="2">Score</th></tr>'
    "<tr><th>Mean</th><th>Std</th></tr></thead>"
    "<tbody><tr><td>A</td><td>1</td><td>2</td></tr></tbody></table>"
)


def test_spans_expanded_with_origin() -> None:
    g = table_grid(ROWSPAN)
    assert (g.n_rows, g.n_cols, g.header_rows) == (3, 3, 2)
    assert g.cells[0][0].row_span == 2 and g.cells[0][0].origin is None
    assert g.cells[1][0].text == "G" and g.cells[1][0].origin == (0, 0)
    assert g.cells[0][2].text == "Score" and g.cells[0][2].origin == (0, 1)
    assert not is_simple(g, is_nested(ROWSPAN))


def test_html_roundtrip_keeps_spans() -> None:
    g = table_grid(ROWSPAN)
    assert table_grid(grid_to_html(g)) == g


def test_simple_table_pipe_and_header_detection() -> None:
    html = "<table><tr><th>a</th><th>b|c</th></tr><tr><td>1</td><td>2</td></tr></table>"
    g = table_grid(html)
    assert g.header_rows == 1 and is_simple(g)
    assert grid_to_pipe(g).splitlines() == ["| a | b\\|c |", "| --- | --- |", "| 1 | 2 |"]


def test_nested_table_is_not_simple() -> None:
    html = "<table><tr><td><table><tr><td>x</td></tr></table></td></tr></table>"
    assert is_nested(html) and not is_simple(table_grid(html), True)


def test_merge_drops_repeated_header_and_checks_columns() -> None:
    a = table_grid(
        "<table><thead><tr><th>h1</th><th>h2</th></tr></thead><tr><td>1</td><td>2</td></tr></table>"
    )
    b = table_grid(
        "<table><thead><tr><th>h1</th><th>h2</th></tr></thead><tr><td>3</td><td>4</td></tr></table>"
    )
    m = merge_grids(a, b)
    assert m is not None and m.n_rows == 3 and m.header_rows == 1
    assert [r[0].text for r in m.cells] == ["h1", "1", "3"]
    wide = table_grid("<table><tr><td>1</td><td>2</td><td>3</td></tr></table>")
    assert merge_grids(a, wide) is None


def test_strip_markup() -> None:
    assert strip_markup("text", "**bold** and *it*") == "bold and it"
    assert strip_markup("list", "- a\n- b") == "a\nb"
    assert strip_markup("formula", "$$\nE=mc^2\n$$") == "E=mc^2"
    assert strip_markup("table", "<tr><td>a</td><td>b</td></tr>") == "a | b |"


def test_bbox_conventions() -> None:
    assert bbox_to_norm((100, 200, 300, 400)) == (0.1, 0.2, 0.3, 0.4)
    # PDF points, origin bottom-left: a box in the top-left of the page has a high bottom value
    left, bottom, right, top = norm_to_pt((0.1, 0.0, 0.5, 0.1), 600.0, 800.0)
    assert (left, right) == (60.0, 300.0) and (bottom, top) == (720.0, 800.0)
    assert bbox_to_norm((500, 900, 100, 100)) == (0.1, 0.1, 0.5, 0.9)  # inverted -> sorted
    assert bbox_to_norm((0, 0, 1200, 1000))[2] == 1.0  # clamped


def test_crop_box_pad_and_clamp() -> None:
    assert crop_box((0.5, 0.5, 0.75, 0.75), (1000, 2000)) == (492, 992, 758, 1508)
    assert crop_box((0.0, 0.0, 1.0, 1.0), (100, 100)) == (0, 0, 100, 100)
