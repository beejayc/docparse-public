from pathlib import Path

import pytest

from docparse.vlm.grounding import RawBlock, parse_grounding

FIXTURES = sorted(Path(__file__).parents[1].glob("fixtures/grounding/*.raw.txt"))

SYNTH = """![title](80,70,920,140)
# Company report 2025

![text](80,175,920,225)
Revenue rose 20%.

![image](80,285,460,545)
A solar-powered factory.

![chart](540,285,920,545)
<table>
  <tr><th>Year</th></tr>
</table>
"""


def test_synthetic_blocks() -> None:
    blocks, warns = parse_grounding(SYNTH)
    assert warns == []
    assert [b.label for b in blocks] == ["title", "text", "image", "chart"]
    assert blocks[0] == RawBlock("title", False, (80, 70, 920, 140), "# Company report 2025")
    assert blocks[3].content.startswith("<table>")


def test_continuation_both_placements() -> None:
    blocks, warns = parse_grounding(
        "![table+](0,0,1000,500)\n<table></table>\n![text](1,2,3,4)+\nx"
    )
    assert [b.cont for b in blocks] == [True, True]
    assert warns == []


def test_leading_text_and_unknown_label() -> None:
    blocks, warns = parse_grounding("stray\n![weird](1,2,3,4)\nbody")
    assert warns == ["parse_error", "unknown_label:weird"]
    assert blocks[0].bbox is None and blocks[1].label == "text"


def test_bad_bbox_and_empty() -> None:
    _, warns = parse_grounding("![text](10,10,5,5)\nx")
    assert warns == ["bad_bbox"]
    assert parse_grounding("") == ([], [])


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_real_fixtures(path: Path) -> None:
    blocks, warns = parse_grounding(path.read_text(encoding="utf-8"))
    assert blocks
    assert "parse_error" not in warns
