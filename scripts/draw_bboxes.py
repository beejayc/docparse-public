"""Draw grounding bboxes and labels on a fixture PNG: ``python scripts/draw_bboxes.py <fixture>``.

``<fixture>`` is a path with or without extension, e.g. tests/fixtures/grounding/abc_p1.
Output goes to ``.scratch/bboxes/<name>.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from docparse.vlm.grounding import parse_grounding


def draw(fixture: Path, out_dir: Path = Path(".scratch/bboxes")) -> Path:
    base = fixture.with_suffix("") if fixture.suffix in {".png", ".txt", ".json"} else fixture
    base = Path(str(base).removesuffix(".raw"))
    img = Image.open(f"{base}.png").convert("RGB")
    raw = Path(f"{base}.raw.txt").read_text(encoding="utf-8")
    blocks, _ = parse_grounding(raw)
    d = ImageDraw.Draw(img)
    w, h = img.size
    for i, b in enumerate(blocks):
        if b.bbox is None:
            continue
        x1, y1, x2, y2 = (
            b.bbox[0] * w / 1000,
            b.bbox[1] * h / 1000,
            b.bbox[2] * w / 1000,
            b.bbox[3] * h / 1000,
        )
        d.rectangle((x1, y1, x2, y2), outline=(220, 30, 30), width=3)
        d.text((x1 + 3, max(0, y1 - 12)), f"{i}:{b.label}{'+' if b.cont else ''}", fill=(0, 0, 200))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{base.name}.png"
    img.save(out)
    return out


if __name__ == "__main__":
    sys.stdout.write(f"{draw(Path(sys.argv[1]))}\n")
