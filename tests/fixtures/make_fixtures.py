"""Generate fixtures (reportlab, dev-only): uv run python tests/fixtures/make_fixtures.py"""

from __future__ import annotations

import random
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).parent / "files"
LOREM = (
    "The quarterly report summarises revenue, operating costs and headcount across all regions. "
    "Die Ergebnisse des Quartals zeigen ein stabiles Wachstum in allen Regionen. "
) * 6


def make_digital(path: Path) -> None:
    styles = getSampleStyleSheet()
    rows = [["Region", "Revenue", "Costs"]] + [
        [f"R{i}", str(i * 100), str(i * 40)] for i in range(1, 36)
    ]
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    story = [
        Paragraph("Quarterly report", styles["Title"]),
        Paragraph(LOREM, styles["BodyText"]),
        PageBreak(),
        Paragraph("Regional table", styles["Heading1"]),
        Paragraph(LOREM, styles["BodyText"]),
        Spacer(1, 360),  # pushes the table across the page 2 -> 3 break
        table,
    ]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def make_scanned(src: Path, path: Path) -> None:
    rng = random.Random(0)
    doc = pdfium.PdfDocument(str(src))
    frames = []
    for i in range(2):
        img = doc[i].render(scale=1.0).to_pil().convert("L")
        px = img.load()
        assert px is not None
        for _ in range(4000):
            x, y = rng.randrange(img.width), rng.randrange(img.height)
            px[x, y] = rng.randrange(256)
        frames.append(img.convert("RGB"))
    doc.close()
    frames[0].save(path, "PDF", save_all=True, append_images=frames[1:], resolution=72, quality=40)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_digital(OUT / "digital.pdf")
    make_scanned(OUT / "digital.pdf", OUT / "scanned.pdf")
    Image.new("RGB", (800, 1200), "white").save(OUT / "blank.png")
