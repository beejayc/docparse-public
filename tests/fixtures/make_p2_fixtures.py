"""P2 fixtures: xlsx, csv, eml(+pdf attachment), nested zips. memo.docx/slides.pptx were made once
with `soffice --convert-to docx|pptx` from a tiny HTML/flat-ODP source and are committed as is.
MSG has no permissive writer, so tests use a stub oxmsg Message.

    uv run python tests/fixtures/make_p2_fixtures.py
"""

from __future__ import annotations

import io
import zipfile
from email.message import EmailMessage
from pathlib import Path

from openpyxl import Workbook

OUT = Path(__file__).parent / "files" / "p2"


def make_xlsx(path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Sales"
    ws.append(["Region", "Q1", "Q2"])
    for i in range(1, 6):
        ws.append([f"R{i}", i * 10, i * 12])
    ws.append([])
    ws.append([])
    ws.append(["Total", "|pipe|", None, "far"])  # second range, leading blank rows
    ws2 = wb.create_sheet("Empty")
    ws2["A1"] = None
    ws3 = wb.create_sheet("Notes")
    ws3["B2"] = "hello"
    ws3["C3"] = "world"
    wb.save(path)


def make_eml(path: Path, pdf: bytes) -> None:
    m = EmailMessage()
    m["From"] = "Alice <alice@example.com>"
    m["To"] = "bob@example.com"
    m["Cc"] = "carol@example.com"
    m["Date"] = "Thu, 12 Nov 2026 10:00:00 +0000"
    m["Subject"] = "Report attached"
    m.set_content("Hi Bob,\nplease see the attached report.\n")
    m.add_attachment(pdf, maintype="application", subtype="pdf", filename="report.pdf")
    m.add_attachment(
        b"GIF89a" + b"\0" * 50,
        maintype="image",
        subtype="gif",
        filename="logo.gif",
        disposition="inline",
        cid="<logo@x>",
    )
    path.write_bytes(bytes(m))


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = (OUT.parent / "digital.pdf").read_bytes()
    make_xlsx(OUT / "sales.xlsx")
    (OUT / "table.csv").write_text("name;qty\nbolt;10\nnut;20\n", encoding="utf-8")
    make_eml(OUT / "mail.eml", pdf)
    inner = zip_bytes({"inner.csv": b"a,b\n1,2\n"})
    (OUT / "nested.zip").write_bytes(zip_bytes({"top.csv": b"x,y\n3,4\n", "sub/in.zip": inner}))
    (OUT / "evil.zip").write_bytes(zip_bytes({"ok.csv": b"p,q\n5,6\n", "../evil.txt": b"owned"}))


if __name__ == "__main__":
    main()
