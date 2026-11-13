"""EML (stdlib) and MSG (python-oxmsg, not extract-msg which is GPL) -> body block + attachments."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from email import message_from_binary_file, policy
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from markdownify import markdownify

from docparse.normalise.native import make_block

log = logging.getLogger(__name__)

MIN_INLINE_IMAGE_BYTES = 10 * 1024


@dataclass
class ParsedEmail:
    headers: dict[str, str]
    body: str
    attachments: list[tuple[str, bytes, bool]] = field(default_factory=list)  # name, data, inline


def safe_name(name: str | None, fallback: str) -> str:
    base = re.split(r"[\\/]", name or "")[-1].strip().lstrip(".")
    return base or fallback


def _headers_table(headers: dict[str, str]) -> str:
    rows = [(k, v.replace("|", "\\|").replace("\n", " ")) for k, v in headers.items() if v]
    return "| Header | Value |\n| --- | --- |\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def parse_eml(path: Path) -> ParsedEmail:
    with path.open("rb") as fh:
        msg = message_from_binary_file(fh, policy=policy.default)
    assert isinstance(msg, EmailMessage)
    headers = {h: str(msg[h] or "") for h in ("From", "To", "Cc", "Date", "Subject")}
    part = msg.get_body(preferencelist=("plain", "html"))
    body = ""
    if part is not None:
        content = part.get_content()
        body = markdownify(content) if part.get_content_type() == "text/html" else content
    atts: list[tuple[str, bytes, bool]] = []
    for i, att in enumerate(msg.iter_attachments(), start=1):
        name = safe_name(att.get_filename(), f"attachment{i}")
        if att.get_content_type() == "message/rfc822":
            inner = att.get_payload(0)
            data = inner.as_bytes()  # type: ignore[union-attr]
            if not name.lower().endswith(".eml"):
                name += ".eml"
        else:
            data = att.get_payload(decode=True) or b""
        inline = att.get_content_disposition() == "inline" or bool(att.get("Content-ID"))
        atts.append((name, bytes(data), inline))
    return ParsedEmail(headers, body.strip(), atts)


def parse_msg(path: Path | Any) -> ParsedEmail:
    """``path`` may be a file path or an already-loaded oxmsg-like Message (tests)."""
    if isinstance(path, Path):
        from oxmsg import Message

        m = Message.load(str(path))
    else:
        m = path
    to = ", ".join(str(r) for r in m.recipients)
    date = m.sent_date
    headers = {
        "From": str(m.sender or ""),
        "To": to,
        "Cc": "",
        "Date": date.isoformat() if date else "",
        "Subject": m.subject or "",
    }
    body = m.body or ""
    if not body.strip() and m.html_body:
        body = markdownify(m.html_body)
    atts = []
    for i, a in enumerate(m.attachments, start=1):
        if a.file_bytes is None:
            continue
        name = safe_name(a.file_name, f"attachment{i}")
        atts.append(
            (name, bytes(a.file_bytes), True)
        )  # MSG gives no inline flag; size filter decides
    return ParsedEmail(headers, body.strip(), atts)


def keep_attachment(data: bytes, name: str, inline: bool) -> bool:
    """Drop empty files and small inline images (logos, tracking pixels)."""
    if not data:
        return False
    is_image = name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"))
    return not (is_image and inline and len(data) < MIN_INLINE_IMAGE_BYTES)


def email_blocks(doc_id: str, parsed: ParsedEmail) -> list[dict[str, Any]]:
    md = _headers_table(parsed.headers) + "\n\n" + parsed.body
    text = "\n".join(f"{k}: {v}" for k, v in parsed.headers.items() if v) + "\n\n" + parsed.body
    return [make_block(doc_id, 1, 1, "email_body", text.strip(), md.strip(), "email:body")]
