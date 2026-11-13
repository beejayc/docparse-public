"""DOCX/PPTX -> PDF through headless LibreOffice (MPL, separate process only)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from docparse.osutil import file_uri, kill_process_tree, popen_group_kwargs, soffice_binary

log = logging.getLogger(__name__)


class ConversionError(RuntimeError):
    pass


def convert_to_pdf(
    src: Path,
    out_dir: Path,
    worker_id: str | int,
    timeout: float = 120,
    soffice_bin: str = "soffice",
) -> Path:
    """Convert ``src`` into ``out_dir/{src.stem}.pdf``. Each worker has its own profile dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.gettempdir()) / f"lo_{worker_id}"
    cmd = [
        soffice_binary(soffice_bin),
        "--headless",
        "--norestore",
        f"-env:UserInstallation={file_uri(profile)}",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(src),
    ]
    proc = subprocess.Popen(  # noqa: S603
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **popen_group_kwargs()
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        kill_process_tree(proc)
        proc.communicate()
        raise ConversionError(f"timeout after {timeout:g}s") from exc
    pdf = out_dir / f"{src.stem}.pdf"
    if proc.returncode != 0 or not pdf.is_file():
        tail = out.decode("utf-8", "replace").strip()[-200:]
        raise ConversionError(f"soffice exit {proc.returncode}: {tail}")
    return pdf


def convert_doc(
    doc_id: str,
    detected_type: str,
    src: str,
    converted_dir: str,
    timeout: float,
    soffice_bin: str = "soffice",
) -> str:
    """Pool task: convert to ``{converted_dir}/{doc_id}.pdf``; raises ConversionError."""
    with tempfile.TemporaryDirectory(prefix="docparse_lo_") as tmp:
        link = Path(tmp) / f"{doc_id}.{detected_type}"  # extension drives LibreOffice's filter
        shutil.copyfile(src, link)
        pdf = convert_to_pdf(link, Path(tmp) / "out", os.getpid(), timeout, soffice_bin)
        dest = Path(converted_dir) / f"{doc_id}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pdf), dest)
    return str(dest)
