"""OS differences in one place: priority, CPU pinning, process groups, LibreOffice (ADR-031)."""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MAC_SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
_WIN_SOFFICE = (
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
)


def is_windows() -> bool:
    return sys.platform.startswith("win")


def lower_priority(nice: int) -> bool:
    """Raise the niceness of this process (POSIX only). False when unsupported or refused."""
    if not hasattr(os, "nice"):
        return False
    try:
        os.nice(nice)
    except OSError:
        return False
    return True


def pin_cores(cores: Iterable[int] | None) -> bool:
    """Restrict this process to ``cores`` (Linux only). False when unsupported."""
    wanted = set(cores or ())
    if not wanted or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, wanted)
    except OSError as exc:  # core ids beyond this machine, container limits
        log.warning("cannot pin to cores %s: %s", sorted(wanted), exc)
        return False
    return True


def popen_group_kwargs() -> dict[str, Any]:
    """Popen arguments that put the child in its own process group / session."""
    if is_windows():
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill ``proc`` and its children (soffice.bin is a child of the soffice wrapper)."""
    if is_windows():
        subprocess.run(  # noqa: S603
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],  # noqa: S607
            capture_output=True,
            check=False,
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def soffice_binary(name: str = "soffice") -> str:
    """Resolve LibreOffice: explicit name, ``DOCPARSE_SOFFICE``, PATH, then usual install dirs."""
    env = os.environ.get("DOCPARSE_SOFFICE")
    if name != "soffice":
        return name
    if env:
        return env
    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found:
        return found
    for cand in (_MAC_SOFFICE, *_WIN_SOFFICE):
        if Path(cand).is_file():
            return cand
    return name  # let Popen raise a clear FileNotFoundError


def file_uri(path: Path) -> str:
    """``file:///...`` URI valid on every OS (LibreOffice ``-env:UserInstallation``)."""
    return path.resolve().as_uri()
