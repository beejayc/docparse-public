import os
import subprocess
import sys
from pathlib import Path

import pytest

from docparse import osutil


def test_lower_priority_without_nice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "nice", raising=False)
    assert osutil.lower_priority(10) is False


def test_lower_priority_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_n: int) -> int:
        raise PermissionError

    monkeypatch.setattr(os, "nice", boom, raising=False)
    assert osutil.lower_priority(-5) is False


def test_pin_cores_noop_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    assert osutil.pin_cores(None) is False
    assert osutil.pin_cores([]) is False
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)
    assert osutil.pin_cores([0]) is False


def test_pin_cores_bad_core_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_pid: int, _cores: set[int]) -> None:
        raise OSError("no such cpu")

    monkeypatch.setattr(os, "sched_setaffinity", boom, raising=False)
    assert osutil.pin_cores([999]) is False


def test_popen_group_kwargs_per_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert osutil.popen_group_kwargs() == {"start_new_session": True}
    monkeypatch.setattr(sys, "platform", "win32")
    assert "creationflags" in osutil.popen_group_kwargs()
    monkeypatch.setattr(sys, "platform", "darwin")
    assert osutil.popen_group_kwargs() == {"start_new_session": True}


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX process groups")
def test_kill_process_tree_kills_children() -> None:
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", "import time; time.sleep(60)"], **osutil.popen_group_kwargs()
    )
    osutil.kill_process_tree(proc)
    assert proc.wait(timeout=10) != 0
    osutil.kill_process_tree(proc)  # already gone: must not raise


def test_soffice_binary_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert osutil.soffice_binary("/opt/lo/soffice") == "/opt/lo/soffice"
    monkeypatch.setenv("DOCPARSE_SOFFICE", "/x/soffice")
    assert osutil.soffice_binary() == "/x/soffice"
    monkeypatch.delenv("DOCPARSE_SOFFICE")
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/soffice" if n == "soffice" else None)
    assert osutil.soffice_binary() == "/usr/bin/soffice"


def test_file_uri_is_valid_everywhere(tmp_path: Path) -> None:
    uri = osutil.file_uri(tmp_path / "lo profile")
    assert uri.startswith("file:///") and "%20" in uri
