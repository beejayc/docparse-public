import pytest
from typer.testing import CliRunner

from config.settings import PROFILES_DIR, Settings, load_profile
from docparse.cli import app

NAMES = sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def test_all_profiles_present() -> None:
    assert NAMES == sorted(["consumer24", "generic", "gb10", "pro6000", "remote"])


@pytest.mark.parametrize("name", NAMES)
def test_profile_loads(name: str) -> None:
    profile = load_profile(name)
    assert profile.name == name
    assert profile.vllm.models.fast.id == "lightonai/LightOnOCR-3-0.8B"
    assert profile.sampling.max_tokens.pass2 == 8192


def test_unknown_profile() -> None:
    with pytest.raises(FileNotFoundError):
        load_profile("nope")


def test_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCPARSE_PROFILE", "gb10")
    assert Settings().profile == "gb10"


def test_cli_help_lists_commands() -> None:
    out = CliRunner().invoke(app, ["--help"]).output
    for cmd in [
        "inventory",
        "normalise",
        "smoke",
        "run",
        "assemble",
        "watch",
        "qa",
        "eval",
        "report",
        "bench",
    ]:
        assert cmd in out


def test_cli_stub_exits_2() -> None:
    assert CliRunner().invoke(app, ["bench"]).exit_code == 2


def test_endpoint_env_overrides_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCPARSE_ENDPOINT", "http://gpu-box:9000/")
    assert load_profile("remote").endpoint == "http://gpu-box:9000"
    monkeypatch.delenv("DOCPARSE_ENDPOINT")
    assert load_profile("remote").endpoint == "http://localhost:8000"


def test_portable_profiles_do_not_pin_cpus() -> None:
    for name in ("generic", "remote"):
        p = load_profile(name)
        assert (p.vllm.cpuset, p.render.cpuset, p.cpu_other.cpuset) == ("", "", "")
