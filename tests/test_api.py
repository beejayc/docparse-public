import json
import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from config.settings import Profile, load_profile
from docparse import api
from docparse.batch.httpx_runner import HttpxRunner
from docparse.batch.runner import RunStats
from docparse.cli import app

FILES = Path(__file__).parent / "fixtures" / "files"
WORDS = " ".join(f"wort{i}x" for i in range(30))


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "model": body["model"],
            "choices": [
                {"message": {"content": f"![text](0,0,1000,1000) {WORDS}"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        },
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "in"
    root.mkdir()
    for n in ("digital.pdf", "scanned.pdf", "blank.png"):
        shutil.copy(FILES / n, root / n)
    return root


@pytest.fixture
def mock_server(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the network with an in-process handler; returns the list of passes executed."""
    passes: list[int] = []
    p = load_profile("remote")
    served = [p.vllm.models.fast.id]
    monkeypatch.setattr(api, "served_models", lambda _e, timeout=10.0: served)

    def fake_run_pass(
        profile: Profile, data_root: Path, pass_no: int, runner: str = "x", limit: int | None = None
    ) -> RunStats:
        from docparse.batch.rows import iter_pending_pages

        passes.append(pass_no)
        rows = iter_pending_pages(data_root / "manifest", pass_no)
        out = data_root / "out" / "raw" / f"pass{pass_no}"
        prof = profile.model_copy(
            update={"render": profile.render.model_copy(update={"workers": 1})}
        )
        return HttpxRunner(prof, transport=httpx.MockTransport(_handler)).run(rows, out, pass_no)

    monkeypatch.setattr(api, "run_pass", fake_run_pass)
    return passes


def test_parse_corpus_end_to_end(corpus: Path, tmp_path: Path, mock_server: list[int]) -> None:
    out = tmp_path / "out"
    steps: list[str] = []
    s = api.parse_corpus(corpus, out, "remote", progress=steps.append)
    assert (s.docs, s.pages) == (3, 6)
    assert s.pass1 is not None and s.pass1["rows_processed"] == 6 and s.pass1["errors"] == 0
    assert steps[:3] == ["inventory", "normalise", "pass 1"] and steps[-1] == "assemble"
    docs = sorted(s.docs_dir.iterdir())
    assert len(docs) == 3
    for d in docs:
        assert (d / "document.md").is_file() and (d / "blocks.jsonl").is_file()
        assert (d / "meta.json").is_file()
    again = api.parse_corpus(corpus, out, "remote")  # idempotent re-run
    assert again.pass1 is not None and again.pass1["rows_processed"] == 0
    assert mock_server.count(1) == 2


def test_parse_corpus_warns_when_4b_not_served(
    corpus: Path, tmp_path: Path, mock_server: list[int]
) -> None:
    s = api.parse_corpus(corpus, tmp_path / "o", "remote")
    if s.escalated:  # the mock text disagrees with the scanned layer on some pages
        assert s.pass2 is None and any("not served" in w for w in s.warnings)
    assert s.to_dict()["profile"] == "remote"


def test_endpoint_argument_overrides_profile(
    corpus: Path, tmp_path: Path, mock_server: list[int]
) -> None:
    s = api.parse_corpus(corpus, tmp_path / "o", "remote", endpoint="http://gpu:9000/")
    assert s.endpoint == "http://gpu:9000"


def test_run_pass_fails_clearly_when_model_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "served_models", lambda _e, timeout=10.0: [])
    with pytest.raises(RuntimeError, match="not served"):
        api.run_pass(load_profile("remote"), tmp_path, 1)
    with pytest.raises(ValueError):
        api.run_pass(load_profile("remote"), tmp_path, 3)


def test_served_models_unreachable() -> None:
    assert api.served_models("http://127.0.0.1:9", timeout=0.5) == []


def test_cli_parse_command(corpus: Path, tmp_path: Path, mock_server: list[int]) -> None:
    res = CliRunner().invoke(
        app, ["--profile", "remote", "parse", str(corpus), str(tmp_path / "o"), "--no-escalate"]
    )
    assert res.exit_code == 0, res.output
    data: dict[str, Any] = json.loads(res.stdout)
    assert data["docs"] == 3 and data["pass2"] is None


def test_cli_parse_reports_missing_server(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "served_models", lambda _e, timeout=10.0: [])
    res = CliRunner().invoke(
        app, ["--profile", "remote", "parse", str(corpus), str(tmp_path / "o")]
    )
    assert res.exit_code == 1
