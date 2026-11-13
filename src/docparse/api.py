"""Stable Python API: ``parse_corpus`` runs every stage (ARCHITECTURE section 8, ADR-031)."""

from __future__ import annotations

import itertools
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from config.settings import Profile, Settings, load_profile
from docparse.batch.runner import RunStats

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass
class RunSummary:
    input_dir: str
    output_dir: str
    profile: str
    endpoint: str
    docs: int = 0
    pages: int = 0
    pass1: dict[str, Any] | None = None
    escalated: int = 0
    pass2: dict[str, Any] | None = None
    assembled: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def docs_dir(self) -> Path:
        return Path(self.output_dir) / "out" / "docs"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def served_models(endpoint: str, timeout: float = 10.0) -> list[str]:
    """Model ids the vLLM server at ``endpoint`` reports; [] when unreachable."""
    try:
        data = httpx.get(f"{endpoint}/v1/models", timeout=timeout).json()["data"]
        return [str(m["id"]) for m in data]
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        return []


def run_pass(
    profile: Profile,
    data_root: Path,
    pass_no: int,
    runner: str = "saturate",
    limit: int | None = None,
) -> RunStats:
    """One OCR pass over the pending pages (pass 2: only pages QA escalated)."""
    from docparse.batch.rows import iter_pending_pages
    from docparse.qa.store import escalated_ids

    if pass_no not in (1, 2):
        raise ValueError("pass_no must be 1 or 2")
    model = (profile.vllm.models.fast if pass_no == 1 else profile.vllm.models.accurate).id
    if model not in served_models(profile.endpoint):
        hint = "fast" if pass_no == 1 else "accurate"
        raise RuntimeError(
            f"{model} is not served at {profile.endpoint}: start it (make up MODEL={hint}) "
            "or pass --endpoint"
        )
    ids = escalated_ids(data_root / "out", 1) if pass_no == 2 else None
    rows = iter_pending_pages(data_root / "manifest", pass_no, ids)
    if limit is not None:
        rows = itertools.islice(rows, limit)
    output = data_root / "out" / "raw" / f"pass{pass_no}"
    if runner == "httpx":
        from docparse.batch.httpx_runner import HttpxRunner

        return HttpxRunner(profile).run(rows, output, pass_no)
    from docparse.batch.saturate_runner import SaturateRunner

    return SaturateRunner(profile).run(rows, output, pass_no)


def parse_corpus(
    input_dir: Path,
    output_dir: Path,
    profile: str | Profile = "generic",
    *,
    endpoint: str | None = None,
    escalate: bool = True,
    runner: str = "saturate",
    progress: Progress | None = None,
) -> RunSummary:
    """Inventory -> normalise -> pass 1 -> QA -> (pass 2 -> QA) -> assemble.

    ``output_dir`` is the data root: documents land in ``output_dir/out/docs/<doc_id>/``.
    Safe to re-run; every stage resumes. Pass 2 needs the 4B model on the server; when it is not
    served the run still finishes, pages stay provisional (``escalation_pending``) and a warning
    says how to finish them.
    """
    from docparse.assemble import assemble_all
    from docparse.inventory.manifest import write_manifest
    from docparse.normalise import normalise_all
    from docparse.qa.run import run_qa

    say = progress or (lambda _m: None)
    prof = load_profile(profile) if isinstance(profile, str) else profile
    if endpoint:
        prof = prof.model_copy(update={"endpoint": endpoint.rstrip("/")})
    out = Path(output_dir)
    t0 = time.monotonic()
    summary = RunSummary(str(input_dir), str(out), prof.name, prof.endpoint)
    workers = max(1, min(prof.cpu_other.workers, os.cpu_count() or 1))

    say("inventory")
    summary.docs, summary.pages = write_manifest(Path(input_dir), out, workers)
    say("normalise")
    normalise_all(Settings(data_root=out, profile=prof.name))
    say("pass 1")
    summary.pass1 = asdict(run_pass(prof, out, 1, runner))
    run_qa(out, 1, workers=workers, nice=prof.cpu_other.nice)

    from docparse.qa.store import escalated_ids

    summary.escalated = len(escalated_ids(out / "out", 1))
    if escalate and summary.escalated:
        say(f"pass 2 ({summary.escalated} pages)")
        if prof.vllm.models.accurate.id in served_models(prof.endpoint):
            summary.pass2 = asdict(run_pass(prof, out, 2, runner))
            run_qa(out, 2, workers=workers, nice=prof.cpu_other.nice)
        else:
            summary.warnings.append(
                f"{summary.escalated} pages need {prof.vllm.models.accurate.id}, which is not "
                "served: output keeps the pass-1 text for them (flag escalation_pending). "
                "Serve the 4B model, then run `docparse run --pass 2 && docparse qa --pass 2 "
                "&& docparse assemble --force`."
            )
    say("assemble")
    summary.assembled = assemble_all(out, prof)
    summary.elapsed_s = round(time.monotonic() - t0, 2)
    return summary
