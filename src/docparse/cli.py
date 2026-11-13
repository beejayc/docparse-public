"""docparse CLI skeleton. Commands are stubs until their owning phase lands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, NoReturn

import typer

app = typer.Typer(help="Local, batch-first document parser for RAG.", no_args_is_help=True)

ProfileOpt = Annotated[
    str, typer.Option("--profile", envvar="DOCPARSE_PROFILE", help="Hardware profile.")
]


@app.callback()
def main(
    ctx: typer.Context,
    profile: ProfileOpt = "generic",
    data_root: Annotated[
        Path, typer.Option("--data-root", envvar="DOCPARSE_DATA_ROOT", help="Data root.")
    ] = Path("data"),
) -> None:
    ctx.obj = {"profile": profile, "data_root": data_root}


def _stub(phase: int) -> NoReturn:
    typer.echo(f"not implemented (P{phase})", err=True)
    raise typer.Exit(code=2)


@app.command()
def inventory(ctx: typer.Context, root: Path) -> None:
    """Detect, hash and dedup input files into the manifest."""
    from config.settings import load_profile
    from docparse.inventory.manifest import write_manifest

    profile = load_profile(ctx.obj["profile"])
    workers = min(profile.cpu_other.workers, os.cpu_count() or 1)
    n_docs, n_pages = write_manifest(root, ctx.obj["data_root"], workers)
    typer.echo(f"inventory: {n_docs} docs, {n_pages} pages -> {ctx.obj['data_root'] / 'manifest'}")


@app.command()
def normalise(ctx: typer.Context) -> None:
    """Convert Office/email/ZIP inputs to PDFs and native blocks."""
    from config.settings import Settings
    from docparse.normalise import normalise_all

    stats = normalise_all(Settings(data_root=ctx.obj["data_root"], profile=ctx.obj["profile"]))
    typer.echo(f"normalise: {stats}")


@app.command()
def smoke(
    ctx: typer.Context,
    pages: Annotated[int, typer.Option("--pages")] = 5,
    model: Annotated[str, typer.Option("--model")] = "lightonai/LightOnOCR-3-0.8B",
) -> None:
    """Smoke-test the vLLM server and save grounding fixtures."""
    from config.settings import load_profile
    from docparse.vlm.smoke import run_smoke

    profile = load_profile(ctx.obj["profile"])
    for meta in run_smoke(profile, ctx.obj["data_root"], pages, model):
        typer.echo(
            f"{meta['name']}: {meta['finish_reason']} "
            f"{meta['completion_tokens']} tok {meta['latency_s']}s"
        )


@app.command()
def run(
    ctx: typer.Context,
    pass_: Annotated[int, typer.Option("--pass", help="1 or 2.")],
    runner: Annotated[str, typer.Option("--runner", help="saturate | httpx")] = "saturate",
    limit: Annotated[
        int | None, typer.Option("--limit", help="Only the first N pending pages.")
    ] = None,
) -> None:
    """Run an OCR pass (safe to kill and re-run; resumes exactly)."""
    from config.settings import load_profile
    from docparse.api import run_pass

    if pass_ not in (1, 2):
        raise typer.BadParameter("--pass must be 1 or 2")
    profile = load_profile(ctx.obj["profile"])
    try:
        stats = run_pass(profile, ctx.obj["data_root"], pass_, runner, limit)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(stats.to_json())


@app.command()
def assemble(
    ctx: typer.Context,
    doc: Annotated[str | None, typer.Option("--doc", help="Only this doc_id.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-assemble finished docs.")] = False,
) -> None:
    """Assemble final per-document output."""
    import json

    from config.settings import load_profile
    from docparse.assemble import assemble_all

    profile = load_profile(ctx.obj["profile"])
    counts = assemble_all(
        ctx.obj["data_root"], profile, only=[doc] if doc else None, force=force or doc is not None
    )
    typer.echo(json.dumps(counts))


@app.command()
def watch(
    ctx: typer.Context,
    interval: Annotated[float, typer.Option("--interval", help="Seconds between cycles.")] = 300.0,
    once: Annotated[bool, typer.Option("--once", help="One cycle, then exit.")] = False,
) -> None:
    """Incremental assembly while the GPU pass runs."""
    import json

    from config.settings import load_profile
    from docparse.watch import watch as run_watch

    profile = load_profile(ctx.obj["profile"])
    for counts in run_watch(ctx.obj["data_root"], profile, interval, once):
        typer.echo(json.dumps(counts))


@app.command()
def qa(
    ctx: typer.Context,
    pass_: Annotated[int, typer.Option("--pass", help="1 or 2.")],
    force: Annotated[
        bool, typer.Option("--force", help="Re-check pages already in the QA file.")
    ] = False,
) -> None:
    """Run QA checks on raw output (writes out/qa/pass<N>.parquet)."""
    import json

    from config.settings import load_profile
    from docparse.qa.run import run_qa

    profile = load_profile(ctx.obj["profile"])
    workers = min(profile.cpu_other.workers, os.cpu_count() or 1)
    typer.echo(json.dumps(run_qa(ctx.obj["data_root"], pass_, force=force, workers=workers)))


@app.command(name="eval")
def eval_(
    ctx: typer.Context,
    suite: Annotated[str, typer.Argument(help="olmocr | calibrate | german")],
    bench_dir: Annotated[
        Path, typer.Option("--bench-dir", help="olmOCR-Bench bench_data directory.")
    ] = Path("dataset/olmOCR-bench/bench_data"),
    source: Annotated[
        str, typer.Option("--source", help="olmocr: document | pass1 | pass2.")
    ] = "document",
    pdf_dir: Annotated[
        Path | None, typer.Option("--pdf-dir", help="german: directory of PDFs.")
    ] = None,
) -> None:
    """Run an evaluation suite (writes out/reports/)."""
    from docparse.eval.run import run_suite

    typer.echo(run_suite(suite, ctx.obj["data_root"], bench_dir, source, pdf_dir))


@app.command()
def report(ctx: typer.Context, run_id: str) -> None:
    """Write the per-run report to out/reports/run-<run_id>.md."""
    from docparse.eval.report import build_report

    out = ctx.obj["data_root"] / "out" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"run-{run_id}.md"
    path.write_text(build_report(ctx.obj["data_root"], run_id), encoding="utf-8")
    typer.echo(str(path))


@app.command()
def bench(
    ctx: typer.Context,
    sample_roots: Annotated[
        list[Path], typer.Option("--sample-root", help="Data roots to draw the sample from.")
    ],
    perf_cpuset: Annotated[str, typer.Option(help="Performance cores for rendering.")],
    eff_cpuset: Annotated[str | None, typer.Option(help="Efficiency cores to compare.")] = None,
    hourly_cost: Annotated[float | None, typer.Option(help="Hardware cost per hour.")] = None,
    sample_pages: Annotated[int, typer.Option()] = 2000,
    sweep_pages: Annotated[int, typer.Option()] = 400,
    expected_pps: Annotated[float, typer.Option(help="Start value for the worker formula.")] = 1.0,
    work_dir: Annotated[Path, typer.Option()] = Path(".scratch/bench"),
) -> None:
    """Hardware benchmark: render speed, worker sweep, full sample run (needs the GPU server)."""
    from config.settings import load_profile
    from docparse.bench.full import run_full, write_measured
    from docparse.bench.sample import build_sample

    profile = load_profile(ctx.obj["profile"])
    sample_root = work_dir / "sample"
    meta = build_sample(sample_roots, sample_root, n=sample_pages)
    typer.echo(f"sample: {meta['pages']} pages {meta['composition']}")
    reports = ctx.obj["data_root"] / "out" / "reports"
    res = run_full(
        profile, sample_root, work_dir, reports, hourly_cost, perf_cpuset, eff_cpuset,
        expected_pps, sweep_pages, sample_pages,
    )  # fmt: skip
    mode = profile.render.mode
    write_measured(
        Path("config/profiles") / f"{profile.name}.yaml",
        {
            "render_s_per_page": res["render_s_per_page"][f"{mode}@perf"]["mean_s"],
            "pages_per_s": res["full"]["pages_per_s"],
        },
        res["chosen_workers"],
    )
    typer.echo(str(reports / f"bench-{profile.name}.md"))


@app.command()
def parse(
    ctx: typer.Context,
    input_dir: Annotated[Path, typer.Argument(help="Folder of documents to parse.")],
    output_dir: Annotated[
        Path, typer.Argument(help="Output (data root); docs go to <out>/out/docs.")
    ],
    endpoint: Annotated[
        str | None, typer.Option("--endpoint", help="vLLM server URL (overrides the profile).")
    ] = None,
    no_escalate: Annotated[
        bool, typer.Option("--no-escalate", help="Skip pass 2 (4B model).")
    ] = False,
) -> None:
    """Run every stage on a folder: inventory, normalise, OCR, QA, assemble."""
    import json

    from docparse.api import parse_corpus

    try:
        summary = parse_corpus(
            input_dir,
            output_dir,
            ctx.obj["profile"],
            endpoint=endpoint,
            escalate=not no_escalate,
            progress=lambda m: typer.echo(f"[docparse] {m}", err=True),
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    for w in summary.warnings:
        typer.echo(f"warning: {w}", err=True)
    typer.echo(json.dumps(summary.to_dict(), indent=2))
