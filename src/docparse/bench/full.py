"""``docparse bench``: sample -> render speed -> worker sweep -> core-type check -> full run."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from config.settings import Profile
from docparse.batch.runner import parse_cpuset
from docparse.bench.render_speed import render_s_per_page
from docparse.bench.run import BenchResult, choose_workers, cost_per_1000, run_bench, workers_for


def sweep_values(centre: int, span: int = 2) -> list[int]:
    return sorted({max(1, centre + d) for d in range(-span, span + 1)})


def run_full(
    profile: Profile,
    sample_root: Path,
    work_root: Path,
    reports: Path,
    hourly_cost: float | None,
    perf_cpuset: str,
    eff_cpuset: str | None,
    expected_pps: float,
    sweep_pages: int,
    full_pages: int,
) -> dict[str, Any]:
    core = parse_cpuset(perf_cpuset)
    assert core, "perf_cpuset required"
    speeds = {
        f"{mode}@perf": render_s_per_page(sample_root, core[0], mode)
        for mode in ("direct", "supersample")
    }
    if eff_cpuset:
        ecore = parse_cpuset(eff_cpuset) or [core[0]]
        speeds["direct@eff"] = render_s_per_page(sample_root, ecore[0], "direct")
    mode = profile.render.mode
    s_page = speeds[f"{mode}@perf"]["mean_s"]
    centre = workers_for(expected_pps, s_page)
    sweep: list[BenchResult] = []
    for w in sweep_values(centre):
        sweep.append(
            run_bench(
                profile,
                sample_root,
                work_root,
                f"sweep-w{w}",
                sweep_pages,
                w,
                perf_cpuset,
                reports.parent,
            )
        )
    chosen = choose_workers(sweep)
    eff = None
    if eff_cpuset:
        eff = run_bench(
            profile,
            sample_root,
            work_root,
            f"eff-w{chosen}",
            sweep_pages,
            chosen,
            eff_cpuset,
            reports.parent,
        )
    full = run_bench(
        profile, sample_root, work_root, "full", full_pages, chosen, perf_cpuset, reports.parent
    )
    result = {
        "profile": profile.name,
        "render_s_per_page": speeds,
        "formula_workers": centre,
        "sweep": [asdict(r) for r in sweep],
        "chosen_workers": chosen,
        "eff": asdict(eff) if eff else None,
        "full": asdict(full),
        "hourly_cost": hourly_cost,
        "cost_per_1000_pages": cost_per_1000(full.pages_per_s, hourly_cost)
        if hourly_cost
        else None,
        "hours_per_1m_pages": round(1e6 / full.pages_per_s / 3600, 1) if full.pages_per_s else None,
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"bench-{profile.name}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (reports / f"bench-{profile.name}.md").write_text(report_markdown(result), encoding="utf-8")
    return result


def report_markdown(r: dict[str, Any]) -> str:
    f = r["full"]
    lines = [
        f"# Benchmark — {r['profile']}",
        "",
        "## Render seconds per page (one pinned core)",
        "",
        "| mode@cores | mean | median | p95 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, v in r["render_s_per_page"].items():
        lines.append(f"| {k} | {v['mean_s']} | {v['median_s']} | {v['p95_s']} |")
    lines += [
        "",
        f"## Render worker sweep (formula start: {r['formula_workers']})",
        "",
        "| workers | pages/s | engine share | out tok/page | GPU SM % |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in r["sweep"]:
        lines.append(
            f"| {s['workers']} | {s['pages_per_s']} | {s['bound_by'].get('engine', 0):.2f} | "
            f"{s['out_tokens_per_page']} | {s['gpu_util_pct']} |"
        )
    lines.append(f"\nChosen: **{r['chosen_workers']}** (smallest with engine share >= 0.95).")
    if r["eff"]:
        e = r["eff"]
        lines += [
            "",
            f"Render on efficiency cores `{e['cpuset']}` with {e['workers']} workers: "
            f"{e['pages_per_s']} pages/s, engine share {e['bound_by'].get('engine', 0):.2f}.",
        ]
    lines += [
        "",
        f"## Full run ({f['pages']} pages, workers {f['workers']}, cores `{f['cpuset']}`)",
        "",
        f"- pages/s (excl. 2 min warm-up): **{f['pages_per_s']}**",
        f"- bound_by: {f['bound_by']}",
        f"- output tokens/page: {f['out_tokens_per_page']}",
        f"- GPU SM utilisation: {f['gpu_util_pct']}%",
        f"- wall time: {f['elapsed_s']} s; errors {f['errors']}",
        f"- 1M pages: {r['hours_per_1m_pages']} h",
        f"- cost per 1,000 pages: {r['cost_per_1000_pages'] if r['cost_per_1000_pages'] is not None else 'n/a (no --hourly-cost given)'}",
    ]
    return "\n".join(lines) + "\n"


def write_measured(profile_yaml: Path, values: dict[str, float | int | str], workers: int) -> None:
    """Replace the ``measured:`` line and ``render.workers`` in the profile YAML, keeping comments."""
    text = profile_yaml.read_text(encoding="utf-8")
    inline = ", ".join(f"{k}: {v}" for k, v in values.items())
    text = re.sub(r"^measured:.*$", f"measured: {{{inline}}}", text, flags=re.MULTILINE)
    text = re.sub(
        r"^(\s+workers:\s*)\d+(\s*)$", rf"\g<1>{workers}\2", text, count=1, flags=re.MULTILINE
    )
    profile_yaml.write_text(text, encoding="utf-8")
