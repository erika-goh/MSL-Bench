#!/usr/bin/env python3
"""Regenerate demo/leaderboard.html from results/raw/*.json.

The demo HTML is a self-contained single file. The data (runs, per-problem
grid, sample kernels) is injected into a single JS block between two markers:

    // >>>GEN_DATA_START
    const DATA = {...};
    // <<<GEN_DATA_END

Everything else in the file — CSS, layout, chart rendering, drawer — is
edited by hand and left alone. Regeneration only rewrites the block above.

Run:
    .venv/bin/python scripts/make_demo.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkb import problems as P
from mkb.llm.generate import build_prompt, extract_metal
from mkb.score import fast_p

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
DEMO = ROOT / "demo" / "leaderboard.html"

MARKER_START = "// >>>GEN_DATA_START"
MARKER_END = "// <<<GEN_DATA_END"


def _run_label(run_id: str) -> str:
    """Turn 'groq_llama-3.3-70b-versatile_one_shot' into 'groq · llama-3.3-70b · one_shot'.

    Run tag layout (see scripts/run_suite.py):
        {provider}_{model_with_/-and-:-replaced}_{mode}[_mt{N}]
    Mode is 'one_shot' or 'repair' (one_shot contains an underscore, so we
    can't just split on _). We peel known suffixes off the end instead.
    """
    rest = run_id
    # Optional _mt{N} suffix
    mt_suffix = ""
    m = re.search(r"_mt(\d+)$", rest)
    if m:
        mt_suffix = f" · mt={m.group(1)}"
        rest = rest[: m.start()]
    # Mode suffix
    if rest.endswith("_one_shot"):
        mode_label = "one_shot"
        rest = rest[: -len("_one_shot")]
    elif rest.endswith("_repair"):
        mode_label = "repair k=5"
        rest = rest[: -len("_repair")]
    else:
        mode_label = "?"
    # rest is now provider_model
    provider, _, model = rest.partition("_")
    # Cosmetic simplification for common Meta / OpenAI names
    model_short = model.replace("-versatile", "").replace("-instant", "")
    return f"{provider} · {model_short} · {mode_label}{mt_suffix}"


def build_runs(runs: dict[str, list[dict]], total_problems: int) -> list[dict]:
    out = []
    for run_id, recs in runs.items():
        out.append({
            "id": run_id,
            "label": _run_label(run_id),
            "n": len(recs),
            "fast0": fast_p(recs, 0),
            "fast1": fast_p(recs, 1),
            "fast2": fast_p(recs, 2),
            "partial": len(recs) < total_problems,
        })
    # Sort by fast_1 descending so the leader ends up first; ties broken by n descending
    out.sort(key=lambda r: (-r["fast1"], -r["n"], r["id"]))
    return out


def cell_from_record(r: dict) -> dict:
    if r.get("correct"):
        return {"stage": "ok", "speedup": r.get("speedup")}
    stage = r.get("fail_stage") or "error"
    mapped = {"compile": "compile", "verify": "verify", "runtime": "runtime",
              "no_code": "none", "provider_error": "error"}.get(stage, "error")
    return {"stage": mapped, "speedup": None}


def build_problems(runs_index: list[dict], runs_recs: dict[str, list[dict]]) -> list[dict]:
    # Discover every (pid, tier) that appears in ANY record.
    by_key: dict[tuple[str, int], dict[str, dict]] = {}
    for run_id, recs in runs_recs.items():
        for r in recs:
            by_key.setdefault((r["problem"], r["tier"]), {})[run_id] = r

    out = []
    for (pid, tier) in sorted(by_key, key=lambda k: (k[1], k[0])):
        cells = []
        for run in runs_index:
            r = by_key[(pid, tier)].get(run["id"])
            cells.append(cell_from_record(r) if r else None)
        out.append({"pid": pid, "tier": tier, "cells": cells})
    return out


def build_samples(runs_recs: dict[str, list[dict]], prompts: dict[str, str]) -> dict[str, dict]:
    """One entry per (problem, run) that has a transcript. Kernel + prompt + attempts + timestamp."""
    samples: dict[str, dict] = {}
    for run_id, recs in runs_recs.items():
        for r in recs:
            pid = r["problem"]
            asst = next((m for m in r.get("transcript", []) if m.get("role") == "assistant"), None)
            kernel = extract_metal(asst["content"]) if asst else None
            if not kernel:
                # For no_code / provider_error rows we don't have a runnable
                # kernel; leave the sample out so the drawer shows its empty state.
                continue
            samples[f"{pid}|{run_id}"] = {
                "kernel": kernel,
                "prompt": prompts.get(pid, ""),
                "diag": None,  # future: recompile to capture diagnostics
                "attempts": r.get("attempts", 1),
                "timestamp": r.get("timestamp", ""),
            }
    return samples


def build_prompts_by_problem() -> dict[str, str]:
    """Build the user-prompt text for every discovered problem (used in the drawer)."""
    prompts = {}
    for spec_path in P.discover():
        problem, _ = P.load(spec_path)
        messages = build_prompt(problem)
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        prompts[problem["id"]] = user_msg
    return prompts


def collect_runs() -> tuple[dict[str, list[dict]], int]:
    runs: dict[str, list[dict]] = {}
    for f in sorted(RAW.glob("*.json")):
        rec = json.loads(f.read_text())
        runs.setdefault(rec["run"], []).append(rec)
    total_problems = len(list(P.discover()))
    return runs, total_problems


def render_data_block(data: dict) -> str:
    # json.dumps with default indent produces a 4-space indent; keep it 2 for
    # readability inside the HTML.
    body = json.dumps(data, indent=2, ensure_ascii=False)
    return f"{MARKER_START}\nconst DATA = {body};\n{MARKER_END}"


def rewrite_demo(new_block: str) -> None:
    html = DEMO.read_text()
    pattern = re.compile(
        re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
        re.DOTALL,
    )
    if not pattern.search(html):
        raise SystemExit(
            f"markers {MARKER_START} / {MARKER_END} not found in {DEMO}; "
            "cannot regenerate (was the template edited?)"
        )
    # A plain-string replacement would let re.sub interpret escape sequences like
    # `\n` in the JSON payload as literal newlines, destroying the JSON. The
    # lambda form passes the replacement through verbatim.
    DEMO.write_text(pattern.sub(lambda _: new_block, html))


def main() -> None:
    runs_recs, total_problems = collect_runs()
    if not runs_recs:
        raise SystemExit("no records in results/raw/; nothing to generate")

    prompts = build_prompts_by_problem()
    runs_index = build_runs(runs_recs, total_problems)
    problems = build_problems(runs_index, runs_recs)
    samples = build_samples(runs_recs, prompts)

    data = {
        "generated": time.strftime("%Y-%m-%d"),
        "totalProblems": total_problems,
        "runs": runs_index,
        "problems": problems,
        "samples": samples,
    }

    new_block = render_data_block(data)
    rewrite_demo(new_block)

    n_cells = sum(1 for p in problems for c in p["cells"] if c is not None)
    print(f"regenerated {DEMO.relative_to(ROOT)}:")
    print(f"  runs:     {len(runs_index)}")
    print(f"  problems: {len(problems)}")
    print(f"  cells:    {n_cells}")
    print(f"  samples:  {len(samples)}")
    print(f"  data:     {len(json.dumps(data)) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
