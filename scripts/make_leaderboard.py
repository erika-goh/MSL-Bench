#!/usr/bin/env python3
"""Aggregate results/raw/*.json into a markdown leaderboard (Phase 3/4)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mkb.score import fast_p, tier_table

RAW = Path(__file__).resolve().parents[1] / "results" / "raw"
TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"

STAGE_GLYPH = {
    None: "✓",       # correct
    "compile": "c",
    "runtime": "r",
    "verify": "v",
    "no_code": "n",
    "provider_error": "e",
}
NOT_RUN = "·"


def main() -> None:
    runs: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(RAW.glob("*.json")):
        rec = json.loads(f.read_text())
        runs[rec["run"]].append(rec)

    if not runs:
        print("no results in results/raw/ yet")
        return

    lines = ["# Metal KernelBench — Leaderboard", "",
             "| Run | n | fast_0 (correct) | fast_1 (≥MPS) | fast_2 (≥2×MPS) |",
             "|---|---|---|---|---|"]
    for run, recs in sorted(runs.items()):
        lines.append(f"| {run} | {len(recs)} | {fast_p(recs, 0):.1%} "
                     f"| {fast_p(recs, 1):.1%} | {fast_p(recs, 2):.1%} |")

    lines += ["", "## Per-tier breakdown", ""]
    for run, recs in sorted(runs.items()):
        lines.append(f"### {run}")
        lines.append("| Tier | n | fast_0 | fast_1 | fast_2 |")
        lines.append("|---|---|---|---|---|")
        for tier, row in tier_table(recs).items():
            lines.append(f"| {tier} | {row['n']} | {row['fast_0']:.1%} "
                         f"| {row['fast_1']:.1%} | {row['fast_2']:.1%} |")
        lines.append("")

    lines += _problem_status_section(runs)

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "leaderboard.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def _problem_status_section(runs: dict[str, list[dict]]) -> list[str]:
    """One row per problem, one column per run. Cell = ✓ (correct) or fail-stage glyph."""
    run_names = sorted(runs)
    # index: (pid, tier) -> {run_name: record}
    by_problem: dict[tuple[str, int], dict[str, dict]] = {}
    for run_name, recs in runs.items():
        for r in recs:
            by_problem.setdefault((r["problem"], r["tier"]), {})[run_name] = r

    lines = ["## Per-problem failure stage",
             "",
             "Legend: `✓` correct · `c` compile · `r` runtime · `v` verify · "
             "`n` no code emitted · `e` provider/harness error · `·` not run.",
             ""]

    header = ["Problem", "T"] + run_names + ["✓/n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for (pid, tier) in sorted(by_problem, key=lambda k: (k[1], k[0])):
        row_recs = by_problem[(pid, tier)]
        cells: list[str] = []
        n_pass = 0
        n_ran = 0
        for run_name in run_names:
            r = row_recs.get(run_name)
            if r is None:
                cells.append(NOT_RUN)
                continue
            n_ran += 1
            if r["correct"]:
                n_pass += 1
                sp = r.get("speedup")
                cells.append(f"✓ {sp:.1f}×" if sp else "✓")
            else:
                cells.append(STAGE_GLYPH.get(r["fail_stage"], "?"))
        lines.append(f"| {pid} | {tier} | " + " | ".join(cells) + f" | {n_pass}/{n_ran} |")
    lines.append("")

    # Callout: problems no run has passed yet.
    unbeaten = [(pid, tier) for (pid, tier), rs in by_problem.items()
                if not any(r["correct"] for r in rs.values())]
    if unbeaten and len(runs) >= 2:
        lines += ["### Unbeaten problems (0 correct across all runs)",
                  "",
                  ", ".join(pid for pid, _ in sorted(unbeaten, key=lambda k: (k[1], k[0]))),
                  ""]
    return lines


if __name__ == "__main__":
    main()
