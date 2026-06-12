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

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "leaderboard.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
