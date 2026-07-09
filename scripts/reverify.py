#!/usr/bin/env python3
"""Re-verify records whose evaluate_kernel step failed due to a broken environment.

Motivation: if a sweep is launched with an interpreter that can't `import torch`
(e.g. system python instead of .venv/bin/python), every kernel that compiles +
runs successfully will get bucketed as `provider_error` at the verify step,
even though the LLM output and compile are fine.

This script salvages those records: it extracts the emitted kernel from the
saved transcript, re-runs evaluate_kernel, and updates the record's correct /
fail_stage / speedup fields. Only records whose current fail_stage is
`provider_error` AND whose transcript yields an extractable kernel are touched.
Everything else (compile-fail, no_code, already-correct) is left alone.

Usage:
    .venv/bin/python scripts/reverify.py <run_tag>

    e.g.  .venv/bin/python scripts/reverify.py groq_qwen-qwen3-32b_one_shot_mt5200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkb import problems as P
from mkb.llm.generate import extract_metal
from scripts.run_suite import evaluate_kernel

RESULTS_RAW = Path(__file__).resolve().parents[1] / "results" / "raw"


def reverify_one(record_path: Path) -> tuple[str, str]:
    """Returns (before_stage, after_stage) or (before, 'skipped:reason')."""
    rec = json.loads(record_path.read_text())
    before = rec.get("fail_stage") or ("pass" if rec.get("correct") else "?")

    if rec.get("correct"):
        return before, "skipped:already-correct"
    if rec.get("fail_stage") != "provider_error":
        return before, f"skipped:not-provider-error ({rec.get('fail_stage')})"

    # Extract the emitted kernel from the last assistant message
    assistant_msgs = [m for m in rec["transcript"] if m.get("role") == "assistant"]
    if not assistant_msgs:
        return before, "skipped:no-assistant-message"
    kernel = extract_metal(assistant_msgs[-1]["content"])
    if kernel is None:
        return before, "skipped:no-extractable-kernel"

    # Load the problem spec + reference so we can verify
    pid = rec["problem"]
    spec_path = next((p for p in P.discover() if p.parent.name == pid), None)
    if spec_path is None:
        return before, "skipped:spec-not-found"
    problem, reference = P.load(spec_path)

    ok, feedback, metrics = evaluate_kernel(kernel, problem, reference)
    rec["correct"] = ok
    rec["fail_stage"] = None if ok else metrics.get("stage")
    rec["speedup"] = metrics.get("speedup")
    rec["reverified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    record_path.write_text(json.dumps(rec, indent=2))
    return before, ("pass" if ok else metrics.get("stage", "?"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_tag", help="e.g. groq_qwen-qwen3-32b_one_shot_mt5200")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing anything")
    args = ap.parse_args()

    files = sorted(RESULTS_RAW.glob(f"{args.run_tag}__p*.json"))
    if not files:
        print(f"no records match tag: {args.run_tag}", file=sys.stderr)
        sys.exit(1)

    print(f"{len(files)} records for {args.run_tag}")
    changed = 0
    for f in files:
        pid = f.stem.split("__", 1)[1]
        if args.dry_run:
            rec = json.loads(f.read_text())
            stage = rec.get("fail_stage") or ("pass" if rec.get("correct") else "?")
            eligible = (not rec.get("correct")) and rec.get("fail_stage") == "provider_error"
            print(f"  {pid:32} {stage:16} -> {'would-reverify' if eligible else 'skip'}")
            continue
        before, after = reverify_one(f)
        marker = "*" if before != after and not after.startswith("skipped") else " "
        print(f" {marker} {pid:32} {before:16} -> {after}")
        if before != after and not after.startswith("skipped"):
            changed += 1
    if not args.dry_run:
        print(f"\n{changed} records changed")


if __name__ == "__main__":
    main()
