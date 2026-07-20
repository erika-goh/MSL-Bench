#!/usr/bin/env python3
"""Export repair@5 transcripts into SFT-ready JSONL (the flywheel's next link).

Phase 3 produced repair transcripts in results/raw/. This turns them into
training data. Two framings, because a repair flywheel wants both:

  write   — (system, problem) -> final CORRECT kernel.
            Teaches writing a working kernel. Every converged trajectory
            yields one. The wrong intermediate attempts are dropped, so no
            example ever has a broken kernel as its target.

  repair  — the full converged trajectory (system, problem, wrong kernel,
            error feedback, ..., correct kernel). Teaches the *fix* skill:
            given a broken kernel and its compiler/verify error, produce a
            corrected one. Only trajectories that actually needed a repair
            (attempts > 1) yield these. SFT tooling must train loss on the
            final assistant turn only — the earlier (wrong) assistant turns
            are prompt context, not targets.

Failed trajectories (never converged) are NOT SFT targets — training on them
teaches failure. They're written to a separate file as future DPO/preference
negatives, labeled, not mixed into the SFT sets.

Run:
    .venv/bin/python scripts/export_sft.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "sft"

# Roles we keep in a training message list. The repair loop's provider-error
# markers (role "error", added by repair_k when a provider aborts mid-loop)
# are bookkeeping, never training content.
_CHAT_ROLES = {"system", "user", "assistant"}


def _clean(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]}
            for m in messages if m.get("role") in _CHAT_ROLES]


def _first_user_idx(msgs: list[dict]) -> int:
    return next(i for i, m in enumerate(msgs) if m["role"] == "user")


def _heldout_pids() -> set[str]:
    """Problem ids marked split=heldout in their spec — the Phase-2.5 test set.

    These must NEVER become training data, or the held-out eval is meaningless.
    We filter by the spec (source of truth), not by run tag, so a mis-tagged run
    still can't leak a test problem into SFT.
    """
    from mkb import problems as P
    out = set()
    for sp in P.discover():
        prob, _ = P.load(sp)
        if prob.get("split", "train") == "heldout":
            out.add(prob["id"])
    return out


def main() -> None:
    records = [json.loads(f.read_text()) for f in sorted(RAW.glob("*_repair*__*.json"))]
    if not records:
        raise SystemExit("no repair records in results/raw/; run run_suite.py --mode repair first")

    heldout = _heldout_pids()
    dropped = [r["problem"] for r in records if r["problem"] in heldout]
    records = [r for r in records if r["problem"] not in heldout]
    if dropped:
        print(f"excluded {len(dropped)} held-out transcript(s) from SFT: "
              f"{sorted(set(dropped))}")

    OUT.mkdir(parents=True, exist_ok=True)
    write_rows, repair_rows, failed_rows = [], [], []

    for r in records:
        msgs = _clean(r["transcript"])
        meta = {"problem": r["problem"], "tier": r["tier"], "run": r["run"],
                "attempts": r["attempts"]}

        if not r["correct"]:
            # Never converged — keep as a labeled negative, not SFT. But a
            # negative is only useful if it contains an actual (wrong) kernel:
            # bare provider-abort records clean down to zero assistant turns and
            # carry no signal, so drop them.
            if any(m["role"] == "assistant" for m in msgs):
                failed_rows.append({"messages": msgs, "meta": {**meta, "fail_stage": r.get("fail_stage")}})
            continue

        # The last assistant turn is the correct kernel (the trajectory converged).
        last_assistant = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
        if last_assistant is None:
            continue  # defensive: converged record with no assistant turn shouldn't happen
        ui = _first_user_idx(msgs)
        system = [m for m in msgs[:ui] if m["role"] == "system"]

        # write: collapse to (system, problem) -> final correct kernel.
        write_rows.append({"messages": system + [msgs[ui], last_assistant], "meta": meta})

        # repair: only if a fix actually happened (more than one attempt).
        if r["attempts"] > 1:
            repair_rows.append({"messages": msgs, "meta": meta})

    def dump(rows: list[dict], name: str) -> None:
        path = OUT / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        return path

    dump(write_rows, "sft_write.jsonl")
    dump(repair_rows, "sft_repair.jsonl")
    dump(failed_rows, "dpo_negatives.jsonl")

    by_tier: dict[int, int] = {}
    for row in write_rows:
        by_tier[row["meta"]["tier"]] = by_tier.get(row["meta"]["tier"], 0) + 1
    print(f"exported to {OUT.relative_to(ROOT)}/")
    print(f"  sft_write.jsonl     {len(write_rows):3d}  (problem -> correct kernel)")
    print(f"    by tier: " + "  ".join(f"T{t}:{by_tier.get(t,0)}" for t in (1, 2, 3, 4)))
    print(f"  sft_repair.jsonl    {len(repair_rows):3d}  (wrong + error -> fixed; success-after-repair)")
    print(f"  dpo_negatives.jsonl {len(failed_rows):3d}  (never converged; labeled, not SFT)")


if __name__ == "__main__":
    main()
