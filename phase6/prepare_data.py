#!/usr/bin/env python3
"""Prepare MLX-LM LoRA training data from the exported SFT seeds.

v0 scope: train on sft_write.jsonl only — single-turn (system, problem) ->
correct kernel, so every assistant target is a WORKING kernel. The multi-turn
sft_repair.jsonl trajectories contain wrong intermediate kernels; MLX-LM trains
loss on every assistant turn, so feeding those in would teach the model to
emit broken code. Repair-trajectory training needs completion-format loss
masking (target = final turn only) — deferred to a later iteration.

Outputs phase6/data/{train,valid}.jsonl in MLX-LM chat format ({"messages":..}).

Run:
    .venv/bin/python phase6/prepare_data.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "results" / "sft" / "sft_write.jsonl"
OUT = ROOT / "phase6" / "data"


def main() -> None:
    if not SEEDS.exists():
        raise SystemExit(f"{SEEDS} missing — run scripts/export_sft.py first")

    rows = [json.loads(line) for line in SEEDS.read_text().splitlines() if line.strip()]
    # MLX-LM wants bare {"messages": [...]}; drop our metadata sidecar.
    examples = [{"messages": r["messages"]} for r in rows]

    random.seed(0)  # deterministic split so re-runs are reproducible
    random.shuffle(examples)

    # 90/10 train/valid. With ~38 examples this valid set is tiny (~4) — it
    # exists to satisfy MLX-LM and give a loss to eyeball, not to be a rigorous
    # held-out metric (that needs held-out PROBLEMS; see the phase-6 plan).
    n_valid = max(2, round(len(examples) * 0.1))
    valid, train = examples[:n_valid], examples[n_valid:]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in train))
    (OUT / "valid.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in valid))

    print(f"prepared MLX-LM data in {OUT.relative_to(ROOT)}/")
    print(f"  train.jsonl  {len(train)}")
    print(f"  valid.jsonl  {len(valid)}")


if __name__ == "__main__":
    main()
