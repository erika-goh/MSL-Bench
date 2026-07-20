# Phase 2.5 — Held-out test split: plan

*Status: DONE (2026-07-20). Mechanism A wired; 15/15 held-out problems authored
and golden-verified (T1×3, T2×4, T3×4, T4×4). Written after the Phase-6 v0 eval
showed N=34 SFT gives no lift on T2 — and reminded us that any future "accuracy
went up" is meaningless without problems the model was never trained on.*

## Why

Phase-6's honest-eval problem: the 60 seed problems are both the SFT source and
the benchmark, so a naive accuracy gain partly measures memorization. A held-out
split — problems reserved as a **test set, never exported into SFT** — is the
prerequisite for any clean generalization claim. This is the #1 carry-forward
from the Phase-6 eval.

## What it must test: idiom *transfer*, not new capability

The held-out problems should exercise the **same MSL idioms** as each tier (so
they measure whether SFT taught transferable patterns) while being **different
ops** than the 60 seeds (so nothing is memorized):

- T1 — one-thread-per-element, buffer binding
- T2 — threadgroup/SIMD-group reduction, `threadgroup_barrier(mem_flags::…)`
- T3 — threadgroup-staged tiles, cooperative load
- T4 — fused reduce + elementwise

## Proposed roster (~15, weighted to the tiers that carry the idiom story)

Reserved id ranges keep them visually distinct from seeds (seed T2 = p101–p115;
held-out T2 = p151+). All new ops, none duplicate a seed:

| tier | ids | ops (candidates) |
|---|---|---|
| T1 (3) | p051–p053 | softplus, elu, hardswish |
| T2 (4) | p151–p154 | row_l1_norm, row_prod, row_argmin, row_range (max−min) |
| T3 (4) | p251–p254 | matmul Aᵀ@B, avg_pool2d 2×2, tiled outer-product, im2col |
| T4 (4) | p351–p354 | groupnorm, fused linear+sigmoid, layernorm (no affine), fused bias+tanh |

(Exact op list is up for edit — this is the review point.)

## The one design decision: how to mark a problem "held-out"

`mkb/problems.py` auto-discovers every `tier*/p*/spec.py`. Held-out problems
must be **excluded from normal runs by default** or they'd silently pollute the
60-problem suite (and could leak into SFT). Two ways:

### Option A — `split` field + `--split` filter (recommended)
- Add `"split": "heldout"` to the PROBLEM dict (default `"train"` when absent).
- Held-out specs live in the **same** `problems/tierN/` tree — loader,
  executor, golden-kernel handling all unchanged.
- `run_suite.py` gains `--split {train,heldout,all}`, default `train`.
- `export_sft.py` also filters out `heldout` defensively, so even if we run
  them, they can't reach training data.
- ✅ tiny change (one field + one filter), zero risk to existing code paths.
- ❌ held-out specs sit next to seeds on disk (mitigated by reserved id range).

### Option B — separate `problems_heldout/` tree
- New top-level dir; `discover()` scans only `problems/` so held-out is isolated
  on disk and cannot be discovered by the normal path at all.
- ✅ strongest physical isolation.
- ❌ touches `discover()` + anything that globs problems; two code paths to keep
  in sync; more moving parts for the same guarantee Option A gives via a filter.

**Recommendation: A.** The leak that matters (training on test problems) happens
at the *SFT-export* layer, not discovery — and a `split` field lets us block it
there directly, which Option B's dir-isolation doesn't even address. A is less
code for the guarantee that counts.

## Build order (once mechanism + roster are approved)

1. Wire Option A: `split` field default + `run_suite --split` + `export_sft`
   filter. (~15 lines, no GPU.)
2. Author specs + golden kernels, one tier at a time, smallest first (T1→T4).
3. Verify each golden compiles + passes — a **single-kernel run per problem**
   (`run_problem.py`), which is sub-second GPU work (thermally fine per the
   safety rule; not a sweep).
4. Only then is the split usable as a test set for the next retrain.
