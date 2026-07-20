# Phase 6 — Data Flywheel: plan

*Status: plan, not executed. Written 2026-07-20 after the repair@5 seed set
landed. This is the bridge doc; the actual work lives in a separate repo
(decision below).*

## Goal

Close the loop the whole project is named for:

    repair transcripts  ->  SFT a base model  ->  re-benchmark  ->  measure lift

We have the first arrow (Phase 3 produced 60/60 repair transcripts;
`scripts/export_sft.py` turns them into training JSONL). Phase 6 is the rest.

## What we have as training signal

From `scripts/export_sft.py` (regenerate from `results/raw/`):

    sft_write.jsonl      38  (system, problem) -> final correct kernel
    sft_repair.jsonl     15  full converged trajectory (wrong + error -> fixed)
    dpo_negatives.jsonl  23  never-converged trajectories (labeled, NOT SFT)

**Read this number honestly: 38 examples is tiny.** This is a
proof-of-concept, not a production fine-tune. Expect modest and noisy
results. A real flywheel needs an order of magnitude more seeds — more
models sweeping repair, iterated over more problems. The point of the v0 is
to stand the loop up end-to-end and see the *shape* of the lift, not to ship
a great model.

## The honest evaluation problem (decide this before training)

The seeds came from the same 60 problems we'd re-benchmark on. So a naive
"accuracy went up" would partly measure **memorization**, not skill. Two
things make it less circular, but neither fully fixes it:

- The seed kernels are *other models'* solutions (mostly gpt-oss), so the
  fine-tuned base isn't just memorizing its own outputs.
- The real question is **idiom uptake**: does SFT teach the MSL-specific
  patterns the T1->T2 cliff is about (`threadgroup` memory, barrier
  `mem_flags`, SIMD-group intrinsics)? That transfers across problems even
  within the same suite.

**Recommended framing for the write-up:** don't headline raw accuracy.
Report (a) the failure-mode shift (compile -> verify -> pass, the §5
mechanism) and (b) tier-level gains on T2/T3, as evidence of idiom uptake.
Treat clean generalization as out of scope until the suite has a held-out
split (a Phase-2.5 task: author ~15 more problems reserved as a test set).

## Tooling — mostly already here

- **Re-benchmark needs no new harness code.** `run_suite.py` already has an
  `ollama` provider. Serve the fine-tuned model in Ollama, then:

      run_suite.py --provider ollama --model msl-coder-ft --mode one_shot
      run_suite.py --provider ollama --model msl-coder-ft --mode repair --k 5

  and compare against the base model's own one_shot run as the control.
- **Training** is the only new piece. On this machine (32 GB, Apple silicon,
  no MLX yet) the natural path is **MLX-LM LoRA** — Apple-silicon native,
  LoRA on a 7B fits comfortably in 32 GB. Needs `pip install mlx-lm`.

## Decisions for you (two reasonable options each)

### 1. Base model
- **Qwen2.5-Coder-7B (recommended).** Strong code base, LoRA fits 32 GB in
  MLX, fast iteration. Already know it as a family (`qwen2.5-coder:14b` is in
  Ollama). Big enough to *have* the capability SFT can surface, small enough
  to train locally tonight-scale.
- Qwen2.5-Coder-14B — already pulled in Ollama, stronger, but LoRA training
  in 32 GB is tight (needs 4-bit + small batch) and slower.
- Qwen2.5-Coder-1.5B — trains in minutes, but §5's lesson says a weak model
  can't act on repair signal; likely too weak to show uptake.

### 2. Training method
- **LoRA via MLX-LM (recommended)** — cheap, local, reversible, matches the
  learning-project ethos (you see every step). Merge adapters, convert to
  GGUF, serve via Ollama.
- Hosted fine-tune API (e.g. Together/Fireworks) — less local fiddling, but
  costs money, sends the data out, and hides the mechanics.

### 3. Repo location
- **Separate repo `msl-flywheel` (recommended, matches the roadmap).** Layout:
  `data/` (exported JSONL, gitignored like here), `train/` (MLX LoRA config +
  scripts), `eval/` (thin wrapper calling this repo's `run_suite`), `README`.
- Subdirectory `phase6/` in this repo — simpler, but couples training deps
  (mlx) into the benchmark repo.

## Proposed first run (once decisions are made)

1. `pip install mlx-lm`; convert Qwen2.5-Coder-7B to MLX format.
2. LoRA SFT on `sft_write.jsonl` + `sft_repair.jsonl` (few hundred steps;
   watch for overfit on 53 examples — low rank, early stop).
3. Merge, convert to GGUF, `ollama create msl-coder-ft -f Modelfile`.
4. `run_suite --provider ollama --model msl-coder-ft --mode one_shot` (full 60).
5. Compare fast_0-by-tier vs base `qwen2.5-coder` one_shot; inspect T2/T3
   idiom uptake and the compile->verify failure-mode shift.
6. If there's *any* signal, that validates the loop — then scale the seed set.

## Risks / watch-items

- **Overfit on ~53 examples** — use low LoRA rank, few steps, early stop.
- **GGUF conversion of a LoRA-merged MLX model** can be fiddly; budget time.
- **Re-benchmark thermals** — a full 60-problem local sweep on the Mac is a
  sustained GPU load; get the usual thermal-safety greenlight first.
- **Circularity** — keep saying it out loud; don't let a memorization bump
  read as generalization.
