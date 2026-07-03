# MSL-Bench Phase 4 Report

*How well do LLMs write Metal compute kernels — and where do they break?*

Draft dated 2026-07-03. Numbers reproducible from `results/raw/` at commit
`7a93b0c`. Written to be readable end-to-end without deep GPU background;
concepts introduced inline where first used.

---

## 1. What this benchmark measures

MSL-Bench is the Metal-side counterpart to KernelBench's CUDA suite. It
asks: given a well-specified numerical operation (add two vectors, compute
softmax, do a masked attention), can a frontier LLM write a Metal Shading
Language (MSL) compute kernel that (a) compiles under Apple's real
`xcrun metal` toolchain, (b) produces correct output verified against a
PyTorch MPS reference, and (c) runs at least as fast as MPS on the same
Apple Silicon GPU?

The suite is 60 problems across 4 difficulty tiers:

| Tier | Class            | Count |
|------|------------------|-------|
| T1   | Elementwise      | 15    |
| T2   | Reductions       | 15    |
| T3   | Tiled (matmul, conv, transpose) | 15 |
| T4   | Fused (layernorm, attention, GLU) | 15 |

Each candidate kernel is judged on `fast_p` — the fraction of problems
that are (correct) AND (kernel_ms ≤ reference_ms / p). So `fast_0` is
"just correct," `fast_1` is "matches MPS speed or better," `fast_2` is
"twice MPS speed."

## 2. The two models tested

Two providers, one shot each, both free tier:

- **Groq** — `llama-3.3-70b-versatile` — full 36-problem one_shot sweep.
- **Google** — `gemini-2.5-flash` — 19 of 36 before the free tier quota killed the sweep.

(Repair@5 partial data also collected for both; see §5.)

Headline `fast_0` leaderboard, sorted by sample size (full-suite runs
first, partial samples tagged):

    groq llama-3.3-70b one_shot      n=36     38.9%
    gemini 2.5-flash   one_shot      n=19†    63.2%
    ollama qwen2.5-coder-14b one_shot n=6†     0.0%

† partial samples — not directly comparable to full-suite rows.

## 3. The T1 → T2 cliff (headline finding)

Under one_shot, both models handle elementwise operations well and cliff-
dive at reductions:

| Tier | Groq fast_0 | Gemini fast_0 |
|------|-------------|---------------|
| T1 elementwise | **100.0%** (12/12) | 91.7% (11/12) |
| T2 reductions  | 22.2% (2/9)  | 14.3% (1/7)  |
| T3 tiled       | 0.0%         | *n/a (quota)* |
| T4 fused       | 0.0%         | *n/a (quota)* |

The two models fail T2 in **complementary** ways:

- **Gemini** reaches for the "right" pattern (threadgroup memory, atomics)
  but writes MSL syntax wrong at the *declaration* level — most T2
  failures are compile errors.
- **Groq** writes safer code that compiles and produces correct output,
  but at speeds well below MPS.

Both patterns are diagnostic of what LLMs know and don't know about
Apple's compute programming model. The rest of this report unpacks
what "well below MPS" actually means, and how much of it is model
quality vs. benchmark design.

## 4. The three-layer decomposition of the LLM-vs-MPS gap

Hand-patching Groq's `p101_row_sum` kernel four ways revealed a clear
layered structure to where the ~2.5× gap to MPS actually lives. The
problem is memory-bound: input is 268 MB, output negligible. Effective
memory bandwidth achieved by each version:

    Groq original (mem_device barrier flag):    74.8 GB/s
    Groq patched  (mem_threadgroup barrier):    80.9 GB/s   (+7%)
    Hand-tuned SIMD-group reduce:                88.7 GB/s   (+8%)
    Hand-tuned float4 loads under spec launch:   90.5 GB/s   (~0%)
    MPS reference:                              ~175  GB/s   (2x)

Three layers of cost, measured:

**Layer 1 — Barrier-flag confusion (~7%).** Groq wrote
`threadgroup_barrier(mem_flags::mem_device)` — the wrong memory scope
for a reduction that only touches threadgroup memory. Apple's compiler
partially elides the redundant device-scope fence but not entirely.
This is a CUDA→MSL syntax-transfer failure: `__syncthreads()` in CUDA
has no flag parameter, so the model doesn't know the flag is load-bearing.

**Layer 2 — Missing MSL idioms (~8%).** Groq's algorithm was
correct-but-suboptimal (halving-tree over threadgroup memory instead of
`simd_shuffle_down` register-level reductions). Hand-writing the SIMD-
group version — 5 register-only shuffle steps and one threadgroup
barrier, replacing 8 threadgroup barriers + 8 shared-memory rounds —
saved another ~8%.

**Layer 3 — Bandwidth utilization (~50%).** Even the hand-tuned kernel
peaks at 90.5 GB/s vs MPS's 175 GB/s. A float4-load variant made no
difference — Apple's memory subsystem already coalesces adjacent scalar
loads into cache-line transactions, so transaction count is not the knob.

Layer 3 breaks into two sub-layers:

- **3a. Model-recoverable (~15% of the total gap).** Small extra wins
  under the spec's constraints — micro-optimizations the model isn't
  reaching for.
- **3b. Baked into the spec's launch config (~35% of the total gap).**
  The problem spec forces one threadgroup per row with 256 threads per
  group. Under that constraint, ~90 GB/s appears to be the ceiling. MPS
  is not bound by that spec — it picks its own launch — and uses the
  freedom to run at ~2× the throughput.

## 5. Repair@5: fixes syntax, not semantics

Phase 3's second headline test: does giving the LLM its own compile
diagnostics as feedback let it unstick past errors? Ran `repair@5`
(up to 5 attempts per problem, feeding the failure back) on both
providers. Both quota-died mid-sweep (Groq at 17/36, Gemini at 2/36),
but the salvaged data is unambiguous.

On the 17 problems where Groq one_shot and repair@5 both ran,
**repair@5 added zero new correct kernels vs one_shot**. Both modes
scored 14/17. What repair changed was the **failure mode** on two T2
problems:

    p102_row_max      one_shot: compile-fail  →  repair@5: verify-fail
    p104_row_softmax  one_shot: compile-fail  →  repair@5: verify-fail

The compiler diagnostics got the model past MSL syntax errors, but
5 attempts of "wrong answer, max_abs_err=X" feedback couldn't drive
it from "wrong numbers" to "right numbers."

Mapped onto the layered analysis of §4, this is what we'd predict:

- **Layer 1 errors** (compile fails from wrong syntax): the compiler
  *tells* the model what's wrong. Repair fixes these.
- **Layer 2 errors** (verify fails from wrong idioms): the compiler is
  silent, code runs, semantics are wrong. Scalar-error feedback isn't
  pointed enough to fix the specific line.
- **Layer 3 errors** (slow but correct): invisible to compile and verify
  signals both. Would require performance feedback.

**Concrete implication for Phase 5 / Phase 6:** the ceiling of
repair@5 with only compile + scalar-correctness signal is "unstick
layer-1 errors." Anything past that needs a richer feedback surface —
per-element diff previews, shape-mismatch hints, or performance-vs-
reference signals.

## 6. Benchmark-design caveat: the spec-launch cap

The layered analysis above uncovered a real design issue in the
benchmark itself. Problem specs currently include a fixed `launch`
config that all candidate kernels must obey. MPS is not bound by that
config — it picks its own launch shape for the reference computation.
On memory-bound problems this creates an unfair comparison:

- Candidate kernel: obligated to run under the spec's launch (e.g.,
  "one threadgroup per row, 256 threads per group"), which may forbid
  MPS-shaped designs like multi-row groups or wider threadgroups.
- MPS reference: free to pick a launch shape optimized for the problem.

For `p101_row_sum` this cap alone accounts for ~35% of the gap. Under
the current benchmark rules, no candidate kernel — however well
designed — can close it. The metric wording is "speedup vs MPS," but
what's actually being measured on affected problems is closer to
"speedup vs MPS on a fair-launch shape MPS would never choose."

Three plausible fixes for future versions:

1. **Let candidates propose launch configs.** Adds surface area but
   makes the "which launch" decision part of what's being tested.
2. **Time against a same-launch reference kernel.** Trade the MPS
   ceiling for an internal one; measures LLM-vs-optimal-under-constraint
   rather than LLM-vs-unconstrained-MPS.
3. **Reword the metric.** Keep the fixed launch but describe the
   number honestly as "% of achievable-under-constraint speed."

This is a benchmark-design question, not a model-quality question,
and should be a deliberate decision before Phase 5's public demo.

## 7. Failure taxonomy (§4 summarized as a table)

| Failure class | Example | Fixable by repair@5? | Fixable by better design under spec? |
|---------------|---------|----------------------|--------------------------------------|
| Compile error (layer 1) | `mem_device` flag / wrong keyword | **Yes** — compiler tells you what's wrong | n/a |
| Verify error (layer 2) | Wrong reduction pattern, off-by-one tile bound | No — scalar error not pointed | Yes, given right feedback |
| Slow correct (layer 3a) | Missing SIMD-group reductions, no fusion | No — no perf feedback in loop | Yes |
| Slow correct (layer 3b) | Spec's launch forbids MPS-shape design | No | **No** — benchmark artifact |

## 8. What's next

- **Phase 5** (web demo) and **Phase 6** (data flywheel in a separate
  repo) — the natural next work. Both explicitly future.
- **Broaden the model roster.** Only two providers ran full one_shot;
  add Claude, o1, GPT-4, Gemini Pro when budget permits. Ollama sweep
  pending (needs thermal-safety greenlight for the local machine).
- **Resolve the spec-launch cap.** The design decision from §6 should
  land before public Phase-5 numbers are published, since it changes
  what the fast_1/fast_2 metrics mean.
- **Richer feedback for repair.** Per-element diff previews, tolerance-
  bound hints, and a "your kernel is 3x slower than MPS" signal would
  test whether layer-2 and layer-3a are also learnable via feedback.

## 9. Reproducibility

    make runner                                   # build Swift harness
    python scripts/run_suite.py --provider groq \
        --model llama-3.3-70b-versatile --mode one_shot
    python scripts/make_leaderboard.py            # -> results/tables/leaderboard.md

Free tiers (GROQ_API_KEY, GEMINI_API_KEY) reproduce the numbers in
this report. Fully offline: swap `--provider ollama` after
`ollama pull qwen2.5-coder:14b`.

All raw per-problem records, including full LLM transcripts, are in
`results/raw/*.json`. The transcripts are also the intended Phase-6
seed data: they preserve the compile-diagnostic feedback loop that
repair@5 walks, one attempt at a time.

---

*NOTES.md has session-by-session detail on how these findings were
uncovered — including three self-corrections where measurements
refined earlier claims.*
