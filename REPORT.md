# MSL-Bench Phase 4 Report

*How well do LLMs write Metal compute kernels — and where do they break?*

Draft dated 2026-07-03; §5 revised 2026-07-20 after a full repair@5 sweep
refuted its original "repair adds nothing" claim (that finding was an
artifact of testing only a weak model). Numbers reproducible from
`results/raw/`. Written to be readable end-to-end without deep GPU
background; concepts introduced inline where first used.

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

## 2. Four models tested

Four providers, one shot each, all free tier:

- **Groq / llama-3.3-70b-versatile** — 36-problem one_shot sweep (before the suite expanded to 60).
- **Groq / llama-3.1-8b-instant** — **first full n=60 sweep**.
- **Groq / qwen/qwen3-32b** — reasoning model, 36/60 before paused for thermal cool-down.
- **Google / gemini-2.5-flash** — 19/36 before free tier quota killed the sweep.

(Repair@5 transcripts were later collected across all 60 problems — a
mixed-model patchwork, gpt-oss-120b on the hard tiers — as the Phase-6
flywheel seed set; the one_shot-vs-repair comparison is in §5.)

Headline `fast_0` leaderboard, sorted by sample size:

    groq openai/gpt-oss-120b   one_shot   n=60     45.0%    <- full, leader
    groq llama-3.1-8b-instant  one_shot   n=60     28.3%    <- full
    groq qwen3-32b (reasoning) one_shot   n=60     8.3%     <- full
    groq llama-3.3-70b         one_shot   n=43†    44.2%
    gemini 2.5-flash           one_shot   n=22†    63.6%
    ollama qwen2.5-coder-14b   one_shot   n=6†     0.0%

† partial samples — not directly comparable, and different runs may
have tested different problem subsets.

## 3. The T1 → T2 cliff (headline finding)

All non-reasoning models handle elementwise operations well and
cliff-dive at reductions:

| Tier | gpt-oss-120b (n=60) | llama-3.3-70b (n=43†) | llama-3.1-8b (n=60) | gemini 2.5-flash (n=22†) | qwen3-32b (n=60, reasoning) |
|------|--------------|---------------|--------------|-------------------|-----------------------|
| T1 elementwise | 93.3% (14/15) | **100.0%** (15/15) | **100.0%** (15/15) | 86.7% (13/15) | 20.0% (3/15) |
| T2 reductions  | **46.7%** (7/15) | 30.8% (4/13) | 6.7% (1/15) | 14.3% (1/7) | 0.0% (0/15) |
| T3 tiled       | **20.0%** (3/15) | 0.0% (0/8) | 0.0% (0/15) | n/a | 6.7% (1/15) |
| T4 fused       | **20.0%** (3/15) | 0.0% (0/7) | 6.7% (1/15) | n/a | 6.7% (1/15) |

Three findings pop out of this cross-model view:

**Metal elementwise fits in an 8B model.** Both 8B and 70B llama variants
hit 100% on T1. Whatever the model needs to know about MSL to write
`out[i] = f(x[i])` correctly is compact enough that the small model has
it too. Bigger models retain an edge at T2+ (llama-3.3-70b's 30.8% vs
llama-3.1-8b's 6.7%), which is where MSL-specific patterns —
threadgroup memory, barriers, atomics — start to matter.

**gpt-oss-120b is qualitatively ahead on hard tiers.** It's the only
model in the set that gets meaningful T3 (3/15 = 20%) and T4 (3/15 =
20%) numbers. Every other model tested is ≤ 1/15 on those tiers.
It's also the only model with a meaningful fast_2 rate (10.0% —
6× the next best). Whatever OpenAI's open-source lineage carries into
gpt-oss-120b, it's transferring to Metal in a way the Meta and
Alibaba lineages aren't. Notable: it costs one T1 problem (14/15 =
93.3%, not 100%), so it's slightly imperfect on the easiest tier
even while dominating the hardest ones — different quality profile
from the llama family, not strictly-better.

The T1→T2 cliff still exists for gpt-oss-120b (93% → 47%, a 46-point
drop) but is *shallower* than for the smaller models (llama-3.1-8b:
100% → 7%, a 93-point drop). So the cliff is a universal cross-
provider phenomenon (Meta, Alibaba, OpenAI, Google all show it) but
its height correlates with model quality — better models fall less
far.

**Reasoning-optimized has an inverted tier profile.** qwen3-32b
is the ONLY model in the set that fails T1 (3/15 = 20%). Its
dominant T1 failure mode is using `thread_position_in_grid` as a
free variable (OpenCL/CUDA style) instead of a Metal parameter
attribute. Reading its `<think>` blocks: it derives thread-index
handling from adjacent-language knowledge and applies the wrong
pattern confidently. Non-reasoning models that pattern-match Metal
directly (from training data) get the attribute syntax right
without "reasoning" about it.

But: qwen3-32b is also the ONLY model that got any T3 correct
(1/15). llama-3.1-8b and llama-3.3-70b are both 0/15 and 0/8 on
T3. And qwen3 ties llama-3.1-8b on T4 (1/15 each). So reasoning
appears to be **worst-worse on T1 and best-better on T3/T4**,
even while being dramatically worse overall. Small sample sizes
in the "wins" (n=1 each) mean this could be noise, but the
direction is worth noting: reasoning helps where the algorithmic
structure matters (tiled matmul, attention) and hurts where the
answer is a syntax pattern (elementwise, threadgroup tree reduce).

**Fair-budget experiment: bigger budget makes qwen3 worse.** The
obvious rebuttal to the qwen3 numbers is that `<think>` shares
`max_tokens=4096` with the emitted kernel, so on harder problems
the kernel gets truncated (recorded as `no_code`). To test this,
qwen3 was re-run at `max_tokens=5200` — the ceiling under Groq
free tier's 6000-tokens-per-minute cap for this model — with 65-
second between-request pacing. n=60 again.

Result: **0 confirmed correct at mt=5200 vs 5/60 at mt=4096.** Even
in the maximally generous salvage (5 records lost their transcript
to a harness bug; if every lost record were correct), mt=5200 ties
baseline at 5/60. Realistic bound: mt=5200 is neutral-to-worse.

The mechanism is visible at kernel level. On `p003_elementwise_mul`
qwen3 at mt=4096 emitted the correct MSL attribute:

    uint id [[thread_position_in_grid]]   // mt=4096, verifies correct

At mt=5200, given ~1100 extra tokens of `<think>`, the same model
on the same problem emitted:

    uint id [[thread_index_in_grid]]      // mt=5200, compile-fail

`thread_index_in_grid` is not a real MSL attribute — it's an
invented one, plausibly derived by reasoning about what the name
"should" be. The same `position_in_grid → index_in_grid` overwrite
appears on `p010_abs` (also correct at mt=4096, compile-fail at
mt=5200). Response length on both problems grows from ~3K to ~7K
chars, entirely in the `<think>` block.

**More reasoning budget does not fix — it worsens — the exact
failure mode reasoning-vs-pattern-match already produces.** The
extra tokens go into more elaborate derivations of the wrong
adjacent-language pattern, not toward the correct pattern-match
answer. `<think>` truncation was not the bottleneck for this model
on this benchmark; the overwrite is the story.

(Testing this at a *materially* bigger budget — e.g. 16384 tokens,
where the entire reasoning trace fits and doesn't compete with the
kernel emit — would need a paid tier: Groq's 6000 TPM cap for this
model blocks it, and every request at 16384 tokens 413's
immediately.)

Failure taxonomy on T2 splits the non-reasoning models cleanly:

- **Gemini** reaches for the "right" pattern (threadgroup memory,
  atomics) but writes MSL syntax wrong at the *declaration* level —
  most T2 failures are compile errors.
- **Groq llama-3.3-70b** writes safer code that compiles and produces
  correct output, but at speeds well below MPS.
- **Groq llama-3.1-8b** mostly compile-fails at T2 like Gemini, plus
  some verify-fails where the reduction produces wrong numbers.

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

## 5. Repair@5: a syntax-unsticker whose payoff is model-dependent

Phase 3's second headline test: does giving the LLM its own compile
diagnostics as feedback let it unstick past errors? Ran `repair@5`
(up to 5 attempts per problem, feeding the failure back). Comparing
one_shot vs repair fairly means counting only problems a given model
ran in *both* modes.

**An earlier version of this section drew the wrong conclusion from too
little data.** With only llama-3.3-70b's partial sweep in hand, it
reported "repair@5 adds zero new correct kernels" and generalized that
to "repair fixes syntax, not semantics." A later full repair sweep on a
*stronger* model (gpt-oss-120b) refutes the generalization — while, as it
happens, confirming the underlying mechanism. Both facts matter.

**The two models, paired one_shot → repair:**

    llama-3.3-70b   T1+T2 (27 shared)   19/27 → 19/27   (+0)
    gpt-oss-120b    T3+T4 (19 shared)    3/19 → 16/19   (+13)
                    of which T3 (15):     3/15 → 13/15   (+10)

So repair@5's payoff is not a constant — it ranges from *nothing* to
*recovering most of the tiled-kernel cliff*, depending on the model.

**What repair actually fixes (the mechanism holds).** Of gpt-oss's 10
new T3 wins, **9 were one_shot compile-fails and 1 was a verify-fail**:

    p201, p205, p206, p209, p210, p211, p212, p215, p204   compile → PASS
    p208_conv2d_5x5_tiled                                   verify  → PASS

Repair is overwhelmingly a **syntax-unsticker**: the compiler *tells* the
model what's wrong and a capable model acts on it. It almost never drove
a kernel from "wrong numbers" to "right numbers" (1 of 10). The old
llama-only observation — compile-fail → verify-fail mode-shifts with no
new passes (`p102_row_max`, `p104_row_softmax`) — was the *same mechanism*
seen through a model too weak to finish the fix.

**Why the payoff differs so wildly.** Repair@5's lift ≈ (how many of a
model's one_shot failures are fixable *syntax* errors) × (whether the
model can act on the diagnostic). gpt-oss cliffs on T3 by writing MSL
that doesn't *compile* — highly fixable, and it fixes them. llama's
failures are either unfixable-by-it or already past the compiler into
wrong-semantics territory, where the scalar-error signal isn't pointed
enough. Same repair loop, opposite outcomes.

Mapped onto the layered analysis of §4:

- **Layer 1 errors** (compile fails from wrong syntax): the compiler
  *tells* the model what's wrong. Repair fixes these — *if the model is
  strong enough to act on the message.* This is where the entire +13
  came from.
- **Layer 2 errors** (verify fails from wrong idioms): compiler silent,
  code runs, semantics wrong. Scalar-error feedback fixed exactly 1.
- **Layer 3 errors** (slow but correct): invisible to compile and verify
  signals both. Would require performance feedback.

**Concrete implication for Phase 5 / Phase 6:** repair@5 with only
compile + scalar-correctness signal is a **layer-1 unsticker, and a
strong one** — for a capable model it recovers most of the T3 cliff for
free. It is *not* a floor-raiser: it does nothing for a model that can't
act on its own errors, and it barely touches layer-2 semantics for
anyone. Pushing past that needs a richer feedback surface — per-element
diff previews, shape-mismatch hints, or performance-vs-reference signals.
The repair transcripts collected here (38 converged transcripts, 15 of
them success-after-repair, plus 23 labeled negatives) are the Phase-6 SFT
seed set built on exactly this signal.

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
| Compile error (layer 1) | `mem_device` flag / wrong keyword | **Yes, if the model can act on it** — drove gpt-oss's entire +10 on T3; nothing for llama | n/a |
| Verify error (layer 2) | Wrong reduction pattern, off-by-one tile bound | Rarely — scalar error not pointed (1 of 10 fixes) | Yes, given right feedback |
| Slow correct (layer 3a) | Missing SIMD-group reductions, no fusion | No — no perf feedback in loop | Yes |
| Slow correct (layer 3b) | Spec's launch forbids MPS-shape design | No | **No** — benchmark artifact |

## 8. What's next

- **Phase 5** (web demo) is underway — the leaderboard now carries the
  repair@5 runs and a one_shot-vs-repair lift chart (§5). **Phase 6**
  (data flywheel, separate repo) has its seed set: repair transcripts for
  all 60 problems, exported to SFT-ready JSONL (38 converged, 15
  success-after-repair). The fine-tune + re-benchmark loop is the next
  real work.
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
