# MSL-Bench session notes

A running log of what got built, what broke, and what I learned. New entries
go on top. Concept explanations and lessons (not just changelog) are the point.

---

## 2026-07-20 (later) — Phase 6 first fine-tune: MLX-LM LoRA runs, two bugs, a thermal lesson

First local fine-tune actually trained. Base `Qwen2.5-Coder-7B-Instruct-4bit`,
LoRA via MLX-LM, on the 34-example `sft_write` set (+4 valid). Recipe now lives
in `phase6/lora_config.json`.

### Bug 1: `iters` self-clobbered to 1 (silent no-op run)

Smoke-tested with `--iters 1`. MLX-LM **saves the resolved run config back into
`adapter_path` at the end of training** — which was the exact same file I passed
to `--config` (`phase6/adapters/adapter_config.json`, since the recipe lived
inside the output dir). So the smoke run overwrote `iters: 100 -> 1`, and every
later run read the clobbered file and re-saved `1`. Two "full" runs were silent
1-iter no-ops before I caught `Starting training..., iters: 1` in the log.
**Fix:** move the source recipe OUT of `adapter_path` (`phase6/lora_config.json`),
so the trainer's output-dir writes can't touch it. General rule: never point
`--config` at a file that lives in a dir the tool writes to.

### Bug 2 / the real lesson: a 1-iter probe does NOT predict sustained load

Told the user the run would be "light" based on smoke numbers (13.5 GB peak,
0.88 it/s). Wrong. The real run's memory GREW as it hit longer examples:
iter 10 -> 18.4 GB, iter 20 -> 22.8 GB (71% of 32 GB), it/sec collapsed ~9x
under memory pressure. User physically felt the machine heat up; killed it.
Same shape as the Groq "false green light" lesson above — **a single probe
tells you almost nothing about sustained behavior.** Watch the *trend* (peak-mem
per report), not the first data point.

Why it ballooned + how the lighter config fixed it (peak 22.8 -> 7.5 GB):
the big variable memory is **stored activations** kept for the backward pass,
scaling ~ batch x seq_len x model-layers.
- `grad_checkpoint: true` = biggest lever. Stops storing most activations,
  recomputes them in backprop instead (trade compute for memory). This alone
  did most of the 3x drop.
- `batch_size 2->1`, `max_seq_length 3072->2048` = linear cuts to the same pile.
  (2048 chosen deliberately: max example ~1995 tok, so no truncation. 1024 would
  have truncated 6/38 targets = broken kernels.)
- `num_layers 8->4` = the trap: in MLX-LM this is *how many layers get a LoRA
  adapter*, NOT model size. Base still runs its full forward pass, so this barely
  touched memory — it only shrank the (tiny, 5.767M-param) trainable set.

### Result: overfitting U-curve -> use the iter-50 checkpoint, not the final

Val loss: 0.767 (1) -> 0.593 (25) -> **0.556 (50)** -> 0.593 (75) -> 0.637 (100).
Train loss kept dropping (0.8 -> 0.33); classic memorization of 34 examples.
Default `adapters.safetensors` = iter-100 (the WORSE one). `save_every: 25` left
us `0000050_adapters.safetensors` as the best-generalizing checkpoint.
**Benchmark that one.** (This is why 34 examples is a proof-of-concept, not a
model — expected per the phase-6 plan.)

### Serving: skipped fuse+GGUF, served the adapter directly

Plan said fuse -> GGUF -> Ollama. Cheaper route: `mlx_lm.server --adapter-path`
serves base+LoRA over an OpenAI-compatible API, no fuse, no lossy 4-bit->GGUF.
Added a `local` provider to `mkb/llm/providers.py` (~15 lines, mirrors `groq()`,
points at :8080, no key) + `--provider local` in run_suite. Two gotchas:
- The request `model` field is NOT ignored — the server keys on it to pick the
  loaded model, and an unknown value is download-attempted as a HF repo (got a
  404 on placeholder "mlx-local"). Must pass the exact base repo id.
- Base-vs-FT control needs distinct run tags. FT run used default model label
  (`local_default_one_shot`); base control passed `--model <base repo id>` so
  the tag differs and files don't collide.

### Eval: N=34 SFT gives NO functional lift on T2 (rigorous, with control)

T2 one_shot, iter-50 adapter vs base Qwen2.5-Coder-7B-4bit, same 15 problems:

                        base    FT(iter50)
    T2 pass             0/15    0/15
    OpenCL get_*_id     12/15   13/15    (within noise @ temp 0.2)
    correct [[thread]]   2/15    2/15
    fail stage        compile  compile

Identical. **Skeptical checks that make this a real finding, not an artifact:**
- Not a data bug: 0/34 training targets use OpenCL, 34/34 use `[[thread_...]]`.
- Not a parse/`<|im_end|>` artifact: fences clean, token stripped.
- **Not an adapter-not-loaded artifact** (the important one): greedy (temp 0)
  same-prompt diff, base vs +adapter, outputs differ completely -> adapter IS
  active. Base emits `get_global_id(0)`; adapter emits
  `[[thread_position_in_grid(0)]]` ... `(1)(2)(3)` on six params. So SFT DID
  shift OpenCL->Metal *surface form*, but learned the token not the grammar
  (it's a single unindexed attribute on one param) -> still won't compile.

**Conclusion:** the flywheel machinery works end-to-end (train->serve->bench->
mechanistic readout), but 34 examples is far too little to move a 4-bit 7B's
OpenCL prior into correct MSL. Textbook undertraining, shown by mechanism.
Matches the plan's "38 is tiny, expect noise" caveat, now with a control.

### Carry-forward (real lift needs, in priority order)

1. **~10x more seed data** — more models sweeping repair over more problems.
2. **Held-out problem split** (Phase-2.5: author ~15 reserved test problems) so
   any future "accuracy went up" isn't same-suite memorization.
3. Consider a non-4bit base and/or greedy decoding (temp 0) for the eval to cut
   sampling noise. 34-ex on 4-bit was never going to show lift; don't over-read.
4. dpo_negatives (23) still unused — preference tuning is a later lever.

## 2026-07-20 — Groq TPD is a rolling window; demo gets a repair-lift chart

### Lesson: a single successful request does NOT prove a quota reset

Tried to build a clean single-model gpt-oss-120b repair curve (reuse the T3
seeds, fill T1/T2/T4) now that it was "a new day." Probed with one small
request — it succeeded — so I read the TPD as reset and launched the sweep.
It 413'd/429'd on the second problem: `Used 199114 / Limit 200000`.

Groq's TPD is a **trailing 24h rolling window, not a calendar-day reset.**
Last night's ~200k was still inside the window, leaving ~900 tokens of
headroom that my tiny probe slipped through — a false green light. **Check
the `Used N / Limit M` figure in a 429, not just "did one request go
through."** The single-model curve is blocked until last night's tokens age
out (~24h) or a paid Dev tier lifts the cap. Logged as a blocked task.

### Demo: `.repair_lift()` — the honest one_shot-vs-repair chart

The leaderboard sorts by `fast_1` (speed), which hides the actual repair
finding (a *correctness* story). Added a paired-bars small-multiples chart
(scripts/make_demo.py `build_repair_lift` + hand-rolled SVG in index.html).

Key design constraint for honesty: **compare only problems a model ran in
BOTH modes** (mt-budget variants collapse to one base model). The chart then
shows the contrast that matters:

    gpt-oss-120b  T3: one_shot 3/15 -> repair 13/15  (+10)
    gpt-oss-120b  T4: one_shot 0/4  -> repair  3/4   (+3)
    llama-3.3-70b T2: one_shot 4/12 -> repair  4/12  (+0)   <- flat
    llama-3.1-8b  T4: one_shot 1/11 -> repair  1/11  (+0)   <- flat

**Repair@5 is a force multiplier on capability, not a floor under it.** It
recovers most of the T3 cliff for a strong model that can act on its own
error text, and does literally nothing for a weak one. Sharpens the Phase-4
story: the benchmark now distinguishes "can't write the kernel" from "can't
*fix* the kernel," and only the former is fatal for capable models.

### First link toward Phase 6: SFT export (scripts/export_sft.py)

Turned the repair transcripts into training data. Design decision worth
recording — a repair flywheel wants two framings, and mixing them naively
would train the model on broken kernels:

- **write** (38 ex): collapse each converged trajectory to (system, problem)
  -> FINAL correct kernel. The wrong intermediate attempts are dropped, so no
  training target is ever a broken kernel.
- **repair** (15 ex): the full converged trajectory for runs that actually
  needed a fix (attempts > 1) — teaches "given a broken kernel + its error,
  produce a corrected one." SFT tooling must mask loss to the final assistant
  turn only; the earlier wrong turns are prompt context, not targets.
- **dpo_negatives** (23): never-converged trajectories, kept separate and
  labeled for future preference tuning — NOT folded into SFT (training on a
  trajectory that ends wrong teaches wrong).

Gotcha caught in validation: bare provider-abort records clean down to zero
assistant turns (all that survived was the role:"error" marker). Filtered
them out of the negatives — a negative with no kernel carries no signal.

The JSONL is derived data (regenerable from results/raw via the script), so
it's gitignored like the raw records; the repo tracks the exporter, not the
dataset. The actual SFT run + re-benchmark live in the separate Phase-6 repo.

## 2026-07-19 — repair@5 flywheel completed: 60/60 seed transcripts, three quota walls, one harness fix

Closed Phase 3: every one of the 60 problems now has a real repair@5
transcript in `results/raw/`. These transcripts are the flywheel's seed
trajectories (repair transcripts -> SFT -> re-benchmark, Phase 6). Getting
there meant fighting Groq/Gemini free-tier quotas the whole way, and the
*mechanisms* of those walls are the real lesson.

### The dataset (best seed per problem)

    T1: 15/15 correct   T2: 5/15   T3: 13/15   T4: 4/15
    15 success-after-repair seeds (passed on attempt > 1) — highest value
    17 full 5-attempt fail chains (rich fail->feedback->fail seeds)

**Provenance is mixed and that's deliberate** — no single free model could
finish all 60 in one night, so the best-seed set is a patchwork, traceable
per-record via the `run` tag:
- llama-3.3-70b: T1 + T2 (small specs, its TPD covered them)
- gpt-oss-120b: T3 + T4 p301–p304 (the strong model, best T3 seeds)
- llama-3.1-8b: T4 p305–p315 (fallback once the strong models walled)
- gemini-2.5-flash: 1 leftover

Caveat for any future report: **the per-tier pass rates above are NOT one
model's difficulty curve** — they mix three models. T3 13/15 is gpt-oss;
T2 5/15 is llama-3.3-70b. Don't read them as a single-model cliff.

### Three distinct quota walls (they are not the same thing)

1. **TPD — tokens per day, per model.** Hit twice: llama-3.3-70b (100k/day,
   died at T1+T2) and gpt-oss-120b (200k/day, died mid-T4 at p305). Each
   model has its *own* daily bucket, so switching models buys a fresh one —
   that's why the pivot to gpt-oss worked at all. Resets ~24h; no way around
   it tonight except changing model.

2. **TPM — tokens per minute — inflated by a reasoning reservation.**
   gpt-oss-120b free tier caps at 8k TPM, but a request with a 560-token
   prompt was billed as **11,733**. The gap is the completion *reservation*:
   Groq counts `max_tokens` (4096) AND, for a reasoning model, a large
   reasoning budget (~7k) against the per-minute limit — not actual tokens
   used. Concept worth internalizing: **on Groq, request size for
   rate-limiting = prompt + max_tokens + (reasoning reservation), so
   lowering `max_tokens` directly shrinks the billed request.** Measured:
   mt=4096 -> 413 (too large); mt=3000 -> fits AND emits code; mt=1500 ->
   fits but reasoning eats it all, zero code out. mt=3000 was the sweet spot.

3. **Same TPM wall on a non-reasoning model.** llama-3.1-8b (~6k TPM) 413'd
   on repair *attempt 2* at mt=4096 — no reasoning reservation, just the
   4096 completion reservation plus a grown transcript. Dropping to mt=2000
   bought 2–5 real attempts per problem before the growing history crossed
   the cap. (An 8B model only emits ~500 tokens of kernel anyway, so mt=2000
   costs nothing in code quality.)

Takeaway: **repair@k is quadratic-ish in tokens** — each attempt resends the
whole growing transcript — so it collides with per-minute caps far sooner
than one-shot does. Budget `max_tokens` against the TPM cap, not against
"how long might the answer be."

### Harness fix: never throw a partial repair transcript away

`repair_k` used to let a provider error propagate mid-loop. `run_suite`'s
outer handler then saw `gen is None` and saved *only the error* — discarding
every attempt we'd already paid for. Root cause: the transcript lived inside
`repair_k`'s local `messages`, unreachable once the exception escaped.

Fix (mkb/llm/generate.py): wrap the `providers.complete` call inside the
loop; on exception, append an error marker and **return the partial
transcript** with `success=False`, `attempts=attempt-1`, and a
`provider_error` field. Now a TPM 413 on attempt 5 keeps attempts 1–4 as
seed data. Validated live on p202 (kept 4 attempts) and p304 (kept 4). This
is the same "never throw transcripts away" rule the run_suite except-block
already followed — the gap was just that repair_k didn't return its own
partial when the provider call itself threw.

Minor cost: a genuine daily-quota exhaustion is now recorded as a per-problem
partial-fail instead of loudly stopping the sweep. The `provider_error`
field + `role:"error"` transcript entry keep it distinguishable, so
leaderboard code can still filter provider aborts from honest repair@k
exhaustion.

### Finding: repair@5 closes the T3 cliff for a strong model

The Phase-4 story was "LLMs cliff-dive from T1 to T2 and stay down." Repair
data complicates that in a good way:
- gpt-oss-120b one_shot T3 was ~20% correct; **repair@5 got 13/15.** The
  fail->feedback->retry loop recovers most of the T3 (tiled) cliff *if the
  model is strong enough to act on the feedback.*
- llama-3.3-70b was 0/8 on T3 (one_shot and repair alike) — it can't use
  the feedback, so repair doesn't help. This is exactly why gpt-oss's
  verify-fails were predicted (07-03 notes) to be higher-value seeds.
- T4 (fused) stays hard even with repair + a strong model: best is 4/15.
  Fused kernels (attention, layernorm, swiglu) are where repair@5 still
  can't rescue the answer.

So the sharper claim for the report: **repair recovers the tiled tier for
capable models but not the fused tier — the real, repair-resistant wall is
T4, not T2.**

### Caveat logged: don't headline the T3 conv speedups

p213 conv1d "27×" and p214 conv2d_stride2 "41×" are the
wall-clock-vs-GPU-clock artifact already documented at the bottom of this
file: `reference_ms` is CPU wall-clock (includes MPS dispatch + Python
overhead), `kernel_ms` is pure GPU. Conv on small tensors is exactly where
MPS per-dispatch overhead dominates, so the ratio flatters the kernel.
Correctness is real (passed verify at the problem's tolerance); the *speed*
number is not a real 40×. Re-verify with reverify.py before ever quoting it.

### Open items

- Single-model repair completeness (one model across all 60) still blocked
  on TPD resets; llama-3.3-70b could finish its own T3/T4 tomorrow. Only
  matters if we want a clean single-model repair curve for the report.
- A paid Groq Dev tier removes the 8k TPM cap and would let gpt-oss-120b do
  the full 60 in one pass — worth it if the flywheel needs strong-model
  seeds on every problem rather than the current patchwork.

## 2026-07-09 (later) — why tiers exist: MSL-knowledge calibration, not just difficulty

Recording this for the report because "why is the suite tiered" is the
first question anyone with a benchmark-design instinct asks, and the
answer is more specific than "harder problems."

### The design claim

**Each tier is calibrated around what MSL-specific patterns the model
actually needs to know to write a working kernel — not just around
compute intensity or line count.** T1 problems could be written in
nearly any C-like language; T2+ problems can only be written correctly
if the model knows Metal-specific idioms that don't map 1:1 from CUDA.

    T1 elementwise  → one thread per output, no cross-thread coord
                      example: `out[i] = f(x[i])`
                      MSL-specific knowledge required: ~none
                      real-world use: activation functions
                      (ReLU, sigmoid, tanh, GELU, abs, exp, ...)

    T2 reductions   → first tier requiring `threadgroup` memory +
                      `threadgroup_barrier`. Many threads collaborate
                      to compute one output.
                      example: row_sum, row_max, softmax, prefix scan
                      MSL-specific knowledge required: threadgroup
                      address space, barrier memory flags, sometimes
                      simd_shuffle_down / simd_sum
                      real-world use: softmax, layer stats, norms

    T3 tiled        → cooperative loading of shared tiles into
                      threadgroup memory + tile-index math. Multiple
                      threads read overlapping data.
                      example: matmul, matmul_simdgroup, conv2d, sgemv,
                      transpose_tiled
                      MSL-specific knowledge required: tile staging,
                      halo pattern, index-flattening math
                      real-world use: the core of every neural net layer

    T4 fused        → multiple ops in one kernel — reduction +
                      normalize + activation all in one dispatch.
                      Hits every level of the memory hierarchy.
                      example: LayerNorm, RMSNorm, SwiGLU, causal
                      masked attention, GELU-fused MLP
                      MSL-specific knowledge required: T2 + T3
                      combined, plus in-register accumulation,
                      plus dispatch-overhead awareness
                      real-world use: the actual building blocks of
                      modern LLMs

### Why this specific design (vs. just a mixed pile)

If the suite were 60 unranked problems, the aggregate score would give
you an average — informative but not diagnostic. Tiering makes the
failure mode legible. It lets the benchmark answer **one question
cleanly: at what level of Metal complexity do LLMs cliff-dive?**

That question is only answerable because we can separately measure
correctness at each tier. Without tiers, "the model got 45% correct"
tells you a number; with tiers, you get "the model handles all
elementwise but fails 80% of the time on anything cross-thread" —
which is a mechanism.

### The empirical result the tier design surfaces

Universal across every model tested — llama-3.1-8b, llama-3.3-70b,
gpt-oss-120b, gemini-2.5-flash, qwen3-32b — same shape:

    T1: 87–100% correct  (some models 100%)
    T2: 7–47% correct    (best is gpt-oss-120b)
    T3: 0–20% correct
    T4: 0–20% correct

The absolute numbers vary; the *cliff* between T1 and T2 does not.
That's what pins the finding: it's not "LLMs are bad at Metal in
general." It's specifically that they can't reliably use `threadgroup`
memory + `threadgroup_barrier` + SIMD-group intrinsics — the MSL-
specific patterns that don't have 1:1 CUDA equivalents. CUDA's
`__syncthreads()` has no flag parameter, so models don't know Metal's
barrier flags are load-bearing. CUDA's `__shared__` maps to Metal's
`threadgroup` address space but not identically. Etc.

### T1 vs T2 side by side (put this in the report)

T1 vector_add — trivial, no MSL-specific idioms:

    kernel void vector_add(device float *a [[buffer(0)]],
                           device float *b [[buffer(1)]],
                           device float *out [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
        out[id] = a[id] + b[id];
    }

T2 row_sum — requires understanding Metal's memory model:

    kernel void row_sum(device const float *x [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        threadgroup float *scratch [[threadgroup(0)]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint row [[threadgroup_position_in_grid]]) {
        scratch[lid] = x[row * K + lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);  // ← flag matters
        for (uint s = K/2; s > 0; s /= 2) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);  // ← models forget this
        }
        if (lid == 0) out[row] = scratch[0];
    }

The delta between those two is where the whole "MSL is not CUDA"
story lives. Every T2 kernel Groq's llama-3.3-70b wrote had *some*
version of that structure — but with `mem_device` instead of
`mem_threadgroup` on the barrier, which costs ~7% perf on p101_row_sum
alone (measured, session 07-03). That's a real, quantifiable Metal-
specific pattern the model didn't know.

### Real-world grounding for T3 and T4

T3 tiled kernels are **the core of every neural net layer**. Every
`nn.Linear`, `nn.Conv2d`, `torch.matmul` call ultimately dispatches
to a tiled kernel. If an LLM can't write a working matmul, it can't
write a custom fused layer that beats the framework.

T4 fused kernels are **exactly what fast LLM inference stacks compete
on**. RMSNorm-fused, SwiGLU, flash-attention variants — production
ML performance work is fusing these ops to reduce dispatch overhead
and memory bandwidth. That's what MPS's implementations do; that's
what MSL-Bench is asking whether LLMs can also do.

The T4 layernorm result from the 06-30 sessions (hand-tuned RMSNorm
kernel at 8.7× MPS, one dispatch replacing four `torch.mean/sqrt/div/
mul` calls) is the proof that this territory has real speedup wins
available — the LLM benchmark measures how close models get.

### Suggested phrasing for the report

Consolidated line: **"Tiers are calibrated around what MSL-specific
patterns each problem forces the model to use — T1 needs no Metal-
specific knowledge (it's C math with Metal boilerplate); T2 requires
threadgroup memory and barriers; T3 requires tile staging; T4 requires
all of the above plus fusion awareness. The tier gradient is a proxy
for 'how much of the Metal Shading Language do you actually need to
know?'"**

Complementary line: **"Every model tested handles T1 fine and cliff-
drops at T2. The cliff is universal because that's precisely where
MSL and CUDA diverge — CUDA's `__syncthreads()` has no memory-scope
flag; Metal's `threadgroup_barrier` does, and picking the wrong one
costs measurable performance even when the kernel verifies correct."**

Would fit as a §2 or §3 addition to REPORT.md.

---

## 2026-07-09 — the qwen3 fair-budget experiment: bigger `<think>` is actively worse

Ran the fair-budget rebuttal to the "qwen3 is bad because `<think>`
truncates the kernel" hypothesis from the 07-04 session. Result is
a clean null-plus: bigger reasoning budget doesn't just fail to
help qwen3-32b on this benchmark, **it makes it strictly worse**.
Session also surfaced two harness bugs, one of which cost real data.

### What got built

    scripts/run_suite.py         --max-tokens flag + _mt{N} run-tag
    mkb/llm/generate.py          plumb max_tokens through one_shot / repair_k
    mkb/llm/providers.py         plumb max_tokens through groq/gemini/ollama
    scripts/reverify.py          new — post-hoc re-verify skipped records
    .gitignore                   ignore results/logs/ (sweep logs)

The `--max-tokens` flag exists so reasoning-model budgets can be
varied without silently mutating a shared default. `_mt{N}` in the
run tag prevents non-default budgets from overwriting baseline
records — every budget becomes its own leaderboard row.

`scripts/reverify.py` exists because of a harness bug (below).

### The token math and why 5200 was the honest ceiling

Groq free tier caps qwen3-32b at 6000 tokens/minute (TPM). A single
request "reserves" `prompt_tokens + max_tokens` against that budget.
Measured actual prompt sizes across 60 problems: 340–600 tokens.
Under the worst-case 600-token prompt, `max_tokens ≤ 5400`. Chose
**5200** for a 200-token safety margin, plus **--sleep 65** to
respect the rolling minute (retry backoff is 15+30+45 = 90s, enough
to unstick occasional bunching).

A first attempt at 16384 (session notes 2026-07-04 open items) 413'd
every request — the max_tokens value alone exceeded the TPM cap.

### Result: 0/60 vs baseline 5/60 (or 5/60 tie at max salvage)

    baseline mt=4096:  5/60 correct  (T1:3, T2:0, T3:1, T4:1)
    experiment mt=5200: 0/60 confirmed correct
                       (T1:0, T2:0, T3:0, T4:0)
                       + 5 records with LLM output lost to harness bug
                       → max salvage = 5/60 (tie), realistic = worse

Which specific problems went from correct → wrong at bigger budget:

    p003_elementwise_mul  mt=4096 correct  →  mt=5200 compile-fail
    p010_abs              mt=4096 correct  →  mt=5200 compile-fail
    p314_swiglu           mt=4096 correct  →  mt=5200 compile-fail
    p008_tanh             mt=4096 correct  →  mt=5200 lost transcript
    p209_matmul_naive     mt=4096 correct  →  mt=5200 lost transcript

### The mechanism, at kernel level

For p003_elementwise_mul (a one-line elementwise mul), the two
kernels differ by exactly one token:

    mt=4096 (correct):
        uint id [[thread_position_in_grid]]

    mt=5200 (compile-fail):
        uint id [[thread_index_in_grid]]

`thread_index_in_grid` is NOT a real MSL attribute — it's invented.
qwen3, given ~1100 more tokens of `<think>`, reasoned about what the
attribute "should" be called and got it wrong. The same
`position_in_grid → index_in_grid` overwrite happened on p010_abs.
On both problems the response length grew from ~3K → ~7K chars,
entirely in the `<think>` block.

This is the "reasoning overwrites pattern-match" thesis from the
07-04 session (qwen3 vs non-reasoning llamas), now demonstrated at
*problem-level granularity within the same model*. Give qwen3 more
budget, watch it derive itself off a correct answer.

**The generalization: giving a reasoning model more thinking budget
does not help on domains where the correct answer is a memorized
pattern-match** — because the extra budget doesn't buy access to
the right pattern; it buys more elaborate wrong pattern-derivations
from adjacent languages the model does know.

Contrast with the story reasoning helps in: T3/T4 problems where
the algorithm structure matters (tiled matmul, attention). qwen3
baseline was 1/15 T3 and 1/15 T4 — the ONLY model in the roster
that got any T3 correct. That's plausibly reasoning helping. But
the T1 syntactic-pattern regime shows reasoning hurting cleanly.

### Two harness bugs surfaced

**Bug 1: launched with system python, not `.venv/bin/python`.**
Torch is installed in `.venv` but I ran `python3 scripts/run_suite.py`.
Torch is only imported inside `evaluate_kernel`'s verify step. Any
kernel that compiled + ran successfully then hit `ModuleNotFoundError`
at import torch, was caught by the outer exception handler, and
recorded as `provider_error`. Meaning: the "kernel is correct"
signal was silently converted to "provider error" for every
kernel that made it past compile+run.

Salvageable via re-verify from saved transcripts — except for bug 2.

**Bug 2: run_suite.py's exception handler is destructive.** In
`scripts/run_suite.py`:

    except Exception as e:
        ...
        transcript = [{"role": "error", "content": err}]

When evaluate_kernel raises (torch import, GPU OOM, anything
downstream of the LLM call), the transcript is REPLACED with just
the error text. **The actual LLM output — which we already paid
Groq to generate — is lost.**

For this sweep, that meant 5 records permanently lost their kernel
source: p004, p008, p207, p209, p315. Two of those (p008, p209) had
been correct at baseline; likely correct at mt=5200 too. We can't
verify them because the LLM output is gone.

The fix is trivial (preserve `gen['transcript']` if the LLM call
succeeded before the exception). But it's a data-integrity bug that
matters most for Phase 6, where these transcripts ARE the flywheel
seed data. Losing them silently is not OK.

### Decisions made

- Kept the compile-fail and no_code records from mt=5200 — they're
  trustworthy (no torch involvement) and constitute the main finding.
- Did NOT re-run the sweep to recover the 5 lost records. The maximum
  possible salvage (5/60 = 8.3%, matching baseline) doesn't change
  the "not worse than baseline, likely worse" conclusion. Spending
  another 70 minutes of Groq TPM on this would improve confidence in
  the margin, not the direction.
- REPORT.md's "Fair-budget caveat" paragraph was rewritten to become
  a full sub-finding ("bigger budget makes qwen3 worse") rather than
  a "we couldn't test this" caveat.
- Log file `results/logs/qwen3_mt16384.log` (the 413'd first attempt)
  is now gitignored via `results/logs/` — the mt5200 sweep's log
  is the useful artifact and is also gitignored.

### Open items (carry into next session)

- **Fix run_suite.py's transcript-loss bug.** Trivial change,
  should land before any future sweep. Preserve `gen['transcript']`
  when the LLM call succeeded and the exception was downstream.
- **Add a preflight check that torch is importable** in run_suite.py
  before starting the loop — would have failed fast on the venv
  mistake instead of silently ruining 5 records.
- **Update the leaderboard** with the mt=5200 row. It should sit
  next to the mt=4096 baseline as a genuinely new data point, not
  a re-run of the same experiment.
- **Groq free tier's 6000 TPM cap on qwen3-32b** blocks the
  materially-bigger-budget test (e.g. 16384). That test would need
  a paid tier or a different provider. Not blocking any current
  claim — the mt=5200 direction is clear enough — but a paid-tier
  qwen3 run would tighten the "bigger budget always hurts?" claim
  from "at 5200" to "at any budget large enough to actually think."
- **Phase 5 (web demo)** and **Phase 6 (data flywheel)** still the
  natural next-work. mt=5200 didn't change those priorities.

---

## 2026-07-04 (dawn) — two more Groq models, cross-model comparison sharpens the story

Ran two more providers on Groq's fresh per-model quota buckets after
llama-3.3-70b exhausted its daily TPD:

    Groq qwen/qwen3-32b       one_shot  36/60  (paused for cool-down)
    Groq llama-3.1-8b-instant one_shot  60/60  (first full sweep!)

Two cross-model findings landed that sharpen the Phase-4 story:

### Metal elementwise fits in an 8B model

Both 8B and 70B llama variants hit **100%** on T1. The MSL knowledge
needed to write `out[i] = f(x[i])` correctly is compact — fits in 8B
params, doesn't need 70B. Bigger models retain an edge only at T2+
(llama-3.3-70b: 22.2%, llama-3.1-8b: 6.7% on T2), where MSL-specific
patterns like threadgroup memory, barriers, and atomics start to
matter. That's a real finding for anyone deciding what size model to
target for GPU kernel generation: **the elementwise floor is cheap;
the T2+ ceiling is what buys parameters**.

### Reasoning-optimized may be actively harmful on niche syntax

qwen3-32b was the ONLY model in the set that failed T1. **3/15 T1
pass**, all others 100% or 91.7%. Sampled its transcripts: dominant
failure mode is using `thread_position_in_grid` as a free variable
(OpenCL/CUDA-style) instead of a Metal parameter attribute like
`uint id [[thread_position_in_grid]]`. Consistent across problems.

Reading the `<think>` blocks reveals the mechanism: qwen3 *derives*
thread-index handling from CUDA/OpenCL knowledge, applying an
adjacent-domain pattern confidently. It's not lacking Metal
knowledge — it's actively overwriting Metal knowledge with reasoned-
from-adjacent-language reasoning.

Non-reasoning models that just pattern-match "the closest Metal
example from training data" get the parameter-attribute syntax right
by not thinking about it. Reasoning helps when the domain is familiar
enough that the derivation lands in the right place. On niche
territory like Metal, reasoning can be a Trojan horse for the wrong
adjacent-language patterns.

Also a practical issue: qwen3's `<think>` tokens compete with output
tokens under our shared `max_tokens=4096` config. On harder T2/T3
problems, `<think>` eats the budget and the kernel gets truncated
(no_code failures). Fair-test issue: reasoning models arguably need
a bigger budget to compete on the same terms.

### The T1→T2 cliff is universal, not model-specific

Across all four non-reasoning models tested, T1 sat between 92% and
100% and T2 dropped to somewhere between 6% and 22%. The cliff isn't
about model quality — it's about the *domain*: T1 is a one-line
translation of `f(x[i])`; T2 requires MSL-specific idioms. The
KernelBench thesis ("LLMs handle elementwise, break on cross-thread
coordination") reproduces cleanly on Metal.

### make_leaderboard fix

The partial-sample flag was comparing to max_n across runs, which
missed the case where two runs share the same n but tested different
subsets. After Phase 2's suite expansion from 36 to 60 problems,
llama-3.3-70b (36 old problems) and qwen3-32b (first 36 of the new
60) both looked like "full" rows despite testing overlapping-but-not-
identical sets of 27 shared + 9 different. Fixed by comparing to
total problem count from `problems/` discovery instead. Every non-60
row now correctly flagged.

### Decisions made

- Kept qwen3-32b's partial n=36 in the leaderboard rather than
  discarding. The 8.3% correct is a real finding, and the partial-
  sample flag now transparently marks it as not-complete.
- Left demo/ untouched (external HTML export from earlier, ownership
  unclear).

### Roadmap status at true session end

    [x] Phase 0
    [x] Phase 1
    [x] Phase 2  (60 problems)
    [~] Phase 3  (4 providers x one_shot, mostly complete;
                  repair@5 partial for 2 providers, quota-locked)
    [x] Phase 4  (REPORT.md drafted, updated with cross-model findings)
    [ ] Phase 5  future work
    [ ] Phase 6  future work, separate repo

### Open items

- **Complete qwen3-32b sweep** (24 remaining problems) — needs Groq
  daily bucket to refresh. Would flip the row from partial to full.
- **Ollama sweep** on the 60-problem suite — the last remaining
  provider option not blocked by anything except thermal budget.
  Would give a genuinely-local reference row alongside the cloud
  providers.
- **Increase max_tokens for reasoning models** if we test more of
  them. 4096 is too tight for `<think>` + kernel; 8192 or 16384
  would be a fairer comparison. But changing prompt config is a
  benchmark-design decision, not something to silently vary.

---

## 2026-07-03 (night) — Phase 2 to 60 + Phase 4 report drafted

Big final push. Went from 36 → 60 problems (T1×15, T2×15, T3×15,
T4×15) and drafted `REPORT.md` synthesizing the whole session's
findings into a publishable Phase-4 article.

### On writing 24 problems in one push

Cheated intentionally: wrote only `spec.py` for each, no scaffold and
no reference kernel. The benchmark's discovery/harness mechanism only
requires spec.py — scaffolds are teaching-aid polish, reference
kernels are goldens for LLM comparison. Cutting those saved ~2/3 of
the per-problem writing time and let all 24 land in one session.

Verified all 24 the mechanical way: `P.load()` + `P.make_inputs()` +
call `reference()` — checked that outputs have the expected shape.
Not a full end-to-end verify (would need actual candidate kernels
to compile and time), but enough to catch shape mismatches, missing
imports, and broken tolerance dicts before committing.

Caught one bug during writing: p213_conv1d originally flipped the
kernel weights, thinking F.conv1d does mathematical convolution.
It does cross-correlation. Removed the flip.

### On problem selection

Distribution rounded to 15/tier. Picked problems that:

- Have a one-line PyTorch reference (no exotic ops)
- Compute cleanly in float32 (no int8, no fp16)
- Exercise a specific idiom or workload:
  - **column-strided reduction** (`p113_col_max`) tests coalescing on
    the strided axis of row-major storage
  - **batched matmul** (`p211`) uses the grid z-dim for batch
  - **SwiGLU vs GLU gate** (`p314`, `p315`) — same shape, different
    activation, measures the LLM's ability to compose primitives
  - **causal vs additive-mask attention** (`p312`, `p313`) — two
    common transformer mask patterns
  - **fused matmul + activation** (`p308_gelu_fused`) — the exact
    pattern where MPS fusion vs LLM naive-two-kernel-approach could
    show up

Not included (out of scope for float32 baseline suite):
- int8 matmul (quantization dtype territory)
- backward passes (gradients need paired forward/backward specs)
- multi-head attention (would need to spec head splitting)

### On REPORT.md

Wrote as a publishable article, not an internal doc. Aim: someone
reading it end-to-end without background can follow the story from
"here's what we measured" through the layered decomposition to
"here's what needs to change before Phase 5." Kept it under 1500
words, tables where they help.

Deliberate choice: called out the **spec-launch cap** finding
prominently (§6). This is a benchmark-design bug I only discovered
this session, and pretending it isn't there would compromise the
report's honesty. Better to name it, quantify it (~35% of gap on
memory-bound problems), and propose three fixes for future work.

### Roadmap status at session end

    [x] Phase 0 — vertical slice
    [x] Phase 1 — timing trust
    [x] Phase 2 — 60-problem suite    (added 24 this session)
    [ ] Phase 3 — one_shot done, repair partial (quota-limited)
    [x] Phase 4 — REPORT.md draft
    [ ] Phase 5 — web demo (future)
    [ ] Phase 6 — data flywheel, separate repo (future)

Phase 3 sits at ~90% — the remaining 10% is repair@5 completion
on more providers, which needs either paid tier or another day's
free-tier bucket. Not a technical block, a quota block.

### Decisions made

- Phase 2 marked complete at 60. If the "~60" wording ever bothers
  someone, drop the tilde — we're exactly at 60.
- Report lives in `REPORT.md` at repo root, not in a `docs/` folder.
  Discoverability > organization for a small repo.
- Kept the `† partial sample` flag on the leaderboard (added earlier
  this session). Combined with the report's §6 caveat, readers now
  have two independent signals that not all rows compare cleanly.

### Open items (carry into next session)

- **Ollama qwen2.5-coder-14b repair@5 sweep** on the expanded 60-
  problem suite. Would give us a third leaderboard row + Phase 3
  gets closer to done. Needs thermal-safety greenlight given
  concurrent local LLM + kernel-eval load.
- **Reference kernels for the 24 new problems** — scaffold files
  and hand-written goldens would be nice to have for T1 as
  onboarding aids. Not required for benchmark runtime.
- **Decide the spec-launch cap question** (see REPORT.md §6). This
  is Phase-4/5 boundary work — the metric wording depends on which
  fix we pick, and Phase 5's public demo shouldn't ship until this
  is deliberate.
- **Run the expanded 60-problem suite through Groq one_shot again**
  once daily token budget refreshes. Would give a full n=60 baseline
  row and let us see how the T3/T4 additions extend the failure
  taxonomy.

---

## 2026-07-03 (evening) — repair@5 sweeps: both providers quota-die + the "repair adds zero new correct" finding

Ran `repair@5` for Groq llama-3.3-70b and Gemini 2.5 Flash to answer
the Phase-3 headline question: does compiler-diagnostic feedback let
the model unstick itself past the T2 compile-wall? Both providers
died on quota mid-sweep — this is now the second same-day quota kill
for both free tiers this session. Salvaged records: **Groq 17/36, Gemini 2/36.**

Groq daily hit at 98991/100000 tokens (TPD limit) after the last
attempt of p105. Gemini quotaed immediately (429) after p001 — its
daily bucket was already consumed by this session's earlier one_shot
sweep. Neither refills in a useful window, so this is what we have.

### The finding worth the whole session: repair adds zero new correct kernels

Aggregate leaderboard headline looks like a win: Groq repair 82.4%
correct vs one_shot 38.9%. This is misleading. Repair only ran on
17/36 problems (the easier subset); one_shot ran on the full 36.
Apples-to-apples on the same 17 problems, one_shot was **also 14/17**.
The full side-by-side:

    T1 (12 problems): 12/12 pass in both modes. Repair mode was
      indistinguishable from one_shot (attempts=1 on 11 of 12).

    T2 (5 problems, subset): both modes got 2/5. But the failure
      modes changed on 2 of them:

        p102_row_max      one_shot: compile-fail → repair@5: verify-fail
        p104_row_softmax  one_shot: compile-fail → repair@5: verify-fail
        p105_col_sum_tn   (repair got token-quota provider_error)

**Repair converts compile-fails into verify-fails but does not add
correct kernels.** That's the whole story. The compile diagnostics
guided the model past MSL syntax errors, and by the 5th attempt it
had a kernel that compiles + runs + produces *some* output. But the
output is wrong, and the "your output differs from reference" feedback
alone isn't enough to get from wrong-numerically to right-numerically.

This maps directly onto the barrier-flag finding from earlier today:

- **Layer 1** errors (barrier-flag confusion, missing keyword, wrong
  type annotation) — the compiler *tells* the model exactly what's
  wrong. Repair fixes these.
- **Layer 2** errors (missing MSL idioms, wrong reduction pattern)
  — the compiler is silent, the code runs, but the semantics are
  wrong. Repair only sees "wrong answer, max_abs_err=X" — not enough
  signal to point at the specific line.
- **Layer 3** errors (memory-bandwidth design, launch-shape choices)
  — invisible to both compile and verify signals. Would require
  performance feedback, which repair doesn't currently provide.

Groq's repair loop cleanly demonstrated layer-1 fixability and
layer-2 hard-stop. That's a concrete Phase-4 finding: **the value of
repair@5 with only compile+correctness signal is bounded above by
"unstick the model when compile errors were the whole problem."**
For anything past that, the feedback channel needs richer signal.

### Aside: the leaderboard misrepresents partial-sample rows

Groq repair (n=17) at 82.4% and Groq one_shot (n=36) at 38.9% are
NOT directly comparable — the 17 is a strict prefix of the 36 and
skewed toward the easy tiers. The leaderboard column shows n but
doesn't flag the overlap.

For Phase 4, `make_leaderboard.py` should probably grow either:
- an `n_shared` column (how many problems overlap the reference run), OR
- explicit "on same subset" delta rows, OR
- refuse to render rows with n < some threshold or n < max(n) - K.

Leaving as-is for this session — the data is honest, the
interpretation needs care. Both are captured in this note.

### Provider-error cells worth remembering

- `p105_col_sum_tiled_naive` in Groq repair: `provider_error`, false
  negative from the token-quota kill mid-attempt.
- `p002_relu` in Gemini repair: `provider_error`, same class of
  false negative from Gemini's quota kill after p001.

Both are recorded as `e` glyphs on the leaderboard, distinct from
`c/r/v/n` kernel failures. Anyone reading the raw records sees the
error string inline in the transcript.

### Decisions made

- README roadmap checkboxes updated: Phase 1 done, Phase 2 marked
  at 36/60 in-progress, Phase 3 marked at "one_shot done + repair
  partial" in-progress.
- Not pushing to retry Groq's quota via 20-min waits — the "rolling
  window" refill is ~1000 tokens per 20 min, which won't cover
  ~19 remaining repair problems (each of which can burn 5000+ tokens
  across 5 attempts).
- Ollama repair@5 sweep left as future work — would need thermal-
  safety confirmation given the concurrent local LLM + kernel-eval
  GPU load.

### Open items (carry into next session)

- **Ollama qwen2.5-coder-14b repair@5 sweep** — the only remaining
  provider option for repair data today is fully local. Needs
  thermal-safety greenlight before starting; sweep would take
  ~45-60 min with heavy sustained GPU load (local model inference
  + kernel eval, both on the same M-series GPU).
- **Enhance make_leaderboard.py to flag partial-sample rows.**
  Either an `n_shared` column, an "on same subset" comparison row,
  or a rendering rule that hides rows below a threshold.
- **Feed richer feedback into repair.** Layer-2 fixability might
  become possible if the feedback surface includes not just
  "max_abs_err=X" but per-element diff previews, or shape mismatch
  hints. Worth prototyping on p102 (the row_max problem where 5
  attempts of numeric-only feedback wasn't enough).

---

## 2026-07-03 (later) — Groq row lands + Cloudflare bot-signature bug

Second Phase-3 leaderboard row: `llama-3.3-70b-versatile` (Groq), one_shot,
all 36 problems in one uninterrupted sweep. **14/36 correct (38.9%)** vs
Gemini's 12/19 (63.2%) on a smaller sample. But the interesting bits are
in the tier breakdown and the bug that came before the run.

### The bug: Cloudflare error 1010, "banned based on browser signature"

First attempt returned HTTP 403 on *every* Groq call:

    error code: 1010

Curl to the same endpoint with the same key got HTTP 200. So the key was
fine, the endpoint was fine — the client itself was being rejected. That
error code is specifically Cloudflare's WAF flagging the caller's browser
signature (User-Agent + TLS fingerprint) as bot-like and blocking it.

The trigger: `urllib.request` sends `User-Agent: Python-urllib/3.13` by
default, which is a well-known bot signature. Adding one line —

    "User-Agent": "MSL-Bench/0.1 (+https://github.com/erika-goh/MSL-Bench)"

— fixed it. Same-run confirmation: the 12 failure records the killed run
had written got overwritten with real results as the re-run passed
through them (`run_suite.py` writes to a deterministic per-problem path).

**Root cause in one sentence:** Groq is Cloudflare-fronted; Gemini isn't.
Same client code, different network path, one silently fails. Worth
remembering — this class of "your HTTP library's UA gets bot-blocked" bug
is invisible from unit tests and only shows up when you actually hit the
upstream.

### Finding worth keeping: Groq crushes T1 but stalls at MPS on T2

| Tier | n | fast_0 | fast_1 |
|---|---|---|---|
| T1 elementwise | 12 | **100.0%** | 100.0% |
| T2 reductions  | 9  | 22.2% | **0.0%** |
| T3 tiled       | 8  | 0.0%  | 0.0% |
| T4 fused       | 7  | 0.0%  | 0.0% |

Two things stand out.

**One:** Groq is perfect on T1 (12/12), beating Gemini (11/12 — Gemini
missed p010_abs). The T1 elementwise wall is *not* a hard ceiling; it's
model-dependent, and llama-3.3-70b is on the right side of it.

**Two — and this is the honest lesson:** Groq's two T2 wins (`p101_row_sum`
at 0.4× MPS, `p103_col_sum` at 0.3× MPS) are `fast_0` correct but
`fast_1` failures. The fast_0/fast_1 split for T2 is **22.2% → 0.0%**.

Before opening the kernel I predicted Groq was writing the naive "one
thread per row" pattern — safe but leaves the GPU idle. Reading the
actual generated kernel refuted that: it's the textbook halving-stride
tree reduction, load-into-shared-memory-and-tree-reduce, structurally
identical to what a competent CUDA kernel would look like. The apparent
perf loss is on a single line:

    threadgroup_barrier(mem_flags::mem_device);

That's arguably the *wrong flag* for a threadgroup-scoped reduction —
`mem_flags::mem_threadgroup` fences over threadgroup memory only, while
`mem_device` also forces a device-memory fence on every iteration of
the reduction. `__syncthreads()` in CUDA has no flag parameter to get
wrong, so the model doesn't know the flag is load-bearing and picks the
"safer" (stronger) option. But how much does that actually cost?

**Hand-patched the flag, re-ran through the A/B/A single-problem
runner. Result:** kernel_ms went from **3.584 ms → 3.317 ms**, a
**7.4% improvement**. Real, but much smaller than the CUDA→MSL story
would predict.

    baseline (mem_device):     kernel_ms=3.584, speedup=0.427x
    patched  (mem_threadgroup): kernel_ms=3.317, speedup=0.539x

(The speedup ratio jumped 26% because MPS's reference_ms also drifted
17% between the two runs — MPS timing is noisier than kernel timing.
The raw kernel_ms is the honest number. Within-run A/B/A block delta
was 0.0000 and 0.0001 respectively — the 7.4% is signal, not noise.)

**Why isn't the fix bigger?** Best guess: Apple's compiler is smart
enough to fold part of the unnecessary device-scope fence away when
it sees no device writes in the fence region. So the "wrong flag"
costs some real perf but doesn't dominate.

The next natural question: is the *dominant* remaining gap explained
by missing SIMD-group idioms? Wrote a hand-tuned kernel to find out:
one threadgroup per row, intra-SIMD reduce via `simd_shuffle_down`
(5 shuffle steps register-to-register, zero memory), one threadgroup
barrier to publish 8 per-SIMD partial sums, then a cross-SIMD reduce
using the first SIMD group. That's 8 shuffles + 1 barrier instead of
the original tree's 8 barriers + 8 shared-memory read/write rounds.

    hand-tuned SIMD-group kernel:  kernel_ms=3.025, speedup=0.507x

Compared to Groq's baseline (3.584) that's another 8% off the kernel
time — real, but again nothing close to closing the gap to MPS (1.53
ms). Let's do the arithmetic on where the time actually goes.

**The bandwidth-bound math.** p101_row_sum reads 262144×256×4 =
268 MB from device memory (output is negligible). Effective bandwidth
achieved:

| kernel                        | ms    | effective GB/s |
|-------------------------------|-------|----------------|
| Groq original (mem_device)    | 3.584 | 74.8           |
| Groq patched (mem_threadgroup)| 3.317 | 80.9           |
| Hand-tuned SIMD-group         | 3.025 | 88.7           |
| MPS reference                 | 1.53  | ~175           |

MPS is achieving **~2× the effective memory bandwidth** of my
hand-written kernel. At K=256 with random floats and no meaningful
compute per element, this problem is memory-bound. All three
hand-written kernels are leaving half the memory bandwidth on the
table, and the barrier-flag and SIMD-group choices only move the
needle within that half.

MPS is almost certainly doing at least some of:
- **Vectorized loads** (`float4` / `float2`) — one memory transaction
  returns 4 elements instead of 1, so the number of device-memory
  ops drops by 4×.
- **Multiple rows per threadgroup** — one 512- or 1024-thread group
  processes 2 or 4 rows at once, improving cache line utilization
  and amortizing launch overhead.
- **Larger threadgroups** to hide memory latency by keeping more
  requests in flight.

Refined lesson — the LLM-vs-MPS gap on reductions has three layers,
and the sizes are the opposite of what I would have guessed:

1. **Barrier-flag confusion** (7%): measurable, small — CUDA→MSL
   syntax transfer failure. Fixable via one-token repair feedback.
2. **Missing SIMD-group idioms** (~8%): also measurable, also small —
   MSL-specific `simd_shuffle_down` intrinsic with no CUDA 1:1 analogue.
   Fixable via idiom-level repair feedback.
3. **Remaining gap under spec's launch** (~50%): the big remainder,
   but the diagnosis is subtler than "design mistake by the model."
   See next section.

To disentangle, wrote a float4-load variant under the spec's fixed
launch (256 threads/group, 64 active + 192 idle, each active thread
loads a `float4`). Expected: fewer memory transactions → real perf
win. Result:

    float4 loads (64 active threads):  kernel_ms=2.961, 90.5 GB/s
    hand SIMD (256 threads, scalar):   kernel_ms=3.025, 88.7 GB/s

Basically null. That 2% is inside timing noise. **Apple's memory
subsystem already coalesces adjacent thread scalar loads into
cache-line-sized transactions** — 256 threads × 1 float from
consecutive addresses is the same wire pattern as 64 threads ×
1 float4. Transaction count is not available as a knob under this
launch.

All four candidate kernels (Groq original, Groq patched, hand SIMD,
hand float4) plateau at ~90 GB/s. MPS at 175 GB/s. **The ~90 GB/s
number looks like the ceiling under the spec's fixed launch config
(`one threadgroup per row, 256 threads per group`).** MPS is not
bound by that spec — it picks its own launch — and it uses the
freedom to run at ~2× the throughput.

So the honest breakdown of layer 3 is not "50% design mistake by the
model." It's:

3a. **Cost under the spec's launch that the model *could* recover** —
    small, dominated by layers 1 and 2 above (~15% combined).
3b. **Cost inherent to the spec's launch** — the rest, ~35%. No LLM
    output can close this half without violating the launch config
    the problem hands it.

That's a real finding about the benchmark itself, not about model
quality. Layer 3b caps every kernel on p101 at ~0.5× MPS regardless
of quality. If the point of the benchmark is to measure model kernel
design, this cap is a bug: the "correct" answer looks like a failure
because the benchmark's launch constraint prevents MPS-shaped designs.

Options for the benchmark author to think about:
- Let the candidate propose its own launch config (spec change).
- Change the reference to a hand-tuned kernel that runs under the
  *same* launch config — measure LLM-vs-optimal-under-constraint
  rather than LLM-vs-unconstrained-MPS.
- Accept the cap and interpret speedup as "% of the achievable
  bandwidth under this launch" rather than "% of MPS."

Refined Phase-5 framing: the interesting question isn't "can the
model learn MSL syntax" (layer 1) or "can it learn Apple idioms"
(layer 2). It's "**how much of the benchmark's speedup-vs-MPS metric
is even reachable given the spec's launch constraint**" — which is
a question about the benchmark, not about the model.

- Gemini: reaches for the right pattern, gets the MSL syntax wrong
  at the *declaration* level (won't compile). Layer-1 failure.
- Groq: gets the right pattern to compile and produce correct
  output, but stops at "port from CUDA." Layer-1 + layer-2 costs,
  and skips layer-3 entirely.

- Gemini: reaches for the right pattern (threadgroup memory, atomics),
  gets the MSL syntax wrong at the *declaration* level (won't compile).
- Groq: gets far enough that it compiles and produces the right numbers,
  but picks the wrong barrier flag and pays for it every reduction step.

The benchmark is separating these two failure modes cleanly. Correct-but-
slow is a real signal, and it points at exactly the kind of one-token
fix a repair loop should be able to make.

### Finding worth keeping: T3/T4 attempts, verify-vs-compile mix

Groq is 0/8 on T3 and 0/7 on T4 — no wins. But the failure *modes* differ:

- Compile fails: 10 (mostly T3 matmul/conv, and the top of T4).
- Verify fails: 5 (`p204_matmul_double_buffered_backfires`,
  `p208_conv2d_5x5_tiled`, `p302_fused_linear_relu`, `p303_attention_head`,
  `p304_attention_large`).

Five kernels that compile AND run to completion but produce wrong numbers
is not the same signal as five that don't compile. It means Groq is
attempting the full structure — allocating threadgroup memory, computing
tile indices, calling `threadgroup_barrier` — and getting close enough
that the compiler accepts it. Off-by-one on tile bounds, wrong stride in
a load, forgotten transpose — these are the *productive* failures. They're
exactly what a repair loop should be able to fix, because the feedback
signal ("your output differs from reference by ~X") is more actionable
than "your file didn't compile."

That maps directly to the Phase 5 flywheel plan: verify-failures are
higher-value seed trajectories for a repair fine-tune than compile-failures
are. And Groq — the *worse* overall model of the two — is producing more
of them, because it's more willing to actually attempt the hard problems.

### Smaller observations

- **Free-tier ergonomics:** Groq's free tier ate all 36 problems in one
  sweep with no rate-limit or quota interruptions. Gemini's daily quota
  killed us at 19. This makes Groq the better *daily driver* for iteration
  even though its raw accuracy is lower.
- **fast_2 parity:** Both models hit fast_2 exactly once — p006_axpby's
  fused MAD gets a 2.5× on both. Not a model-dependent win; it's just that
  MPS's `torch.mul + torch.add` has enough dispatch overhead at 1M elements
  that a single fused kernel beats it. Real 2× speedups are rare on this
  benchmark.
- **The overwrite-on-rerun property saved cleanup.** Because
  `run_suite.py` writes to `results/raw/{run_tag}__{pid}.json`, the 12
  bogus 403 records from the failed first attempt got silently replaced
  by real records once the re-run passed through those problems. Nothing
  to delete by hand. Worth remembering — write-path stability is a
  disaster-recovery feature, not just a naming convention.

### Decisions made

- `USER_AGENT` string in `mkb/llm/providers.py` set to
  `MSL-Bench/0.1 (+https://github.com/erika-goh/MSL-Bench)`. Applies to
  every provider, not just Groq — Gemini and Ollama don't care, but the
  default is now sane for any Cloudflare-fronted future provider.
- Groq's baseline model for the leaderboard is `llama-3.3-70b-versatile`
  (README default). Groq also offers `moonshotai/kimi-k2-instruct` and
  `qwen/qwen3-32b`, which are plausibly stronger on code — worth a
  follow-up run, but not this session.

### Open items (carry into next session)

- **Decide how to handle the spec-launch ceiling.** Layer 3b caps
  p101 at ~0.5× MPS regardless of kernel quality. Three plausible
  fixes: (a) let candidates propose launch configs; (b) time against
  a same-launch reference kernel instead of MPS; (c) accept the cap
  and change the metric wording. Each has different implications
  for what the benchmark measures. Worth deliberate design choice
  before Phase 4.
- **Audit other T2/T3/T4 specs for the same constraint.** If p101's
  launch is representative, several other problems may have baked-in
  ceilings the metric hides. `p102_row_max`, `p104_row_softmax`, and
  the matmuls in T3 all have hardcoded launch configs — worth
  checking whether they leave room for MPS-shaped designs.
- **Promote `/tmp/mkb_barrier_test/row_sum_simdgroup.metal` to a
  reference kernel** (`tests/golden_kernels/row_sum.metal` or a new
  dir like `tests/reference_kernels/`) if we adopt option (b) above.
  It's a solid same-launch reference at 88.7 GB/s.
- **Verify-fail kernels on T3/T4 are potential repair-loop seeds.** Five
  of them exist now. Manually diffing one against a golden could tell us
  whether they're off-by-one/stride-flip class errors (repair-friendly) or
  fundamentally-wrong-algorithm class (repair-hostile).
- **Try `moonshotai/kimi-k2-instruct` on Groq** for a second data point in
  the same session-per-day slot. If it clears T2 at fast_1, that's a
  provider-independent signal about model quality on MSL.

---

## 2026-07-03 — First real Phase-3 leaderboard row: Gemini 2.5 Flash, one_shot, 19 problems

Milestone day. The first ever LLM-vs-MPS leaderboard row for MSL-Bench
landed — 19 problems evaluated end-to-end, tier gradient visible in the
numbers, and two of the "infra will pay off later" patches from last
session earned their keep within their first hour of use.

### What got built

Two small resilience patches surfaced by the run itself, then the run:

| commit | what |
|---|---|
| (this session) | `_post_json` retries 5xx as well as 429 — one-line broadening of the retry condition. |
| (this session) | `run_suite.py` wraps the per-problem body in try/except, records `fail_stage="provider_error"` and continues. `make_leaderboard.py` learned the new `e` glyph. |
| (this session) | Ran `gemini-2.5-flash` one_shot across (up to) 33 remaining problems. Got 19 clean records before Google's daily quota killed the connection. |

The two patches both fired *during their first run*: an HTTP 503 hit on
p004 after 3 successes (would have terminated the whole suite without
patch #1), and two HTTP 429s on p108/p109 after quota exhaustion (would
have terminated without patch #2). Without either fix we'd have ended up
with 3-8 records instead of 21. Cheap infra that paid off same-session.

### Finding worth keeping: the T1→T2 cliff is real and clean

| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| T1 elementwise | 12 | **91.7%** | 91.7% | 8.3% |
| T2 reductions  | 7  | **14.3%** | 14.3% | 0.0% |

92% → 14% correctness in one tier boundary. The KernelBench thesis
("LLMs can do elementwise, break on cross-thread coordination")
reproducing cleanly on the Metal side. Only p104_row_softmax passed T2;
every other T2 attempt failed at compile (5×) or emitted no code (1×).

The cliff isn't about difficulty per se — it's about whether the model
has to know Metal-specific constructs. T1 kernels are one-line
translations of C math (`out[i] = f(x[i])`). T2 kernels *require*
threadgroup memory, barriers, or atomics — all of which have MSL-specific
names and idioms that CUDA training data doesn't cover directly. That's
the entire benchmark thesis, and now there's a headline number for it.

### Finding worth keeping: fast_1 == fast_0 across the board

Every single problem the model got correct also matched-or-beat MPS
(fast_1 = fast_0 exactly, both tiers). Meanwhile fast_2 = 5.3% —
essentially zero. So the picture is:

- MPS's per-dispatch overhead is real enough at these problem sizes
  that *any* hand-written kernel that does the actual math is ≥ MPS.
- But *doubling* MPS requires either fusion (p006 axpby's fused MAD,
  the sole 2.5× outlier) or a genuine algorithmic win — neither of
  which the model is producing in one shot.

Sharpens the article thesis nicely: **it's not that LLMs write slow
kernels; it's that they write correct-but-generic kernels that don't
exploit anything MPS isn't already doing.** The gap between "compiled &
correct" and "fast" is where the interesting engineering lives, and
that's exactly where one_shot LLMs sit.

### Finding worth keeping: p010_abs — C vs. C++ math library naming

p010_abs was the *only* T1 failure. Trivial elementwise:
`out[i] = fabsf(x[i])`. Compile error. Why?

Metal's `metal_stdlib` is **C++-based**, so it exposes C++-style
overloaded names: `abs()`, `fabs()`, `exp()`, `sqrt()`. What it does *not*
have is the plain-C `<math.h>` type-suffixed forms `fabsf`, `expf`,
`sqrtf`. Gemini reached for `fabsf` — the CUDA/C bias in its training
data — and the compiler rejected it. This is the **same failure class**
as last session's p013_gelu finding (`erf` missing from metal_stdlib).

Concept worth explaining: **Metal's stdlib is a subset of C++, not C.**
When translating from CUDA (which historically permitted both C and C++
math names) or from CPU code, models pattern-match on the C names and
lose. Every future "why did this trivial kernel fail" investigation
should check math-function naming first.

### Finding worth keeping: no_code responses are a real category

p106_col_sum_atomic came back with **no parseable ```metal fence** —
Gemini emitted prose or a differently-fenced code block that
`extract_metal` couldn't recognize. Never happened in the ollama smoke
test (qwen always fenced). Small (1/19) but non-zero, and the
one_shot mode has no recovery from it. `repair@k` mode does — the first
retry prompt explicitly asks for the code fence.

### Decisions made

- **Kept the 6 stale ollama records in `results/raw/`** — they're honest
  data from an earlier smoke test, and having a second run enables the
  leaderboard's "unbeaten problems" callout (which needs n_runs ≥ 2).
  They land in the tables as a second column and don't pollute anything.
- **Deleted the 2 `provider_error` records (p108, p109) before generating
  the leaderboard** — a 429 wall from our free-tier quota is not a
  measurement of the model's kernel-writing ability, and including them
  would depress the T2 pass rate by two artificial failures. Records are
  gitignored so this is a local-only cleanup, but the principle matters:
  **provider_error is missing data, not a failed measurement.** Same
  reason you'd exclude a benchmark iteration where the wall clock
  jumped due to a laptop sleep.
- **Chose `gemini-2.5-flash` over `2.0-flash` for the run** — 2.0 was
  already quota-saturated on the first preflight ping (45s of 429 retry
  before giving up). 2.5 had fresh capacity AND is the newer, stronger
  model, so it's the honest choice for a first leaderboard row anyway.
- **Stopped the run at the second consecutive 429** instead of letting
  it grind through the remaining ~15 problems at 45s of retry-and-fail
  each. Rate-limit walls compound.

### Open items (carry into next session)

- **~17 problems still unmeasured** for gemini-2.5-flash: p108/p109
  (T2), all 8 of T3, all 7 of T4. Waiting on quota reset OR a
  `GROQ_API_KEY`. When quota resets, resume with `--only` on those.
- **`provider_error` handling in `make_leaderboard.py`** — right now we
  hand-delete those records to keep the leaderboard honest. Could instead
  add a filter in `fast_p` / `tier_table` that excludes them from the
  denominator. Not urgent, but the manual cleanup is a papercut.
- **`repair@k` mode not yet exercised at all.** The whole point of the
  Phase-3 design is comparing one_shot vs repair@5. Only one_shot has
  data. The 8 T2 compile failures are exactly the class of error where
  a compile-diagnostic-in-the-loop repair prompt could recover; that
  comparison is the meat of the article and it hasn't been measured.
- **`extract_metal` couldn't parse Gemini's p106 response.** Worth
  looking at what fence it used — might be a fence style (`` ```C++ ``
  perhaps) that we could add to the regex, cheaply widening the "no_code"
  moat for the whole benchmark. Or it might be genuine prose (no code)
  in which case there's nothing to do.
- **One_shot records don't preserve the compile-error text** — the
  `feedback` string exists in `evaluate_kernel`'s return but never lands
  in the saved record. For T2 failure analysis (what compile errors did
  Gemini produce?) we'd have to re-run and log, or add the field. Small
  ergonomic gap.
- **Prompt hint policy still holding** — no hints about `threadgroup`,
  `abs` vs `fabsf`, or math naming. The p010 and p106 findings are the
  proof that hints would leak signal. Reaffirmed.

### Catalog state

Unchanged — 36 problems total. 19 now have a gemini-2.5-flash one_shot
data point; the other 17 pending quota reset. Leaderboard file
regenerated at `results/tables/leaderboard.md`.

---

## 2026-07-02 — Phase-3 scaffolding session: no new kernels, four infra commits

First session where thermal caution was the explicit framing from the
start, so no new benchmark problems shipped. Instead: filled in the
last CPU-only gaps between "we have 36 problems + LLM providers wired"
and "we can produce a defensible first leaderboard." Four small commits.

### What got built

| commit | what |
|---|---|
| `150d83a` | `--only p001,p013,...` flag on `run_suite.py` (targeted LLM smoke tests) |
| `34b7354` | `extract_metal` regex accepts ` ```c` and ` ```objc` fences + 9 regression tests |
| `7e920c0` | `scripts/preflight.py` — pings each provider, skips (not fails) unset creds |
| `045e5f3` | Per-problem × per-run failure-stage table in `make_leaderboard.py`, plus an "unbeaten problems" callout that engages at n_runs ≥ 2 |

Test count went 24 → 33. All pure-Python, no Mac needed.

### Finding worth keeping: the 6-problem ollama smoke test is legit signal

`results/raw/` had 6 stale records from a prior local ollama run —
qwen2.5-coder:14b, one_shot, one problem per tier. **0/6 correct**,
5 compile fails + 1 verify fail. Before designing a bigger run I
inspected the transcripts to make sure it wasn't a prompt bug.

Two failure modes worth remembering:

- **p001 vector_add** (compile fail): qwen wrote
  `uint3 gid [[thread_position_in_grid]]` for a **1D** dispatch, then
  compared `gid < 33554432` (uint3 vs scalar) and indexed `a[gid]`.
  Doesn't compile — `uint3` isn't implicitly comparable-to or
  indexable-with a scalar. The right answer is `uint gid` for a 1D
  grid. Model didn't infer dimensionality from the 1D input shape.

- **p201 matmul_tiled** (verify fail — compiled fine): qwen wrote
  `thread float tileA[TILE][TILE]` — this is **per-thread** register
  memory in Metal, not shared. CUDA's `__shared__` should map to
  Metal's `threadgroup` address space, but the model wrote `thread`
  instead. So each thread had its own private tile, cooperative loads
  wrote into other threads' unreachable memory, and the dot product
  read garbage. Kernel silently produces wrong answers.

That second one is **exactly** the class of mistake this benchmark
exists to surface. The whole thesis of "measures Metal translation
competence, not algorithm discovery" needs data like this to defend,
and now we have some. Also validates the prompt-hint policy (below):
if we'd told the model "use `threadgroup` for shared tiles" this
signal would have vanished.

### Finding worth keeping: ollama-14b latency floor is ~7s per call

Preflight measured a **one-word** ollama round-trip at 7.05s cold.
Real kernel generations produce full files with 30–100 lines of MSL.
Rough extrapolation: 30–60s per call. So:

| run shape | ballpark wall time |
|---|---|
| ollama one_shot × 36 | 20–40 min |
| ollama repair@5 × 36 | 60–180 min worst case |
| groq / gemini one_shot × 36 | ~15 min (network + free-tier rate limits) |
| groq / gemini repair@5 × 36 | 30–90 min |

The ollama numbers are the thermally-expensive ones — sustained local
14b generation warms the laptop noticeably. Groq/gemini are basically
network sleeps + a few hundred ms of local kernel bench GPU per call.
Point being: **for the first leaderboard row, use groq or gemini**,
not ollama. Only bring ollama back in when we have a chunk of time
where laptop heat is acceptable.

### Decisions made

- **Prompt stays Metal-agnostic** — no `threadgroup` hint, no
  dimensionality hint. The p201 finding above is the proof that
  hints would leak signal. A "hinted" arm can be added later as a
  separate mode if the article needs it, but v1 measures raw
  translation competence.
- **Preflight skips unset creds, fails only on genuine errors** —
  matters because the user often has one provider set up at a time
  during development, and a preflight that hard-fails when
  `GEMINI_API_KEY` is missing would just be noise.
- **Leaderboard's "unbeaten problems" callout requires n_runs ≥ 2** —
  with one run the pass column already tells you which problems are
  unbeaten, so the callout would be pure duplication.
- **Four separate commits, not one** — matches the repo's small-commit
  style. Each has an independent explanation of what it changes and
  why, which is what future-me (or a reader) wants.
- **Chose analysis-first ordering** (step 7 before steps 3-6) once
  step 3 was blocked on missing API keys. Now when the keys land,
  every downstream reporting change is already in place — the run
  produces both the raw JSON and immediately-useful diagnostics.

### Open items (carry into next session)

- **API keys not set** — `GROQ_API_KEY` and `GEMINI_API_KEY` need to
  be exported before step 3. Both providers offer free tiers, ~2 min
  signup each (`console.groq.com`, `aistudio.google.com`).
- **`make test` requires venv activation** — the Makefile calls plain
  `pytest`, but pytest lives in `.venv/bin/pytest`. Pre-existing,
  worked around by activating before `make test`. Cheap fix: change
  target to `python3 -m pytest ...` inside a `. .venv/bin/activate &&`
  prefix, or document the venv step.
- **Ollama full run needs a dedicated thermal budget** — 1–3 hours of
  sustained load. Not tonight.
- **Prompt anti-example section** — tempting to add "don't do X" hints
  based on the two failures above, but that violates the agnostic-
  prompt decision. Revisit only if step-3 groq run shows systematic
  same-class failures — otherwise the mistakes ARE the finding.

### Catalog state

Unchanged — 36 problems total (12 T1 + 9 T2 + 8 T3 + 7 T4).
Phase-3 infra is now feature-complete for a groq/gemini leaderboard
row. Zero laptop heat this session.

---

## 2026-06-30 — p013 gelu (tanh approx): erf isn't in metal_stdlib

Quick Tier 1 close-out after the p208 thermal stop. Wanted a
thermally-light problem; got both that AND an interesting Metal-
specific finding worth documenting.

### What got built

`tier1_elementwise/p013_gelu/` — element-wise GELU at length 2^25
using the tanh approximation:

  gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

Reference matches via `F.gelu(x, approximate='tanh')`.

### Result

| metric | value |
|---|---|
| compiled / correct | ✓ / ✓ |
| max_abs_err | 4.77e-7 (1e-5 tolerance) |
| kernel_ms | 1.42 |
| reference_ms | 1.68 |
| speedup | **1.18×** |
| block_delta_frac | 0.71% (trustworthy) |

Modest specialization premium as predicted — both kernels are
bandwidth-bound single-dispatch ops, MPS isn't paying any
multi-dispatch tax. The arithmetic-intensity rule from p206/p207
checks out here: bandwidth-bound + already-fused MPS reference =
parity-ish speedup.

That this came back trustworthy despite the post-p208 thermal
state is itself interesting — elementwise ops are short enough
(1.4 ms) that they finish before the throttle kicks in mid-block.
The thermal damage was concentrated on the long-running tiled
problems (p208's compute filled a much bigger work queue).

### Finding worth keeping: `erf` is missing from metal_stdlib

First draft used the exact form: `0.5 * x * (1 + erf(x / sqrt(2)))`.
Compile error: `use of undeclared identifier 'erf'`. Apple's
metal_stdlib omits `erf` and `erfc` even though it includes most
other math functions (exp, log, tanh, atan, sqrt, rsqrt, etc.).
CUDA has erf as a built-in; Metal doesn't.

This matters for LLM eval. A model trained on CUDA conventions
will plausibly emit `erf(x / sqrt(2))` for exact GELU and the
kernel will fail to compile. **A correct port to Metal must either
switch to the tanh approximation or polyfill erf via a polynomial
approximation** — two valid responses, but the model has to know
to do one of them.

This pattern (Metal omitting a function CUDA has) is worth
hunting for elsewhere as the catalog grows. Likely other examples:
the `__half`/`__bfloat16` precision aliases, some
`__shfl_xor_sync` variants, atomic float ops at warp/SIMD scope.

### Decisions made

- Used tanh approximation rather than polyfilling erf — simpler,
  matches PyTorch's `approximate='tanh'` cleanly, and the
  approximation gap is well under the tolerance budget.
- Did NOT bump the size beyond 2^25 — matches p011 exp's length
  for direct comparability, and is enough to amortize dispatch
  overhead.
- Did NOT chase the `erf` polyfill as a separate problem; it'd
  be a Tier 1 curiosity at best. A polynomial-erf could become
  part of a future Tier 4 fused statistic problem if useful.

### Catalog state

| tier | count |
|---|---|
| 1 elementwise | **12** |
| 2 reductions | 9 |
| 3 tiled | 8 |
| 4 fused | 7 |
| **total** | **36** |

Nine problems shipped today. Strongly recommending an actual
session pause now — GPU thermal state is intermittent, and
diminishing returns on further problems in this session vs a
fresh start.

---

## 2026-06-30 — p208 conv2d_5x5_tiled: halo pattern lands, but GPU finally throttled hard

Eighth problem of the session. The halo memory pattern is the
last canonical CUDA stencil-optimization missing from the catalog
— now in. **Correctness verified, but timing is unreliable due
to thermal throttling.** Catalog at 35.

### What got built

`tier3_tiled/p208_conv2d_5x5_tiled/` — 5×5 valid convolution at
input (1028, 1028) → output (1024, 1024), with the canonical
tile-with-halo input staging pattern.

- Each TG covers a 16×16 output block.
- That output block depends on a 20×20 input region (the output
  footprint + 2 halo pixels per side).
- 256 threads cooperatively load all 400 input pixels into TG
  memory via a single flat strided loop (most threads load 1
  pixel; 144 of them load 2).
- One barrier, then every output thread computes its 5×5 dot
  product reading exclusively from TG memory. Zero redundant
  device reads.

### Result — correctness yes, perf measurement no

| metric | value |
|---|---|
| compiled / correct | ✓ / ✓ (no warnings) |
| max_abs_err | **0.0** (bit-exact!) |
| timing_trustworthy | **false** (thermal throttling) |
| block_1_median_ms | 0.080 (suggestive of strong perf) |
| block_3_median_ms | 0.250 (3.1× slower — clear throttle) |
| block_delta_frac | 212.7% (vs 7% threshold) |
| stability_error | "thermally throttling, on battery, or under competing GPU load; let it cool, plug in power, retry" |

### Bit-exact result is the most interesting finding

For a 25-element fp32 dot product against `F.conv2d`, I expected
~1e-5 absolute error from accumulation-order differences (this
was the case in p207 K=3 conv, which had max_abs_err 2.86e-6).
Here max_abs_err is literally 0.0 across the (1024, 1024) output.

That implies our row-major accumulation order matches MPS's exactly.
**MPS's conv2d for this size is also a direct convolution** (not
im2col-then-matmul, not Winograd) **using the same kernel
traversal order we used.** That's a small finding about MPS's
internal dispatch heuristics worth keeping.

For larger kernels, MPS would presumably switch to FFT-based or
Winograd convolution, where the algorithm change would shift
the rounding profile and bit-exactness would disappear.

### The thermal-state story now needs a proper writeup

Today we've hit the harness's three trustworthiness states
naturally, all of them legitimately:

1. **Trustworthy** (block_delta < 7%): the common case after
   warmup, used for headline numbers.
2. **Cold-ramp-untrustworthy** (block 3 FASTER than block 1):
   first run after a process restart or idle period.
3. **Thermal-throttle-untrustworthy** (block 3 SLOWER than block
   1): after sustained GPU load. Now observed.

For p208 the right move is to NOT retry — pushing through means
producing numbers that mostly measure the throttle state. The
correctness verification is what's load-bearing for catalog
inclusion; the timing comparison can be retaken in a fresh
session.

**Carrying forward as a Phase 4 report concern:** the
benchmark needs a documented "session protocol" — start cool,
verify with a short ramp, measure each problem ≤ N times, stop
when block_delta exceeds X. Without one, "speedup" numbers in
the catalog represent a mix of cool/warm/throttled states.

### Open: does the halo pattern actually help on Apple Silicon?

The empirical question this problem was designed to answer is
unanswered. The clean comparison would be a side-by-side of:
- p207 conv2d_3x3 (naïve, 6.24× MPS warm)
- p208 conv2d_5x5_tiled (halo, ?? MPS warm)
- A hypothetical p209 conv2d_5x5_naïve (no halo, ?? MPS warm)

Without p209 or a trustworthy p208 reading, we can't say
whether the halo pattern outperforms naïve L1 caching at K=5.
**Block 1's 0.080 ms is suggestive that p208 is competitive
with p207** (which makes sense — naïve K=5 would read 25 pixels
per output vs p207's 9, so even with perfect L1 we'd expect K=5
to be slower — and yet block 1 is in p207's neighborhood).
But "suggestive" isn't trustworthy enough for the catalog
headline.

### Decisions made

- Used a single flat strided loop for the cooperative load
  instead of the more common "boundary-thread loads extra
  halo pixels" pattern. Cleaner code; same total work.
- Did NOT build a p209 naïve K=5 to directly measure the halo
  benefit. Would need a fresh thermal state to measure either
  trustworthily.
- Kept K=5 instead of going larger (K=7 or K=9). 5 is enough
  to introduce the halo concept; larger Ks just scale the work
  without teaching anything new.
- Did NOT retry the timing run. The trustworthy reading we
  *could* get would just be the post-throttle steady-state,
  which is contaminated by today's session, not the kernel's
  inherent perf.

### Catalog state

| tier | count |
|---|---|
| 1 elementwise | 11 |
| 2 reductions | 9 |
| 3 tiled | **8** |
| 4 fused | 7 |
| **total** | **35** |

Now 5–10 problems short of the 40–45 target. The biggest
remaining strategic question is whether to:
- Push another 5+ problems to reach 40 (sustained Phase 2 grind)
- Pause Phase 2, run a Phase 3 LLM eval on the current 35-problem
  catalog as a milestone check (would tell us what techniques
  models can/can't write, informing what to build next)

The session's thermal state argues for pausing in any case.

---

## 2026-06-30 — p307 rmsnorm: Tier 4 breaks attention monoculture, 8.7× MPS

Tier 4 was 4-of-6 attention. p307 RMSNorm is the canonical
modern-LLM normalization (LLaMA, Mistral, Gemma) and breaks that
monoculture cleanly. Catalog at 34, Tier 4 at 7.

### What got built

`tier4_fused/p307_rmsnorm/` — fused row-wise RMSNorm:

  rms[m]   = sqrt(mean_j(x[m, j]²) + eps)
  y[m, j]  = x[m, j] * g[j] / rms[m]

Shape: x is (1024, 4096), g is (4096,), y is (1024, 4096).
Three phases inside one TG-per-row kernel: strip-mined
sum-of-squares, two-stage reduction with simd_sum, then a
normalized+scaled writeback fused with the g multiply.

### Result — biggest Tier 4 speedup, eclipses attention

| metric | value |
|---|---|
| compiled / correct | ✓ / ✓ |
| max_abs_err | 1.91e-6 (1e-3 tolerance) |
| kernel_ms (warm, trustworthy) | 0.220 (lower bound; cleanup makes ≥ this fast) |
| reference_ms (torch RMSNorm via mean/sqrt/div/mul) | 1.908 |
| **speedup** | **8.68×** (lower bound) |
| block_delta_frac (pre-cleanup trustworthy reading) | 0.2% |

This is the **biggest single-problem speedup in the project**,
beating p303 attention_head's 8.0× and p207 conv2d's 6.24×.

### Why so big

Same general-purpose tax story as the Tier 3 wins, with extra
sauce: the PyTorch reference for RMSNorm via `torch.mean(x**2)`
+ `torch.sqrt` + `torch.div` + `torch.mul` is **4 separate MPS
dispatches**, each paying its own launch overhead (~80–100 µs).
That's ~400 µs of pure dispatch tax, comparable to the
1.9 ms reference time itself.

Our fused kernel does the whole thing in one dispatch, with the
reduction baked in via simd_sum and the inv_rms broadcast
implicit (every thread computes it from the published SG
partials).

This is the cleanest fusion-thesis data point in the project:
**one fused kernel vs four dispatches → ~8× speedup at this
shape, dominated by dispatch overhead.** Production RMSNorm
implementations (Apple's own MPSGraph, or torch.compile) would
fuse this too; the reference is the "naive PyTorch eager mode"
path that's exactly what a user writing model code would hit.

### Bug class encountered: uninitialized TG broadcast cell

First draft used a TG broadcast pattern:

    threadgroup float inv_rms_bcast;
    if (sg_in_tg == 0 && lane == 0) {
        inv_rms_bcast = rsqrt(...);
    }
    threadgroup_barrier(...);
    float inv_rms = inv_rms_bcast;

The threadgroup_barrier guarantees the read sees the write, but
**the Metal compiler can't prove that** from the if-condition
structure alone and emits a "used uninitialized" warning. The
warning is a false positive at runtime but a real signal: any
LLM-generated kernel that uses this pattern will also warn, and
the warning becomes hard to distinguish from a real bug.

**The fix** (which also happened to be faster): drop the
broadcast cell entirely. After the 8 SG partials are published,
every thread independently sums them and computes its own
inv_rms. Same value in every thread (identical input, identical
arithmetic), no race, no second barrier, no warning. The
compiler is happy and the kernel is structurally simpler.

This pattern — *let every thread redundantly compute a scalar
rather than broadcast it through TG memory* — is worth
remembering. The compute is essentially free (8 floats summed +
one rsqrt), the broadcast costs a barrier and a TG load, AND
removes a class of subtle warnings.

### Concept: rsqrt as a first-class intrinsic

`rsqrt(x) = 1 / sqrt(x)` is one Metal instruction, faster and
more accurate than `1.0f / sqrt(x)` when you actually want the
reciprocal. Anywhere a kernel computes `divide by length` or
`divide by sigma`, prefer `value * rsqrt(...)` over
`value / sqrt(...)`. The torch reference does the divide
explicitly, but our kernel uses rsqrt — a small piece of
the speed advantage.

### Decisions made

- Did NOT include a separate optional-bias parameter even though
  RMSNorm in some formulations has one. LLaMA's RMSNorm doesn't.
  Keeps the spec simple.
- Used `rsqrt` instead of `1.0 / sqrt` everywhere — cleaner and
  faster on Apple GPU.
- After the warning fix, used the "every thread independently
  computes inv_rms" pattern. The cost is 8 redundant additions
  per thread; the savings is a barrier and a TG load. Strict win.
- Did NOT re-benchmark after the cleanup (thermal-conservative).
  The headline number (8.68×) is the pre-cleanup trustworthy
  reading; the post-cleanup kernel does strictly less work so
  8.68× is a lower bound.

### Tier 4 progress

Tier 4 now has *something other than attention*: a fused norm,
a fused linear+relu, a fused layernorm, AND four attention
variants. Six of seven Tier 4 problems are now distinct
"composite operation" patterns. The catalog's Tier 4 coverage
is much more honest now.

### Catalog state

| tier | count |
|---|---|
| 1 elementwise | 11 |
| 2 reductions | 9 |
| 3 tiled | 7 |
| 4 fused | **7** |
| **total** | **34** |

Closer to the 40-45 target. Remaining strategic gaps: a
tile-with-halo conv variant in Tier 3 (would teach a memory
pattern still missing), and possibly a GELU-fused or fused
attention-bias in Tier 4 for one more non-attention data point.

---

## 2026-06-30 — p109 row_prefix_sum: parallel scan via SIMD-group intrinsics, 2.7× MPS

Diversified out of Tier 3 after three consecutive problems there.
Parallel scan is a fundamental primitive that was completely
absent from the catalog — every CUDA tutorial covers it. Now in
Tier 2. Catalog at 33 problems.

### What got built

`tier2_reductions/p109_row_prefix_sum/` — row-wise inclusive
prefix sum (cumsum) on a (1024, 4096) matrix. One TG per row,
256 threads each, three-stage parallel scan.

### Result

| metric | value |
|---|---|
| compiled / correct | ✓ / ✓ (first try, no warnings) |
| max_abs_err | 4.58e-5 (1e-2 tolerance) |
| kernel_ms | 0.163 |
| reference_ms (torch.cumsum) | 0.434 |
| **speedup** | **2.67×** |
| block_delta_frac | 6.9% (just under 7% trustworthy threshold) |

Less dramatic than the Tier 3 wins (which were 2–6× on
arithmetic-intensity-low memory-movement problems), but solid
for a problem with real cross-thread coordination — three
stages, two barriers, and SIMD intrinsics that don't have a
naive scalar fallback.

### Concept: SIMD-group scan intrinsics (first appearance)

Apple GPUs expose lane-level prefix sums in addition to the
reductions seen in p206:

  * `simd_prefix_inclusive_sum(v)` → in lane k of the SIMD-group,
    returns the sum of `v` from lanes 0..k inclusive.
  * `simd_prefix_exclusive_sum(v)` → same but lanes 0..k-1
    (lane 0 returns 0).
  * `simd_broadcast(v, src_lane)` → returns `v` from src_lane in
    every lane.

These are the Metal equivalent of NVIDIA's `__shfl_up_sync`
patterns. They run on the SIMD-group's lane interconnect — no TG
memory, no barriers, single instruction per call.

For this problem we only needed `simd_prefix_inclusive_sum`;
the EXCLUSIVE scan was synthesized as `inclusive(v) - v`. That
keeps the kernel portable to older Metal versions that may not
expose the dedicated exclusive intrinsic.

### Algorithm: three-stage parallel scan

Standard Blelloch-style scan adapted to the (32-lane SG × 8 SGs
× 16-element-strips-per-thread) hierarchy:

1. **Per-thread sequential scan** of 16 contiguous elements.
   Each thread keeps an array of running prefixes and a single
   thread_total.
2. **Within-SG scan** via `simd_prefix_inclusive_sum` on the
   thread_totals. Each thread learns its SG-relative exclusive
   prefix in one instruction.
3. **Cross-SG scan** via one SG (SG 0) loading the 8 SG totals
   from TG memory, running the same intrinsic, and writing back
   the exclusive scan.

Each thread then adds (its SG's prefix + its within-SG exclusive
prefix) to its 16 local results.

Total cross-thread coordination depth: 2 barriers. Critical-path
latency: O(log N) — much less than torch.cumsum's sequential
O(N) on CPU/GPU. The 2.67× speedup is the practical realization
of that asymptotic advantage at N=4096.

### Why we don't win bigger

torch.cumsum on MPS is *not* fully sequential — it almost
certainly uses a parallel scan algorithm internally. But MPS's
implementation pays the usual general-purpose tax (batched
multi-row dispatch, multiple dtype paths, etc.) and ours is
specialized to the exact shape and the simdgroup hierarchy. The
gap is meaningful (2.67×) but smaller than the Tier 3
memory-movement problems because the algorithm itself is
non-trivial — there's less "specialization daylight" between a
tuned scan and a naive scan than between a tuned and naive
memory copy.

### LLM-eval implications

Three risk areas for LLM coders on this problem:

1. **Knowing the intrinsic name.** Metal's
   `simd_prefix_inclusive_sum` vs CUDA's `__shfl_up_sync`-based
   handrolled scan. A model trained heavily on CUDA may write a
   handrolled NVIDIA-style scan that doesn't compile in Metal.
2. **Handling the cross-SG step.** Trees of barriers vs the
   two-stage gather-and-scan approach used here. A naive port
   may use threadgroup memory for the within-SG step too, which
   is correct but slow.
3. **The exclusive-vs-inclusive distinction.** The combining
   formula needs exclusive prefixes; inclusive prefixes give
   off-by-one wrong answers that look almost right but fail
   verification. Classic bug class.

This will be a sharp discriminator in Phase 3.

### Decisions made

- Used `simd_prefix_inclusive_sum(v) - v` instead of
  `simd_prefix_exclusive_sum(v)` for portability — works on any
  Metal version that has inclusive_sum, which is older.
- Used CONTIGUOUS per-thread slices (not strip-mined) because
  the algorithm is cleaner that way. Total bandwidth is the same
  either way; strip-mined would have better per-iteration
  coalescing but worse algorithmic ergonomics.
- Did NOT include a second scaffold variant exercising
  simd_prefix_exclusive_sum directly. One scaffold per problem
  is the established convention.

### Catalog state

| tier | count | techniques covered |
|---|---|---|
| 1 elementwise | 11 | broad |
| 2 reductions | **9** | row/col sums, max, argmax, softmax, **scan** |
| 3 tiled | 7 | matmul ×4, transpose, sgemv, conv2d_3x3 |
| 4 fused | 6 | layernorm, fused_linear_relu, attention ×4 |

33 problems total. Still gunning for 40–45; remaining gaps are
diversifying Tier 4 beyond attention (RMSNorm, GELU-fused,
residual ops) and a tile-with-halo conv variant in Tier 3.

---

## 2026-06-30 — p207 conv2d_3x3: stencil access pattern lands at 6.2× MPS

Third Tier 3 problem of the build-out. Conv2D is the canonical
stencil-access workload — the first such pattern in the catalog.
Catalog at 32 problems, Tier 3 at 7.

### What got built

`tier3_tiled/p207_conv2d_3x3/` — valid 3×3 convolution at
(1026, 1026) → (1024, 1024). One thread per output element; each
thread computes a 9-element 2D dot product over its input window.
The 3×3 weight matrix (36 bytes total) is read by every thread
and lives in L1/constant cache after the first access — no
explicit TG staging needed.

### Result — biggest Tier 3 win in the project

| metric | value |
|---|---|
| compiled / correct | ✓ / ✓ |
| max_abs_err | 2.86e-6 (1e-4 tolerance) |
| kernel_ms | 0.0362 |
| reference_ms (MPS / F.conv2d) | 0.2261 |
| **speedup** | **6.24×** |
| block_delta_frac | 0.0% (literally identical) |
| timing_trustworthy | true |

The kernel is *trivial* — two nested loops, one accumulator, one
store — and beats MPS by 6×. The block-1 vs block-3 medians
came out bit-identical at 0.0362 ms, which is the tightest
trustworthy reading I've seen from this harness.

### Why we win so big

This is the same story as p205 transpose (4.18×) only more so.
PyTorch's `F.conv2d` is built for batched multi-channel
workloads — `(N, C_out, C_in, K_h, K_w)`. For a single-channel,
single-output, no-padding call, MPS dispatches through that
general path and pays for indirection it doesn't need:

1. Batch dim handling.
2. Channel dim handling.
3. im2col-vs-direct path selection.
4. Padding logic (even though we passed none).
5. Backward-pass kernel generation, possibly cached.

Our kernel does *only* the math the problem actually requires.
**Arithmetic intensity:** 18 ops per 36 bytes read = 0.5 op/byte.
Even lower than sgemv. Per the rule from p206's quiz, low
arithmetic intensity → MPS's general-purpose tax dominates the
comparison → big specialization premium.

### Concept: stencil access pattern (first appearance)

A *stencil* is the access pattern where every output element
reads a fixed neighborhood around its input position — 3×3, 5×5,
or whatever the kernel size is. It shows up in convolutions
(deep learning), Laplacians (PDEs), Sobel/Gaussian filters
(graphics), and finite-element methods.

The defining performance characteristic: **input cache reuse
across adjacent output threads**. Thread (i, j) reads x[i..i+2,
j..j+2]. Thread (i, j+1) reads x[i..i+2, j+1..j+3]. These share
6 of 9 input elements. With one-thread-per-output and no
explicit staging, we rely on L1 to catch those shared reads —
and it does well at 3×3 because the working set is tiny.

For larger kernels (5×5, 7×7), the natural optimization is to
stage an input tile into TG memory once per TG, so the (TILE +
K - 1) × (TILE + K - 1) input window is read from device just
once per output tile of size TILE × TILE. That's the standard
tile-with-halo pattern. **Not implemented here** — at K=3 the
naive version is already fast enough to embarrass MPS, so the
TG-staged variant would be a separate problem (p208 or similar).

### LLM-eval implications

Conv2d 3×3 is exceptionally well-trodden in CUDA training data.
I expect every coder model to write a working kernel at this
size. **The interesting LLM-eval question is whether they
produce the naive version (which works here) or over-engineer
to the tile-with-halo version (which is unnecessary here but
sometimes the "textbook" answer).** Both should compile and
verify; the timing comparison will be interesting.

### Decisions made

- Used (1026, 1026) input shape so the (1024, 1024) output
  divides cleanly by TILE=16, avoiding bounds-check overhead.
- Did NOT add a TG-staged-input variant. At K=3 it's unnecessary
  and would muddy the "stencil access pattern, first encounter"
  pedagogy. Worth doing for K=5 or larger as a follow-up.
- Pre-multiplied ki by IN_W and K once per outer iteration
  rather than letting the compiler hoist it — both because it
  makes offset arithmetic easier to audit, and because explicit
  is usually better than implicit when teaching.
- Removed the unused OUT_H constant rather than leaving the
  warning. (Cleanup is comment-only for codegen — verified the
  earlier trustworthy reading still describes the post-cleanup
  kernel.)

### Tier 3 progress

Tier 3 went from 4 problems (all matmul) → 7 problems (matmul
×4, transpose, sgemv, conv2d_3x3) in three problems of work
today. The technique gap that motivated this build-out is now
mostly closed:

| technique | catalog problem |
|---|---|
| tiled matmul (cooperative load) | p201 |
| simdgroup_matrix matmul | p202 |
| simdgroup_matrix + threadgroup staging | p203 |
| double-buffered matmul (failed) | p204 |
| transpose (coalesced writes) | **p205** |
| sgemv (two-stage simd reduction) | **p206** |
| **conv2d (stencil access)** | **p207** |

Still missing from Tier 3 to round it out: batched matmul, a
GEMV-with-bias-add (or fused GEMV+activation), and a larger-K
conv (K=5 or 7) that actually motivates the TG-staged input
tile. That'd take Tier 3 to 10 problems.

### Open items

- p208 staged-tile conv at K=5 or 7 — would teach the
  "tile-with-halo" pattern and probably *not* beat MPS as
  decisively (higher arithmetic intensity at larger K).
- The "MPS general-purpose tax" story is now backed by three
  data points (p205, p206, p207). Worth a dedicated section in
  the Phase 4 report.

---

## 2026-06-30 — p206 sgemv: two-stage simd_sum reduction pattern, 1.8× MPS

Second non-matmul Tier 3 problem in two sessions. Matrix-vector
multiply at M=N=4096. Catalog now at 31 problems; Tier 3 at 6.

### What got built

`tier3_tiled/p206_sgemv/` — y = A @ x with one TG per output row.
Each thread does 16 multiply-adds against its slice of A's row,
then the TG reduces 256 partial sums to a scalar through a
two-stage `simd_sum`.

### Result

| metric | value |
|---|---|
| compiled / correct | ✓ / ✓ |
| max_abs_err | 6.87e-5 (well inside 1e-2 tolerance) |
| kernel_ms (warm, trustworthy) | 0.338 |
| reference_ms (MPS) | 0.607 |
| **speedup** | **1.80×** |
| block_delta_frac | 0.72% |

Comfortable win over MPS for a problem this simple. Memory-bound
(reading ~64 MB of A and reusing x across rows), so the
specialization win is bandwidth utilization rather than compute.

### Concept: two-stage simd_sum reduction

This is the first problem in the catalog using `simd_sum` as the
primary reduction primitive (p305/p306 used it for softmax inside
a single SG; here it's the central pattern). The mechanic is
worth pulling out:

1. **Stage 1 — within-SG.** `simd_sum(acc)` collapses the 32 lanes
   of one SIMD-group into a single float (returned in every
   lane). No barrier, no scratch.
2. **Publish.** Each SG's lane 0 writes its partial sum to a TG
   scratch array indexed by SG. 8 SGs → 8 floats.
3. **Barrier.** Stage 1's TG writes must complete before stage 2's
   reads.
4. **Stage 2 — across-SG.** SG 0 loads the 8 partials (lanes 0..7
   pull from `sg_partials[lane]`, lanes 8..31 contribute 0), runs
   `simd_sum` again to combine them, and lane 0 writes y[m].

The compare-point in the catalog is **p101 row_sum**, which does
the same operation via tree reduction in TG memory: 8 barrier
layers, 6 iterations of conditional writes. p206's pattern is
much shorter, uses one barrier total, and would generalize
trivially to any TG size that's a multiple of 32. **For
project-internal pedagogy, this is the modern replacement for
the tree-reduce-in-TG-memory pattern** any time the reduction
fits in one TG.

### Memory-pattern note (worth flagging for LLM eval)

A is read coalesced: adjacent threads at each strip-mine step
touch adjacent A addresses within a single row. x is read
coalesced for the same reason. **x is also reused M=4096 times
across TGs**, and its footprint (16 KB) almost certainly fits in
L1, so we shouldn't pay much across-TG bandwidth cost on x.
That's the kind of analytical step models often skip — the
naïve port works fine here, so I expect this won't be the
hardest LLM-eval problem; the trickier ones will be where the
naïve port has the access pattern wrong (like p205 transpose).

### Thermal observation worth carrying forward

By the time I went to confirm the post-cleanup measurement, the
GPU had been doing benchmark work for an hour. The next run
came back with block 3 SLOWER than block 1 (19% delta) — the
harness's stability check correctly identified thermal
throttling (suggested "let it cool, plug in power, retry").

I did NOT retry. The earlier warm reading (trustworthy, 0.7%
block delta) was already taken at a cooler thermal state and
the only thing that changed afterward was removing an unused
constant — a no-op for codegen.

**Carrying forward for the report:** the harness has THREE
states it can report: trustworthy / cold-ramp-untrustworthy /
thermal-throttle-untrustworthy. We've now hit all three
naturally in normal use. The Phase 4 thermal discipline writeup
should describe all three and the operator response for each
("retry", "retry warm", "stop and cool").

### Decisions made

- Used `simd_sum` instead of tree reduction. This is the modern
  pattern and the new project-internal default for in-TG
  reductions.
- Did NOT stage x into threadgroup memory. L1 should handle it
  at this size; an empirical p207 variant could verify.
- Removed the unused `M` constant rather than referencing it in
  a comment-only way to silence the warning — the comment
  documents *why* M is implicit in the grid.

### Open items

- p207 staged-x variant if the L1 hit rate proves low (probably
  unnecessary; benchmarking would tell).
- Tier 3 still needs more shape diversity: rectangular sgemv,
  batched matmul, a small conv. Three more would round out Tier
  3 to 9 problems — half the way to a Tier-3-complete catalog.

---

## 2026-06-29 — p205 transpose: first non-matmul Tier 3 problem, 4× MPS at 2048²

After scoping discussion concluded we should target ~40–45
strategically chosen problems rather than literally 60, I picked
the highest-leverage gap to fill first: **Tier 3 had four problems
and all four were matmul variants.** First non-matmul Tier 3
problem lands.

### What got built

`tier3_tiled/p205_transpose/` — tiled transpose at M=N=2048.
Each TG handles a 16×16 block of A, stages it through 1 KB of
threadgroup memory, and writes the transposed result to B with
coalesced stores. 16384 TGs × 256 threads = 4M total threads,
one per input element.

### Result — clean win over MPS

| metric | warm run 1 | warm run 2 |
|---|---|---|
| compiled / correct | ✓ / ✓ | ✓ / ✓ |
| max_abs_err | **0.0** | **0.0** |
| kernel_ms | 0.1212 | 0.1173 |
| reference_ms (MPS) | 0.5022 | 0.4898 |
| **speedup** | **4.14×** | **4.18×** |

Bit-exact (no arithmetic, just memory movement). Consistently
~4× faster than MPS across both warm runs.

### Why we beat MPS on something this simple

The kernel is tiny — read, barrier, write — but MPS's transpose
likely goes through a general-purpose path that handles arbitrary
shapes, dtypes, and non-contiguous strides. Our kernel is
specialized:

1. **Shape compile-time known** (M=N=2048). MPS does runtime
   dispatch.
2. **Single dtype** (fp32). MPS supports more.
3. **Always-contiguous outputs.** MPS supports views/strides.
4. **Optimal tile size baked in.** MPS picks tile size based on
   heuristics that don't know our exact shape.

This is a data point worth keeping for the Phase 4 report: for
**memory-bound ops with small specialization wins**, hand-tuned
kernels can decisively beat MPS even at "trivial" tasks. The
opposite of the matmul story where MPS's BLAS tuning dominates.

### Concept: coalesced writes (first appearance as the primary motivation)

The naïve thread-per-element transpose has the opposite of
phase-3 V reads from p305:

    b[j * M + i] = a[i * N + j];

Adjacent threads (varying `j`) read **adjacent A addresses** → one
cache-line load per SIMD-group row. But they write **B addresses
M floats apart** → 32 separate cache-line writes per SIMD-group
store. Catastrophic on a memory-bound kernel.

The tiled fix stages A into threadgroup memory in natural order,
then writes B with **swapped** indices on the TG read but the
same coalesced pattern on the B write:

    tile[ty][tx] = a[(by*TILE + ty) * N + bx*TILE + tx];  // coalesced A read
    barrier;
    b[(bx*TILE + ty) * M + by*TILE + tx] = tile[tx][ty];  // coalesced B write

The "uncoalesced" access has been pushed into the TG-memory
read (`tile[tx][ty]`), where it doesn't matter because TG memory
is fast for that scale.

This is the cleanest possible illustration of the
across-threads-at-one-instant rule from the corrigendum notes.
The transpose problem is canonical for teaching it because
naïveté here costs you nothing on reads and everything on writes.

### LLM-eval implications

This is exactly the kind of problem where I expect models to
**produce CUDA-like code that compiles in Metal but has
uncoalesced writes**. The naive port is easy and runs. The
tiled fix requires either Metal-specific knowledge or a transfer
of the CUDA bank-conflict pedagogy. Worth watching closely in
Phase 3 evals.

### Decisions made

- Used the simplest tiled approach. Did NOT add the +1 padding
  to `tile[TILE][TILE+1]` that NVIDIA tutorials recommend for
  bank-conflict avoidance — Apple's threadgroup memory has a
  different bank structure and we'd need empirical evidence the
  padding helps here before adopting the pattern. A future p206
  could test it.
- Did NOT add a non-square shape variant. Square is enough for
  the headline data point.
- Set `tolerance.rtol = 0.0` — bit-exact is achievable for pure
  data movement and we got it.

---

## 2026-06-29 — p306 attention_qstaged: Q-staging buys 6%, not 64× — L1 was already doing the work

Sequel to p305. Quiz argument said Q has 64× redundancy within
a TG and staging it should help. p306 implements exactly that.
The empirical answer is **~6%**, not 64× — the L1 cache was
absorbing nearly all of the redundancy in p305.

### What got built

`tier4_fused/p306_attention_qstaged/` — p305 with a single
algorithmic change:

- New TG memory buffer `q_stage[8 × 512]` = 16 KB.
- At the top of the kernel, 256 threads cooperatively load
  Q[m_base..m_base+7, :] into `q_stage` (16 elements per thread,
  coalesced strides across SIMD-group lanes).
- One `threadgroup_barrier`.
- Phase 1's Q load changed from
  `simdgroup_load(Q_frag, q + m_base*D + k_off, D)` to
  `simdgroup_load(Q_frag, q_stage + k_off, D)`.
- Phases 2 and 3 unchanged.

TG memory: 16 KB scores + 16 KB Q-stage = **32 KB**, at Apple's
per-TG cap.

### Result — small win, but a real one

Warm-state, interleaved measurements (back-to-back to keep MPS
state comparable):

| metric                       | p305 (no stage) | p306 (Q staged) |
|------------------------------|-----------------|-----------------|
| compiled / correct           | ✓ / ✓           | ✓ / ✓           |
| max_abs_err                  | 1.64e-7         | **1.64e-7**     |
| kernel_ms (warm)             | 0.3177          | **0.2994**      |
| reference_ms (warm MPS)      | 0.4499          | 0.4608          |
| speedup vs MPS (warm)        | 1.41×           | **1.54×**       |
| block_delta_frac (warm)      | 0.001           | 0.001           |

**Kernel speedup from staging: 5.8%.** A real but modest win.

### Why the prediction was off by ~10× — the L1 lesson

The quiz analysis counted 64 redundant loads of each Q element
per TG. That counts *requests* to the memory subsystem, not
*device traffic*. Apple's L1 cache sits between SGs and the
device. When SG s loads Q[m_base..+7, kt*8..+7] and then SG t in
the same TG loads the same tile microseconds later, the load
hits L1, not device DRAM. The "redundancy" only translates to
latency at the L1 access level — not the DRAM bandwidth level.

The 6% remaining is L1's own access cost (still nonzero — it's
a load that has to be issued, decoded, and pipelined) and any
spillover at the cache eviction boundary.

**One-sentence frame:** *for caches with reuse-friendly access
patterns, redundancy in load count overstates the actual cost
proportional to the cache hit rate.* The Q access pattern in
p305 is maximally cache-friendly (same tile reloaded immediately
in the next inner-loop iteration), so almost all the
"redundancy" was free.

### Warm vs cold variance — the other story this run

While running the comparison I got two artifacts worth keeping:

1. **First p306 run was cold.** block1 = 0.42 ms, block3 = 0.30
   ms — a 29% delta. The harness correctly threw a stability
   error: `timing_trustworthy: false`. The GPU was thermally
   idle from the previous quiz break and took ~3 timing
   iterations to ramp to high-perf state. Re-running with the
   GPU already warm gave the clean 0.30 ms reading.
2. **Steady-state speedup vs MPS is much higher than p305's
   first-shot reading.** p305's "0.996× MPS, tied with the
   reference" reading two sessions back was a cold-MPS artifact
   (MPS's first dispatch had launch overhead). In steady state,
   p305 is **1.41× MPS**, p306 is **1.54× MPS**. Both
   meaningfully beat MPS once the system is warm.

Lesson: report-time, we need to be explicit about which state
the speedup is measured in. The honest framing for the Phase 4
report is something like:

> p305: tied with MPS on a cold first call (0.996×), 1.4× faster
> in warm steady state.

Neither number is "wrong" — they answer different questions
("what does the first dispatch cost?" vs "what does a sustained
workload deliver?"). The benchmark already records both
implicitly via the block1/block3 medians.

### Why p306 starts colder than p305

A small observation worth checking later: p306 needed more
warmup iterations than p305 to reach steady state. Plausible
cause: p306 requests 32 KB of TG memory vs p305's 16 KB. On
Apple Silicon, threadgroup-memory pressure can lower occupancy
(how many TGs can fit on one core simultaneously), which means
fewer concurrent TGs to hide latency early in the launch. Once
the work queue is full it doesn't matter, but at startup with a
cold GPU, lower occupancy means a longer ramp.

Not actionable as a fix — it's a tradeoff inherent to staging.
Worth noting.

### Decisions made

- Did NOT also restructure to share K across SGs (the "p306 plus"
  option the user passed on). The single-variable change isolates
  the Q-staging effect cleanly. The fact that it's only 6%
  suggests further effort here has diminishing returns — the
  remaining gap to a fully-optimized kernel is probably 20–30%
  at best, and would require non-trivial restructure.
- Kept the same launch geometry, scratch layout, softmax, and
  PV phase as p305. The diff between the kernels is small enough
  that the result is *cleanly attributable* to Q-staging alone.
- Did NOT bump the harness's warmup count to absorb the cold-GPU
  ramp. The stability_error caught the problem honestly and
  re-running with a warm GPU gave a clean reading.

### Open items / carry forward

- **Cold-state benchmarking is its own measurement question.**
  The harness's A/B/A check is doing its job (catching cold-GPU
  bias), but doesn't smoothly handle "the kernel needs more
  warmup at higher TG-memory pressure." Bumping warmup count is
  a band-aid; better would be a "ramp until block1≈block3"
  prologue, which is a Phase-1-timing-trust task we punted on.
- **The full-restructure p307 (shared K + Q-stage) would test
  whether the remaining 30% gap is closeable** without giving up
  much TG memory. Expensive to write; small expected payoff.
  Probably not worth doing unless we want a "best-case fully-
  optimized attention" data point for the report.
- **The Phase 4 report needs an explicit cold/warm framing.**
  We have enough data now to characterize both regimes for
  every Tier 4 problem — worth a section.

---

## 2026-06-29 — p305 attention_simdmatmul: matrix engine closes p304's 2:1 deficit in one shot, matches MPS

Direct test of the quiz hypothesis: "if the 2:1 deficit at p304
is the matrix-engine gap, then re-doing the matmul phases with
simdgroup_matrix should close it." It did. **First compile, first
correct, first-run timing tied with MPS to within noise.**

### What got built

`tier4_fused/p305_attention_simdmatmul/` — same fused attention as
p304, but the QK^T and PV phases run on the Apple GPU matrix unit
via `simdgroup_matrix<float, 8, 8>`. The softmax in between still
uses scalar code, now sped up via SIMD-group lane reductions
(`simd_max`, `simd_sum`) instead of tree reductions in TG memory.

### Result

| metric             | p303 (M=64)   | p304 (M=512, scalar) | **p305 (M=512, simdmatmul)** |
|--------------------|---------------|----------------------|------------------------------|
| compiled / correct | ✓ / ✓         | ✓ / ✓                | ✓ / ✓                        |
| max_abs_err        | 1.79e-7       | 1.79e-7              | **1.64e-7**                  |
| kernel_ms          | 0.032         | 1.71                 | **0.435**                    |
| reference_ms (MPS) | 0.26–0.29     | 0.83 (variable)      | 0.43                         |
| **speedup**        | 8.0×          | 0.49×                | **0.996×**                   |
| block_delta_frac   | < 0.01        | 0.029                | 0.0001                       |

Going from p304 to p305 is a **3.9× kernel-time speedup** on the
identical algorithm at the identical size. That entire delta is
the matrix engine kicking in.

Numerical accuracy didn't suffer — max_abs_err is *better* than
p304's scalar version, suggesting the matrix unit's accumulation
order at fp32 is at least as well-conditioned as a straight
left-to-right scalar sum for length-512 random-normal dot products.

### TG layout — the central design decision

The interesting design choice was the threadgroup structure. Two
nearby patterns from the project don't quite work:

- **p303/p304 layout** (one TG per query row): can't reach matrix
  throughput. Each TG only produces 1 row of scores; the matrix
  unit's natural output is 8 rows at a time.
- **p202 layout** (one SIMD group per output tile, TG = 32 threads):
  can't fuse softmax. The full row of scores is spread across 64
  TGs, so cross-TG sync would be needed for the row reduction.

The right middle: **one TG per block of 8 query rows, 8 SIMD groups
inside it.**

- 64 TGs total. TG t handles query rows [8t, 8t+8).
- 8 SIMD groups (256 threads / TG). SG s handles output column
  block [s*64, (s+1)*64) in BOTH phase 1 and phase 3.
- In phase 2, the same SGs change role: SG s does the softmax for
  row s. Beautifully symmetric — 8 rows, 8 SIMDs, 1:1 mapping.

That symmetry is what made the kernel one-shot-correct. Every
phase has a clean owner.

### Concept: writing simdgroup_matrix to threadgroup memory

p202's matmul stored its output tile directly to device memory.
For a fused kernel we need to store into **threadgroup memory**
instead so the next phase can read it without a global round-trip:

    simdgroup_store(C_frag, scores + c0, M);   // scores is threadgroup float*

Same intrinsic, the compiler picks the threadgroup overload from
the pointer type. Latency is much lower than device store.

The corresponding load in phase 3:

    simdgroup_load(P_frag, scores + kt * 8, M);

reads probs[0..7, kt*8..kt*8+7] from threadgroup memory directly
into a matrix register. No copy through registers, no thread-by-
thread shuffling. The matrix unit + threadgroup memory is a clean
pair.

### Concept: SIMD-group lane reductions

Apple GPUs expose `simd_max`, `simd_sum`, `simd_xor`, etc. —
intrinsics that reduce across the 32 lanes of a SIMD group in
hardware. No tree reduction in shared memory, no barriers. The
return value is the same in every lane.

For p305's softmax, each SG holds one row's 512 elements (16 per
lane). Compute a per-lane max over those 16, then `simd_max` to
get the row max. Same shape for sum.

This is much faster than the p303/p304 pattern of writing to
scratch + tree-reduce + barrier — and crucially, it doesn't
require coordination across SIMD groups, which means there's no
intra-phase barrier in phase 2 at all. Just one barrier each
between phases 1↔2 and 2↔3.

### Concept: transpose-load for A @ B^T

Computing C = Q @ K^T using the matrix unit:

    simdgroup_load(K_frag, k + c0 * D + kt * 8, D, ulong2(0, 0), true);

The 5th argument (`transpose_matrix = true`) transposes the
loaded tile in-register. The matrix unit then does its standard
A @ B operation, and the net effect is C = Q @ K^T. Without this
flag, we'd have to either physically transpose K in memory (huge)
or use a different matmul intrinsic.

### What's left on the table (probably small)

The 0.4% behind MPS is within run-to-run noise. To beat MPS we'd
need to stage K/V tiles into TG memory so SGs in the same TG
share loaded tiles (currently each SG loads its own K-strip from
device for every output tile, giving 8× redundant device loads
per TG). That's the standard "Q-shared block" optimization — a
plausible p306 variant. Expected payoff: ~1.5–2× more.

But for the Phase 4 report, **p305 matching MPS is the
load-bearing data point**. It confirms the deficit at p304 was a
hardware-engine choice, not algorithm or memory layout.

### What didn't go wrong (and why)

I expected at least one round of debugging — first GPU kernel of
this complexity in the project, three new Metal idioms at once
(simdgroup_matrix into TG memory, simd lane reductions, transpose
load). It worked first try. Three reasons in hindsight:

1. **p202 already established the matrix-unit mechanics.** The
   tile-loop structure, the `simdgroup_load/store/multiply_accumulate`
   triple, the 8×8 dim — all proven. Only the integration was new.
2. **The 1:1 SG↔row mapping eliminated a whole class of bugs.**
   No cross-SG synchronization within a phase, no thread-index
   arithmetic in the softmax. The 8-rows / 8-SGs choice wasn't
   arbitrary; it was chosen to remove sync surface area.
3. **The offset math was the only place to get wrong**, and I
   commented every tile coordinate inline. Slow to type, fast to
   audit.

### Decisions made

- Did NOT stage K or V into TG memory. Going for "matches MPS"
  was the milestone; further optimization is a separate problem.
- Used `simd_max` / `simd_sum` instead of tree reductions. This
  is the first project use of these intrinsics; the difference
  vs the tree pattern is real (no barriers, no scratch use for
  reduction workspace) and worth highlighting in the report.
- Bumped tolerance to 1e-2 atol (vs p304's 1e-3). simdgroup_matrix
  accumulation order ≠ scalar fp32; p202 already documented this.
  Headroom is 4 decades over observed 1.64e-7, so it's safe.

### Open items / future problems

- p306: staged-K/V variant. Likely 1.5–2× more vs p305. Would
  test whether we can BEAT MPS at this size, not just match.
- Larger sizes (M=D=1024, 2048): if MPS scales well, we likely
  stay at parity. If MPS's matmul has a sub-optimal size point,
  we could overtake. Worth one more data point for the curve.
- The crossover point question from the p304 NOTES — would a
  scalar p305 actually be needed now? p305 is the better kernel
  at large sizes; p303 is the better kernel at small sizes
  (dispatch overhead dominates). A scalar variant at M=D=128/256
  is still the cleanest way to draw the curve.

---

## 2026-06-29 — corrigendum: phase-3 V reads in p303/p304 are coalesced, phase-1 K reads are the uncoalesced ones

While walking through a quiz on "which lever claws p304 back above
1×", I traced the V access pattern in phase 3 and noticed the
kernels and specs in both p303 and p304 had it labeled
backwards. The reality:

- **Phase 1** (`k[tid * D + d]`): at fixed `d`, thread `t` reads
  `K[t*D + d]`. Adjacent threads are **D floats apart** in memory.
  At D=64 (p303) this is bad; at D=512 (p304) it's catastrophic —
  each SIMD-group load becomes 32 separate cache-line fetches.
  **This is the real bandwidth cost** in both kernels.
- **Phase 3** (`v[j * D + tid]`): at fixed `j`, thread `t` reads
  `V[j*D + t]`. Adjacent threads read **adjacent columns of the
  same row** → contiguous addresses → fully coalesced. No issue.

The original docs in p303 had phase 1 labeled "coalesced within
the row" (technically true — *each individual thread* reads its
row sequentially in time — but misleading, because what matters
for memory bandwidth is the **across-threads** pattern, not the
within-thread one) and phase 3 labeled "uncoalesced — leaves
room for a staged-V follow-up problem." That follow-up problem
would not have helped; staging K (or transposing its access
pattern) is what would.

p304 inherited both errors from copying p303's structure.

### Why I missed this twice

I conflated "sequential reads from a single thread's
perspective" with "coalesced loads from the SIMD group's
perspective." Those are different properties and only the second
one matters for memory throughput. When a thread reads its row
in order, that's a single thread's good prefetch behavior — but
in the same instruction, the *other 31 threads* in its SIMD
group are reading their own rows from different cache lines, so
the actual load issued by the hardware touches 32 lines.

**One-sentence frame to remember:** for GPU memory bandwidth,
the only access pattern that matters is the one across adjacent
threads at a single instant — never the one a single thread
takes over time.

### Why it didn't show up as a performance bug

The 8× win at p303 is dispatch-overhead-dominated, not
bandwidth-dominated, so the bad K reads were never the
bottleneck at that size. At p304 the bandwidth cost *is* part of
the deficit, but it's dwarfed by the compute gap vs MPS's
matrix-engine matmul — so it didn't surface as "this looks
slower than expected for the bandwidth." It just looked like
"this loses to MPS," and that read fine without questioning the
coalescing labels.

### Files updated (comments only)

p303_attention_head: spec.py, attention_head.metal,
attention_head_scaffold.metal.
p304_attention_large: spec.py, attention_large.metal,
attention_large_scaffold.metal.

No behavior change. Re-ran both kernels: p303 still ~7-8×
(within MPS-warm/cold noise band), p304 still ~0.25-0.49×.

### Lesson for future kernel reviews

When reading any GPU access-pattern claim, ask: "is this
talking about a single thread's reads over time, or about
adjacent threads' reads at one moment?" The two patterns can be
opposites, and the comment that doesn't specify which is doing
half the job.

---

## 2026-06-29 — p304 attention_large: same kernel, M=D=512 → 0.49×, the other end of the curve

Follow-up to p303 within the same session. Last entry's open
item was: "A future problem at M = D = 512 or 1024 would show the
same kernel hitting its compute-bound regime and the speedup
shrinking. Worth doing next session for the report." This
delivers exactly that.

### What got built

`tier4_fused/p304_attention_large/` — a literal copy of the p303
algorithm with three constants changed (M, D, SCALE) and the
launch config rescaled. No algorithmic change. No new Metal
concepts. The point of the problem is the *measurement*, not the
kernel.

- 512 TGs × 512 threads/TG = 262144 total threads
- Threadgroup memory: 2 × 512 × 4 = 4 KB (still well under 32 KB)
- Reduction tree depth: 6 → 9 levels (M halves three more times)
- Dot products in phase 1 and phase 3 each grow from length 64 to
  length 512 (8× per-thread compute)

### Result — predicted regime inversion

| metric                       | p303 (M=D=64) | p304 (M=D=512) |
|------------------------------|---------------|-----------------|
| compiled / correct           | true / true   | true / true     |
| max_abs_err                  | 1.79e-7       | **1.79e-7**     |
| kernel_ms                    | 0.032         | 1.71            |
| reference_ms (MPS wallclock) | 0.26–0.29     | 0.83            |
| **speedup**                  | **~8×**       | **0.49×**       |

The max_abs_err is **bit-identical** between the two — fp32 reductions
at length 512 with random-normal data did not eat into the precision
budget the way I expected. Tolerance 1e-3/1e-3 has 5 decades of
headroom either way.

### Why the regime inverted (this is the actual lesson)

Compute scaled ~500× (M·M·D went from 0.5M ops to 134M).
Our kernel scaled **53×** in wall time (sublinear, because launch
overhead is amortized across more work now).
MPS scaled **3×** in wall time (essentially constant compute time
plus a fixed multi-dispatch tax that didn't grow).

> The 8× win at p303 was MPS's dispatch overhead, made visible by a
> problem too small to hide it. At p304 the dispatch tax is invisible
> against MPS's BLAS-tuned matmul, and our naive
> one-thread-per-output-element dot product loses to it 2:1.

The crossover (where our kernel matches MPS) is roughly where
compute time ≈ MPS's dispatch overhead — somewhere around M=D≈200
by linear interpolation in log space. Not measured; not worth
measuring just to draw the curve smoother.

### The fusion thesis now has a curve, not just points

| problem | speedup | regime |
|---|---|---|
| p303 attention_head (M=D=64)    | **8.0×** | dispatch-overhead dominated |
| p104 row_softmax                | 1.08×    | mixed |
| p301 layernorm                  | 0.97×    | MPS already fused |
| p302 fused_linear_relu          | 0.55×    | MPS BLAS-tuned matmul |
| p304 attention_large (M=D=512)  | **0.49×**| compute-bound; MPS matmul wins |

p303 and p304 are the same kernel, two different sizes, two
different regimes. That's the cleanest possible illustration that
"fusion wins" is conditional on problem size relative to dispatch
overhead. For the Phase 4 report, this pair carries the argument
on its own — no other annotation needed.

### Concept: a single algorithm can be on either side of a perf crossover

I keep tripping over this — "is our kernel fast?" is not a
property of the kernel alone. It's a property of (kernel,
problem size, baseline implementation, dispatch model). p303 and
p304 share *everything* except size, and one is 8× faster while
the other is 2× slower. That's not 16× of kernel skill — it's
the baseline's fixed costs being divided by different amounts of
real work.

**One-sentence frame:** for any fused candidate, dispatch overhead
is a constant *bonus* and compute efficiency is a *ratio*; whichever
dominates depends on problem size, not on how good your code is.

### Decisions made

- Did NOT try to optimize p304. The 0.49× is the *data*. A
  SIMD-group matmul variant or a staged-V variant might claw it
  back, but those would be separate problems showcasing different
  techniques. p304's job is to be the boring twin of p303.
- Kept the kernel literally identical in structure. Resisted the
  temptation to refactor the shared body into an included header
  — the visual diff of *only constants changed* is itself the
  pedagogical point.

### Open items (carry forward)

- The implied crossover point (~M=D≈200) is interpolated, not
  measured. A p305 at M=D=128 or 256 would draw the actual curve
  if the report wants one.
- p304's phase-3 V reads are still uncoalesced. A staged-V
  variant would test how much of the 2:1 deficit is uncoalesced
  reads vs naive matmul. Probably mostly the latter, but
  unmeasured.
- The "MPS dispatch overhead ≈ 250 µs total" estimate from p303
  is now reinforced: p304's reference_ms = 0.83 ms includes that
  same ~0.25 ms tax plus ~0.58 ms of actual compute. Three
  dispatches, ~0.19 ms each — consistent.

---

## 2026-06-29 — p303 attention_head: biggest MPS win in the project (~8×), fusion thesis at its sharpest

User filled in the p101 scaffold themselves to grok the workflow,
then we returned to building. The natural next problem was the
fusion thesis's biggest stake: a single-kernel attention head.

### What got built

Single-head scaled dot-product attention fully fused in one kernel:

  out = softmax(Q · K^T / sqrt(D)) · V

Shape M = D = 64 (sequence length × head dimension). One TG per
query row (64 TGs), 64 threads per TG, every phase has a clean
1:1 thread-to-work mapping. Five phases share one scratch[M]
array (scores → reduction workspace → probabilities), with
barriers at every transition. Q row staged into a small q_row[D]
buffer for broadcast reuse. K read directly from device (one row
per thread, row-internal reads are sequential). V read directly
from device in phase 3 (uncoalesced — staging would be a future
problem).

The kernel subsumes nearly every concept from earlier tiers:
threadgroup memory, tree reductions (twice — once for max, once
for sum), softmax with max-subtract stability, scratch reuse with
careful barriers, dot products. Building it required NO new
concepts — just composition of existing ones.

### Result — best speedup in the project by far

| metric | value |
|---|---|
| compiled | true (no warnings) |
| correct | true |
| max_abs_err | 1.79e-7 (~1 ULP) |
| kernel_ms | 0.032 (rock-solid across runs) |
| reference_ms | 0.26 – 0.29 (some noise on these microsecond scales) |
| **speedup** | **~8×** vs MPS |

Previous best was p108 row_argmax at 2.16×. This is roughly 4× more
dramatic.

### Why the 8× is real but context-dependent

The actual compute is tiny: M·M·D + M·M·D + M·M ≈ 530K float ops,
which executes in ~30µs on our throughput. MPS dispatches attention
as at least three kernels (matmul, softmax, matmul). Each dispatch
has ~80–100µs of launch overhead on macOS. **Three dispatches ≈
250µs of overhead, which roughly matches MPS's 260–290µs total time.**

Our 1-dispatch kernel pays launch overhead once. The 8× win is
essentially the dispatch-overhead difference, magnified by the small
problem size where compute can't amortize the multi-dispatch cost.

**Critical caveat for the report**: at production attention shapes
(M = D = 512 or 2048), compute time grows as M·M·D while dispatch
overhead stays constant. The fusion advantage shrinks proportionally.
At M = D = 2048 the speedup would be much closer to 1×, possibly
even below it if our matmul deficit (from Tier 3) dominates.

The 8× here is **honest at this size**, not a general claim about
attention. The benchmark catalog should ideally include attention at
multiple sizes to show the curve — flagging that as a future problem.

### The complete fusion picture, 5 data points across Tiers 2 and 4

| problem | speedup | MPS implementation profile |
|---|---|---|
| p303 attention_head | **~8×** | multi-dispatch, dispatch overhead dominates |
| p104 softmax | 1.08× | multi-dispatch, individual ops slow |
| p301 layernorm | 0.97× | fully fused already |
| p302 fused_linear_relu | 0.55× | multi-dispatch but BLAS-tuned matmul |

Four distinct outcomes, four distinct underlying causes. The
benchmark now exercises essentially the full space of fusion
opportunities. The Phase 4 report will have a much richer story
than "fusion wins" or "fusion doesn't matter":

> Fusion wins are real but their magnitude depends on (a) how many
> dispatches MPS uses for the composite, (b) how fast MPS's individual
> kernels are, (c) the size of the problem relative to dispatch
> overhead, and (d) whether the candidate's own per-op cost is
> competitive with MPS's tuned versions.

### Decisions made

- Used M = D = 64 to get the maximum dispatch-overhead win. A future
  problem at M = D = 512 or 1024 would show the same kernel hitting
  its compute-bound regime and the speedup shrinking. Worth doing
  next session for the report.
- Did NOT stage V into threadgroup memory. Phase 3 reads are
  uncoalesced. Acceptable for the baseline; saves complexity. The
  staged variant would be its own problem.
- Did NOT introduce any new Metal concepts in p303. Every piece is
  reused from earlier tiers. This is intentional — the educational
  payoff is in showing that complex kernels are compositions of
  simple patterns, not in piling on new primitives.

### User-side milestone

Earlier in the session, user filled in the p101 scaffold themselves
from scratch (with iterative compiler-error guidance). Took 3
iterations to converge: first attempt had Python-shape syntax
throughout; second fixed the Python-vs-C++ issues but kept TODO 3
and TODO 4 mixed; third fixed the for-loop syntax but missed two
semantic issues (barrier inside loop, thread-0 write inside loop).
I applied the final three fixes when asked. Result: bit-identical
behavior to the reference kernel (max_abs_err 1.14e-5, speedup
0.51x). The learning loop works — the scaffold's TODOs lead
someone through every concept in the right order, and the compiler
errors are sharp enough to catch syntax issues without giving away
semantic ones.

---

## 2026-06-26 — Tier 3 matmul ladder, Tier 4 layernorm + fused_linear_relu, fusion thesis refined

Jumped from Tier 2 (reductions) to Tier 3 (tiled). One problem
shipped: p201_matmul_tiled, a naive 16×16-tile kernel for square
1024×1024 float32 matmul. The biggest concept jump of the project
so far — three new Metal patterns arrive together.

### What's new

- **Cooperative tile loading.** Up to now, threads in a TG either
  loaded their own input (elementwise) or loaded one element into a
  shared scratch for reduction. For matmul, threads cooperatively
  load TILES of A and B into threadgroup memory, then *all 256
  threads reuse the loaded tile* for the inner k-loop. Device-memory
  reads per output drop from 2·K=2048 to 2·K/TILE=128 — a 16× cut
  in device traffic from one design choice.
- **K-loop accumulator.** Each thread carries one float across the
  whole outer loop (K/TILE iterations), adding TILE multiply-adds
  per iteration. No tree reduction, no shared-memory reduction at
  the end — the accumulator stays in a register the whole time.
- **2D output block per TG.** First problem where a single TG
  computes a 2D block of output. Thread (ty, tx) in TG (by, bx)
  owns exactly C[by·TILE+ty, bx·TILE+tx].

### Result

| metric | value |
|---|---|
| compiled | true |
| correct | true |
| max_abs_err | **0.0 exactly** (1M elements all bit-identical) |
| kernel_ms | 3.00 (extremely stable, A/B/A delta < 0.6%) |
| reference_ms | 1.91 first run, 0.70–0.90 steady state |
| speedup | ~0.29× steady state (~0.64× first run) |

### Two findings worth pinning

**1. The reference shows MPS's first-run shader compile cost again.**
First call to torch's matmul on (1024, 1024) shape in a fresh process
takes 1.91ms. Subsequent calls in *new* processes drop to 0.7–0.9ms,
matching the OS-level shader cache pattern we documented for Tier 1
elementwise. Implication for Tier 3: every benchmark run needs at
least one warmup invocation of the reference op before timing, or
the headline number will undercount MPS's overhead. The harness
currently times the reference once after the candidate's first
block, which is enough for the first-run penalty to land inside the
measurement. Worth a follow-up harness pass.

**2. Bit-exact match against torch CPU reference is real, not a bug.**
Manually sanity-checked: same shape, same min/max/mean to all
reported digits, same values at every sampled position. Hypothesis:
PyTorch on macOS uses Apple's Accelerate framework, which uses a
tile structure compatible with our 16×16 layout. The accumulation
order ends up matching exactly. This won't hold for differently-
tiled future matmul variants (32×32, vectorized loads with float4,
SIMD-shuffles), so the spec's tolerance is kept at the realistic
slack (atol=1e-2) rather than being collapsed to 0 just because
p201 happens to match.

### Decisions made

- Sizes set to M=N=K=1024 — small enough for fast iteration (~3ms),
  large enough to expose meaningful timing differences against MPS,
  and divisible by TILE=16 cleanly so we don't need bounds checks
  in this baseline. Bounds-checked variant comes as a later problem.
- TILE=16 chosen for (a) clean SIMD-group division (16×16 threads
  = 256 = 8 SIMD-groups), (b) modest shared-memory footprint
  (2KB per TG), (c) clean dim divisibility. Larger tiles (32×32) will
  be a follow-up optimization problem.
- Did NOT attempt to beat MPS in p201. The naive tiled kernel is the
  baseline; future Tier 3 problems (vectorized loads, more outputs
  per thread, simdgroup_matrix, AMX) will claw back the 4× gap.

### p202 matmul_simdgroup — Apple matrix unit, 1.82× over p201

Same op, same shapes (1024×1024×1024 float32) as p201, but the kernel
hands the heavy lifting to Apple's per-SIMD-group matrix unit via
Metal 3's `simdgroup_matrix<float, 8, 8>` type. Each SIMD-group
(32 threads, one TG) computes one 8×8 tile of C; the K dimension
walks in 128 steps of `simdgroup_multiply_accumulate`. No explicit
threadgroup_barrier in the hot loop — matrix-unit ops are
SIMD-group-synchronous.

### Side-by-side, the Tier 3 picture so far

| problem | kernel_ms | speedup vs MPS | step |
|---|---|---|---|
| p201 (manual 16×16 tiles) | 3.00 | 0.29× | baseline |
| p202 (8×8 matrix unit) | **1.65** | **0.47×** | 1.82× over p201 |
| MPS (steady state) | 0.77 | 1.00× | target |

p202's A/B/A delta was 0.01% — the cleanest timing measurement
the project has produced. Suggests the matrix-unit path is
extremely consistent (no contention, no thermal sensitivity at this
brief runtime), whereas the manual tile path varies a bit more from
scheduler / cache effects.

### What's eating the remaining 2× vs MPS

Three diagnosis candidates, each a candidate next problem:

1. **Per-iteration device-memory loads.** We load A and B tiles from
   device memory on every one of 128 K-iterations. MPS likely stages
   multiple K-tiles into threadgroup memory once and lets the matrix
   unit chew through them — turning device reads into threadgroup
   reads.
2. **One matrix tile per TG.** Each TG produces an 8×8 patch of C.
   MPS likely uses larger TG-level tiles (16×16 or 32×32 of C
   produced from 4 or 16 matrix-unit ops), amortizing launch overhead
   across more arithmetic.
3. **No double-buffering.** Each K-iteration loads, computes, loads,
   computes — no overlap. MPS may pipeline so a load and a compute
   run in parallel.

### Surprising continuation: still bit-exact

p202 also reports `max_abs_err: 0.0` against torch's CPU reference.
PyTorch on macOS uses Accelerate, and on M-series Accelerate routes
matmul through the same matrix unit (likely AMX/AMX-2 on the CPU side
and SIMD-group matrix on the GPU side). Both paths converge on the
same numerical result. This won't survive a future variant that
re-orders the K-tile traversal — kept tolerance at atol=1e-2 anyway.

### Decisions made

- Did NOT chase the remaining 2× this session. The naive matrix-unit
  baseline is the clean teaching artifact; staged / multi-tile-per-TG
  variants are each their own problem with their own concept content.
- Did NOT switch the spec's tolerance to 0 even though the observed
  error is exactly that. The 1e-2 slack is the realistic value;
  collapsing to 0 would make the spec brittle to future kernels
  whose accumulation orders differ.

### p203 matmul staged + multi-tile — two optimization layers, 1.62× over p202

Combined two optimizations in one kernel:
1. Each TG now produces a **16×16 patch of C** as four separate 8×8
   matrix-unit tiles in registers (C_tl, C_tr, C_bl, C_br).
2. The outer K-loop **stages a 32-deep slab of A and B into threadgroup
   memory** once per K_STAGE=32 columns. The matrix unit reads operands
   from TG memory in the inner K-loop.

The two layers reinforce each other: staging without reuse would just
shuffle bytes through TG memory for no gain, but with four output tiles
sharing each staged A/B slab, every device load fans out across four
matrix-unit ops. Per-output device reads drop from p202's ~256 to ~128.

### The full Tier 3 picture now

| problem | kernel_ms | speedup (steady MPS) | over previous |
|---|---|---|---|
| p201 naive 16×16 tiles | 3.00 | 0.29× | baseline |
| p202 simdgroup matrix unit | 1.65 | 0.47× | 1.82× |
| p203 staged + multi-tile | **1.02** | **~0.65×** | 1.62× |
| MPS (steady) | ~0.66 | 1.00× | — |

Three optimization steps; the gap closed from 3.5× down to ~1.5×.
**Each step roughly doubled relative position vs MPS.** That ratio is
worth noting — it suggests the optimization landscape here is fairly
log-linear, not abruptly diminishing.

### Two harness frictions that surfaced (still not blocking, but accumulating)

1. **A/B/A stability threshold tripping on warm GPU.** First p203 run
   measured 12% A/B/A delta and was flagged untrustworthy (threshold:
   7%). Retry resolved it (delta dropped to 0.2%). Cause: we've been
   running Tier 3 kernels in quick succession and the GPU warms up.
   Real fix: either bump threshold to 10-15% with a "thermal warmup"
   note, or run an untimed warmup dispatch before block 1.
2. **MPS reference-time variance.** Across just the p203 retries the
   reference bounced 0.66ms (steady) to 1.65ms (cold). Makes the
   "speedup vs MPS" headline unreliable; the kernel's *own* time is
   solid. Same shader-cache issue we documented for Tier 1.

Both are well-known, both have known fixes, neither blocks any current
problem from being measurable. Promoting them in priority though —
the reference-warmup fix would let us cite a single honest MPS number
per problem instead of a range.

### Decisions made

- Renamed "p203 staged" to actually combine staging with multi-tile,
  because staging alone doesn't deliver the promised reuse. Spec
  description was updated to be accurate to what the kernel does.
- Did NOT bump the spec's A/B/A threshold to silence the false-positive
  thermal alert. The threshold is doing its job; we just want to fix
  the underlying measurement reliability separately.
- Added a scaffold post-hoc (forgot it in the first commit). p203's
  scaffold matters more than usual because the kernel is structurally
  complex — LLMs being benchmarked on this problem need the framing
  the scaffold provides.

### p204 — double-buffered matmul that BACKFIRES (lesson, like p105)

Added textbook double-buffering on top of p203: two sets of A/B
stage buffers, prologue + main loop with overlapped load/compute +
epilogue. Structurally correct, verifies bit-exact. But the kernel
runs **45% slower than p203** (1.48ms vs 1.02ms).

Three teaching artifacts now in the catalog showing different ways
"textbook" GPU optimizations can fail:

- **p105**: 2D thread tile for coalescing → starves parallelism
- **p107**: atomic fan-out for row_sum → cooperative already saturates
- **p204**: explicit double-buffering → compiler already pipelined

Two hypotheses for why p204 failed (likely both contribute):

1. **Apple's Metal compiler already pipelined p203's independent
   loads with compute.** Within each barrier-bounded region, the
   compiler is free to overlap device-memory loads with matrix-unit
   ops as long as the dependency graph allows. Our explicit
   double-buffer added no information the compiler didn't already
   have.
2. **2× threadgroup-memory footprint (8KB vs 4KB) reduced
   occupancy.** Apple Silicon co-resident multiple TGs per core to
   hide latency by switching between them. Doubling TG memory
   halves the resident TG count, weakening latency hiding — the
   optimization meant to hide latency reduced the GPU's *general*
   ability to hide latency.

Renamed to `p204_matmul_double_buffered_backfires` so the failure
mode is visible at the path level. Same playbook as p105.

### The emerging principle, now with three data points

"Textbook GPU optimizations are contextual."

A pattern that helps in one context can be net-negative when:
- The compiler already does it implicitly (p204)
- It breaks a different constraint (p105)
- The baseline didn't have slack to exploit (p107)

Useful framing for the eventual Phase 4 report: an LLM that has
*memorized* GPU optimization recipes will propose p105/p107/p204-
style kernels with confidence. An LLM that has *learned* GPU
performance will recognize when context invalidates the recipe.
The benchmark catalog now has three problems whose specific role
is to surface this distinction.

### The full Tier 3 ladder

| problem | kernel_ms | speedup vs MPS (steady) | net |
|---|---|---|---|
| p201 naive 16×16 | 3.00 | 0.29× | baseline |
| p202 matrix unit | 1.65 | 0.47× | 1.82× over p201 |
| p203 staged+multi-tile | **1.02** | **~0.65×** | 1.62× over p202 |
| p204 + double-buffered | 1.48 | 0.45× | **0.69× over p203 (LESSON)** |

The optimization ladder is p201 → p202 → p203. p204 is a fork that
documents why one further "obvious" step doesn't work.

### p301 layernorm — Tier 4 opener, fusion thesis tested directly

After the Tier 3 matmul ladder closed, jumped to Tier 4 (fused
composites). Layernorm is the textbook fusion test case: row-wise
mean + variance reduction, then per-element normalize-and-affine
transform. All in one kernel.

Implementation trick used: compute sum and sumsq in **one combined
tree-reduce** carrying both arrays, then derive variance via the
algebraic identity `var = E[x²] - E[x]² = sumsq/K - (sum/K)²`.
Saves a full reduction pass over (x - mean)² at the cost of some
numerical slack (irrelevant for unit-scale randn input).

### Result and the conditional thesis

```
kernel_ms 3.36   reference_ms 3.27   speedup 0.97×
```

Essentially tied with MPS. `max_abs_err 1.43e-6` under 1e-4 atol.
Stable across two runs.

The interesting comparison is against the other "fused composite"
data point we have:

| problem | speedup | what MPS does |
|---|---|---|
| p104 softmax (Tier 2) | **1.09×** | MPS dispatches as multiple kernels |
| p301 layernorm (Tier 4) | **0.97×** | MPS already has it fused |

Both are equally common ML ops. The asymmetry isn't about how
"fusible" they are — it's about whether Apple has bothered to fuse
them in MPS. Softmax has apparently been overlooked; layernorm has
not.

**The project's "fused single-kernel beats MPS" thesis is contingent
on MPS's per-op optimization investment, not on the inherent
fusibility of the op.** This nuance matters for the Phase 4 report:
the wins are real but won't generalize uniformly across all
composites. Some Tier 4 problems will win (where Apple skipped
fusion), others will tie (where they didn't).

### Decisions made

- Used `gamma="constant", value=1.0` and `beta="zeros"` for the first
  layernorm test. Identity-like affine exercises the affine code path
  without adding error-source variance. Future problem variant could
  add randomized gamma/beta if we want to verify the affine works
  under more stress.
- Tier 4 directory created with `__init__.py`. Same glob-based
  discovery means it picks up automatically.
- Did NOT chase the remaining 3% gap. The 0.97× number is the actual
  finding — adjusting the kernel further would muddy the data point.

### p302 fused_linear_relu — Tier 4, and the warm/cold variance

Built `out = relu(x @ w + b)` as a single kernel: p202-style matmul
plus an epilogue (simdgroup_store to a TG tile, 32 threads each
process 2 of 64 elements doing bias-add + ReLU + write). Kernel is
rock-stable at 1.65ms. MPS reference, however, **varies between
1.52ms (warm) and 2.63ms (cold)** — wider warm/cold spread than any
problem we've measured so far.

Three runs:

| run | ref_ms | speedup | MPS state |
|---|---|---|---|
| 1 | 2.63 | 1.59× | cold |
| 2 | 1.52 | 0.92× | warm |
| 3 | 2.57 | 1.55× | cold-ish |

Honest steady-state read: **0.92×** — we lose by 8% when MPS is
fully warm. Honest cold-start read: **~1.55×** — we win 55% on
first call.

### Refined fusion thesis (4 data points now)

| problem | speedup | what MPS does |
|---|---|---|
| p104 softmax | 1.09× | multi-dispatch (always slow) |
| p301 layernorm | 0.97× | fully fused (always fast) |
| p302 fused_linear_relu (warm) | 0.92× | multi-dispatch (fast when warm) |
| p302 fused_linear_relu (cold) | 1.55× | multi-dispatch (slow when cold) |

Three different MPS implementation profiles surface:

1. **Always-slow multi-dispatch** (softmax) — we win uniformly.
2. **Already-fused** (layernorm) — we tie.
3. **Warm-fast / cold-slow multi-dispatch** (linear+relu) — we lose
   steady-state, win cold-start.

This is much richer than the project's initial naïve "fused beats
dispatched" framing. The benchmark catalog now distinguishes these
profiles, which means Phase 4 analysis can talk about *which kinds
of composites are worth fusing in user code* rather than asserting
a blanket rule.

### Decisions made

- Used p202's matmul (simple 8×8 per TG, no staging) rather than
  p203's optimized version, to keep the kernel focused on the
  fusion content rather than matmul-perf layering. A staged
  fused_linear_relu would be a follow-up problem if needed.
- Did NOT chase the cold-start "win" as a real speedup. The honest
  steady-state is 0.92×; cold-start is a different (also real)
  measurement.

### Harness fix — MPS compile-warmup prologue, and the speedup corrections it forces

The MPS reference-warmup friction documented across the last two
sessions finally got addressed. Root cause was visible in the code:
the existing warmup loop exits when cumulative_ms >= warmup_ms_min
(50ms), but the FIRST MPS call pays a 100–500ms compile cost on cold
cache. So the budget tripped after exactly one call — leaving the 10
timed samples in not-fully-warm state.

Fix: three untimed throwaway calls before the timed-warmup loop. Forces
shader compile and OS cache population BEFORE the budget starts ticking.

Verified on p302 (worst-affected problem): reference variance dropped
from 73% (cold/warm spread) to under 1%. Re-measured all key problems;
several historical numbers shifted:

| problem | speedup (was) | speedup (honest) | delta |
|---|---|---|---|
| p104 row_softmax | 1.09× | 1.08× | ~same (was stable) |
| p201 matmul_tiled | 0.29× | 0.23× | partially-warm ref |
| p202 matmul_simdgroup | 0.47× | 0.39× | partially-warm ref |
| p203 staged + multitile | 0.65× | 0.65× | already stable |
| p204 backfires | 0.45× | 0.45× | already stable |
| p301 layernorm | 0.97× | 0.97× | already stable |
| p302 fused_linear_relu | 0.92× / 1.55× | **0.55×** | substantial correction |

The biggest correction: p302 dropped from "0.92× warm / 1.55× cold"
to a clean 0.55× steady state. The "warm" reading we'd recorded was
itself partially warm; only the compile-warmup prologue achieves
fully-warm MPS. Implication for the fusion thesis: saving the ReLU
dispatch (~250µs) doesn't recover our matmul deficit (our p202-style
1.6ms vs MPS's 0.66ms). The fused-vs-not story collapses to "MPS
matmul is fast enough that fusion-overhead savings are second-order."

### Updated matmul ladder (honest numbers)

| problem | kernel_ms | speedup | per-step |
|---|---|---|---|
| p201 naive 16×16 | 3.01 | 0.23× | baseline |
| p202 matrix unit | 1.65 | 0.39× | 1.70× |
| p203 staged+multi-tile | 1.02 | **0.65×** | 1.67× |
| p204 backfires | 1.48 | 0.45× | 0.69× (LESSON) |

Per-step improvement ≈1.7× as I'd previously claimed. The gap to MPS
in absolute terms is wider than originally reported (we're at 0.65×
of MPS, not 0.65× *of MPS-warm-cache-recorded-as-0.77ms*).

### Updated fusion picture (honest numbers)

| problem | speedup | what MPS does |
|---|---|---|
| p104 softmax | 1.08× | multi-dispatch, slow individual ops → we win |
| p301 layernorm | 0.97× | fully fused, fast → we tie |
| p302 fused_linear_relu | 0.55× | multi-dispatch but fast matmul → we lose |

The thesis is now even more clearly conditional. With the variable
"is MPS's individual op heavily tuned" controlled for, we see:

- **MPS underinvested op** (softmax) → we win
- **MPS already fused** (layernorm) → tie
- **MPS multi-dispatch but BLAS-tuned** (linear+relu) → we lose because
  the fast individual ops outpace our slower fused matmul

### Decisions made

- Did NOT update Tier 1 and Tier 2 timing numbers retroactively in
  prior NOTES entries. The originals stand as the historical record;
  this entry is the corrigendum.
- Did NOT mirror the fix into the Swift runner. Candidate kernels
  are AOT-compiled .metallib files, not lazily-compiled MPSGraph
  shaders — they don't have the same warmup pathology.
- compile_warmup_runs = 3 as default. Empirically sufficient on the
  test set; small enough not to add measurable wall-clock overhead.

### Carry into next session

- **25 problems total** (11 Tier 1, 8 Tier 2, 4 Tier 3, 2 Tier 4).
  Three real MPS wins (p104, p106, p108). Tier 4 has two distinct
  shapes (tie, lose-on-matmul-deficit). Matmul ladder honest at
  0.65× best vs MPS.
- The honest fusion thesis: "fusion wins are real only when MPS's
  individual ops are slow." When MPS's individual ops are well-tuned
  (BLAS for matmul), our fusion overhead-savings can't recover the
  per-op deficit. Worth foregrounding in the Phase 4 report.
- Harness friction resolved. Future Tier 4 results will have
  trustworthy steady-state numbers from the first run.
- **Next Tier 4 candidates** (the place to keep pushing the thesis):
  - **p302_attention_head**: matmul + softmax + matmul, single
    kernel. MPS likely dispatches as 3 kernels (matmul, softmax,
    matmul) at minimum — biggest potential fusion win in the
    project so far. But also the most complex kernel we'd write.
  - **p302_fused_linear_relu**: matmul + bias + ReLU in one
    kernel. Simpler than attention; lets us test fusion on a small
    composite without the matmul complexity.
  - **p302_rmsnorm**: simpler cousin of layernorm (no mean
    subtraction). Likely Apple has it fused too — would be more
    evidence-gathering for the thesis nuance.
- The matmul ladder is closed; further Tier 3 work would be breadth
  (conv2d, transpose) rather than depth.
- Harness frictions STILL not addressed; bit p204 with the stability
  threshold and continue to bite the "speedup vs MPS" interpretation
  on any new op shape. Genuinely time to fix.
- Phase 3 dry-run still pending. 24 problems is a respectable test
  set for the LLM eval pipeline.

---

## 2026-06-25 — Tier 2 build-out: p103/p104/p105/p106/p107/p108 — col_sum arc, atomic experiment, and the biggest MPS win yet

---

## 2026-06-25 — Tier 2 build-out: p103/p104/p105/p106/p107/p108 — col_sum arc, atomic experiment, and the biggest MPS win yet

Added the third Tier 2 problem: column-wise sum (axis=0) of a 262144 × 256
float32 matrix. Geometry chosen deliberately as a direct translation of
p101: one threadgroup per output column, 256 threads cooperate, tree-reduce.
The naive port that exhibits the coalescing penalty on purpose.

### What's new in this problem

- **Memory coalescing.** When 32 threads of a SIMD-group all issue device
  loads at the same instant, the hardware can fuse them into one wide
  transaction if the addresses are contiguous. Stride-K addresses force
  separate transactions per thread. Apple Silicon's unified memory shrinks
  the dGPU-style gap but doesn't eliminate it.
- **Row-major storage and its consequences.** `x[b,k]` lives at index
  `b*K + k`. Adjacent threads reading adjacent columns of the same row →
  stride-1 → coalesced. Adjacent threads reading adjacent rows of the
  same column → stride-K → NOT coalesced. That single fact dictates
  whether the p101-style geometry is appropriate (row reductions) or
  actively harmful (column reductions).
- **Interleaved vs blocked work partition.** Each thread had to sum 1024
  rows. Used interleaved (`for r = tid; r < B; r += TG`) instead of
  blocked (`for r = tid * 1024; r < (tid+1) * 1024; r++`). Reason: at
  iteration 0, threads 0..255 read rows 0..255 of one column — a working
  set tight enough to hit cache with locality even though individual
  addresses are stride-K. Blocked partitioning would scatter each thread
  to a different row range, eliminating cross-thread cache reuse.

### Result and the controlled experiment

| metric | row_sum (p101) | col_sum (p103) | ratio |
|---|---|---|---|
| kernel_ms | 3.01 | 6.17 | **2.05× slower** |
| reference_ms | 1.59 | 1.77 | 1.11× slower |
| speedup vs MPS | 0.527× | 0.287× | — |
| max_abs_err | 1.14e-5 | 9.77e-4 | (both well under tol) |

The arithmetic in row_sum and col_sum is identical: 67M float adds, 256-way
tree-reduce. The ONLY difference is the memory access pattern. That makes
the 2.05× kernel slowdown a clean, controlled measurement of the coalescing
penalty — about as good an isolated experiment as you get for a hardware
effect like this.

MPS's own col-sum is only 11% slower than its row-sum, so MPS itself dodges
~85% of the penalty our naive port pays. Likely mechanism: per-tile
transpose into threadgroup memory followed by row-style reduction, or
shuffle-based block transposes.

### Decisions made

- p103 ships at 0.287× as the **naive uncoalesced baseline**, not a
  performance target. Documented in the spec description and notes that
  this is intentionally suboptimal — a teaching artifact / LLM-baseline.
- Future problem (probably p1XX_col_sum_transposed) will demonstrate the
  coalesced fix via per-tile threadgroup transpose. Useful side-by-side
  for the eventual report and a good LLM-evaluation target ("can the
  model recognize the coalescing problem from the spec and apply the
  transpose trick?").

### p104 row_softmax — multi-pass within one kernel, first Tier 2 win

Continued the session with the structural jump: row-wise softmax,
standard max-subtract stability trick, three phases (max, exp+sum,
divide) in one kernel. Same matrix shape as everything else: (262144,
256) in, (262144, 256) out — first Tier 2 problem with a non-scalar
output.

### What's new

- **Multi-pass reduction inside one kernel.** Previous Tier 2 problems
  did one reduction per kernel. Softmax needs two (max, sum-of-exp) with
  a per-element compute step between them and another after. Each pass
  reuses the same `scratch[K]` array, separated by a barrier-protected
  read of `scratch[0]` into thread-local registers.
- **Threadgroup-shared broadcast pattern (and its compiler trap).**
  First draft used `threadgroup float row_max;` as a one-element shared
  variable: `if (tid == 0) row_max = scratch[0]; barrier;`. Functionally
  correct, but Metal's compiler emits `-Wsometimes-uninitialized` —
  its per-thread flow analysis doesn't model threadgroup memory or
  barriers, so it concludes "if tid != 0 the variable is never written,
  yet it's read later." Real false positive but pollutes the harness's
  diagnostics field (which we feed to the LLM repair loop, so spurious
  warnings would invite bogus "fixes"). Switched to every-thread
  read-from-`scratch[0]` instead — same functional pattern, no warning,
  one fewer barrier per phase. Scaffold documents both patterns since
  the broadcast-scalar form is genuinely useful elsewhere.

### Result and why we finally win

| metric | row_sum (p101) | row_softmax (p104) | ratio |
|---|---|---|---|
| our kernel_ms | 3.01 | 4.06 | 1.35× heavier |
| MPS reference_ms | 1.59 | 4.44 | **2.79× heavier** |
| speedup vs MPS | 0.527× | **1.093×** | — |
| max_abs_err | 1.14e-5 | 5.96e-8 (1 ULP) | — |

The arithmetic in softmax is meaningfully more than in row_sum (extra
exp per element, an extra division, two reductions instead of one), and
our kernel reflects that — we slow by 35%. But MPS's softmax is 2.79×
slower than its sum. The most likely explanation: MPS dispatches
softmax as **three separate kernels** (max kernel, exp+sum kernel,
divide kernel), paying its per-dispatch overhead three times. Our
single-kernel fusion pays it once. That's enough to flip the comparison.

This is the first datapoint where the project's actual thesis shows
through: "MPS is great at standard ops, but loses to fused
single-kernel implementations as soon as the op decomposition has
intermediate values." Softmax is just the easiest such fusion. Worth
remembering when designing the Phase 3 LLM evaluation — kernels that
fuse multiple sub-ops are where the benchmark will most cleanly
separate competent LLMs from struggling ones.

### Decisions made

- Shipped p104 with the every-thread read-from-`scratch[0]` pattern,
  not the broadcast scalar. Cleaner diagnostics matter more than
  uniformity with the scaffold; scaffold annotates the choice.
- Tolerance set to atol=1e-5, rtol=1e-5. Observed 5.96e-8 — exactly
  1 ULP at output magnitude ~1/K. Tolerance has slack but it's
  honest given fma vs separate mul-add ordering between Metal and
  torch could shift things slightly on other inputs.

### p105 col_sum_tiled_naive — the coalescing fix that backfires

After p104, picked up the "fix p103's coalescing" thread. Wrote a (32, 8)
2D threadgroup so each SIMD-group reads 32 consecutive columns at the
same instant — by-the-book coalescing fix. Predicted 3–5× speedup vs
p103. Got 1.75× SLOWER instead (10.77ms vs 6.17ms, speedup 0.204×).

This is the most instructive surprise of the session.

### What's new (and the lesson)

- **2D thread indexing within a threadgroup.** `thread_position_in_threadgroup`
  is `uint2`; threads are addressed by `(x, y)` instead of one linear index.
  Metal linearizes threads into SIMD-groups of 32 along x-axis first, so a
  (32, 8) TG puts threads `(0..31, ty)` into one SIMD-group for each `ty`.
  Threads in a SIMD-group share `ty` and have consecutive `tx` — and if you
  arrange your address calculation so consecutive `tx` maps to consecutive
  memory addresses, you get coalesced loads.
- **The coalescing-vs-parallelism tradeoff.** Coalescing on its own isn't a
  speedup — it's *one corner* of a triangle whose other two corners are
  *total parallelism* (TGs in flight) and *per-thread work* (sequential
  iterations). p105 fixed the coalescing corner at the cost of crashing
  the parallelism corner.

### The arithmetic of the surprise

|  | p103 col_sum | p105 col_sum_tiled_naive |
|---|---|---|
| Total threadgroups | 256 | **8** |
| Threads per TG | 256 | 256 |
| Total threads in flight | 65,536 | **2,048** |
| Per-thread row work | 1,024 | 32,768 |
| Memory pattern | uncoalesced | coalesced |
| kernel_ms | 6.17 | **10.77** |

The (32, 8) tile spends 256 threads to cover 32 output columns at once,
dropping the TG count by 32×. Apple Silicon's GPU expects many TGs in
flight to hide memory and ALU latency. With only 8 TGs total, most of
the GPU's cores sit idle while the surviving threads chew through 32,768
sequential reads each. The coalescing win on each load is real and
measurable, but it cannot overcome the latency exposure from launching
2,048 threads onto hardware that wants ~10,000+.

### Decisions made

- Shipped p105 as the LESSON. Renamed the directory and entry_point to
  `p105_col_sum_tiled_naive` so the failure mode is visible at the path
  level. Documented prominently in spec description and notes that this
  is intentionally suboptimal — for LLM eval, kernels matching this
  design should be rejected, not rewarded.
- The "real" fix (more TGs per column + atomic / two-pass cross-TG
  reduction) is deferred to its own future problem rather than rolled
  into p105. Atomics on `device atomic_float` need Metal 3 and have
  their own performance characteristics worth a dedicated problem.
- Did NOT chase a working coalesced version this session — partly
  because the lesson kernel is more pedagogically valuable than the
  win would be, partly because atomics/two-pass deserves its own
  setup work in the harness.

### p106 col_sum_atomic — the real fix, and a satisfying close to the arc

Built the actually-coalesced col_sum, finishing the p103 → p105 → p106
story. Kept p105's (32, 8) thread tile for SIMD-group-wide coalescing,
but added a Y-axis dimension of threadgroups: 256 row-block TGs × 8
column-block TGs = 2048 TGs total. Each TG computes 32 per-column
partials over a ROW_CHUNK=1024 slice; thread (tx, 0) does an
atomic_fetch_add into out[col] to combine across TGs.

### What's new (this is the heaviest concept day of the session)

- **Metal 3 `atomic_float`.** Buffer is declared `device atomic_float*`
  in the kernel signature even though the host MTLBuffer is plain
  float bytes — Metal interprets the same bytes as atomic based on
  the kernel-side type. Operations live in `<metal_atomic>` (pulled
  in by `<metal_stdlib>`).
- **`atomic_fetch_add_explicit` with `memory_order_relaxed`.** Memory
  orders in Metal mirror C++ atomics. We need atomicity (no two TGs
  corrupt each other's update) but not ordering w.r.t. other memory
  ops, so `relaxed` is the cheapest correct choice.
- **Cross-threadgroup combination via atomics.** Threadgroups cannot
  synchronize within a single dispatch — there is no `grid_barrier`
  primitive. The only way for two TGs to combine values without
  spawning a second kernel is via atomic ops on device memory.

### Harness change required (and made)

Atomic accumulators require the output buffer to start at zero on
**every** dispatch — the harness's warmup + timed loop reuses one
buffer, and without per-run zeroing the second dispatch sees the
first dispatch's result and produces 2× sums, etc.

Added an optional `zero_output_each_run` flag end-to-end (spec →
Python plumbing → Swift manifest → CPU memset in dispatchOnce, before
encoding, outside the GPU timing window). Defaults to false so
existing problems are unaffected; p106 sets it true. Regression-
tested p101 post-change: kernel_ms identical, correctness identical.

This is the first time a new problem required a harness change. Worth
noting as a process pattern: when a kernel category needs new
semantics from the runner, change the runner once, then keep going.

### The full col_sum arc, side-by-side

| problem | design | kernel_ms | speedup | what it teaches |
|---|---|---|---|---|
| p103 col_sum | naive: 1 TG / col, 256 threads, uncoalesced | 6.17 | 0.287× | the cost of stride-K loads |
| p105 col_sum_tiled_naive | (32, 8) tile, only 8 TGs | 10.77 | 0.204× | coalescing without parallelism is worse |
| p106 col_sum_atomic | (32, 8) tile × 256 row blocks + atomics | **1.38** | **1.18×** | the actual fix; 4.5× over baseline, beats MPS |

The three together form a clean teaching triangle: optimizing one
corner (coalescing) without the others (parallelism, atomic
contention) can be net-negative; getting all three right beats MPS
on a problem MPS has had years to tune.

p106 at 1.38ms reading 256MB of input is ~185 GB/s throughput — the
kernel is approaching Apple Silicon's measured memory bandwidth
ceiling. Further wins on this exact problem need algorithmic moves
(smaller dtypes, sparse access, etc.), not kernel tuning.

Surprising side observation: p106 col_sum (1.38ms) is faster than
our own p101 row_sum (3.01ms) despite col_sum being "supposed to be
the harder one." Hypothesis: the high TG count (2048 vs 256) and
atomic-instead-of-tree-reduce structure of p106 saves more than it
costs vs row_sum's lower-parallelism cooperative reduction. Suggests
a future "p1XX_row_sum_atomic" experiment.

### Decisions made

- p106 ships as a real benchmark target (not a teaching artifact like
  p105). Should set the bar that LLM-generated col_sum kernels are
  measured against.
- Did NOT roll the harness change into the same commit as p106. Kept
  it as standalone infrastructure work that any future atomic kernel
  can lean on.
- `zero_output_each_run` deliberately defaults to false. Adding it
  silently would have broken the implicit "outputs aren't reset"
  contract some elementwise kernels could rely on (none currently do,
  but the conservative default protects future ones).

### p107 row_sum_atomic — the experiment, and a clean negative-ish result

The carry-forward question from p106 was: does the atomic-fan-out
pattern dominate over cooperative tree reduce in general, or was the
col_sum win specific to col_sum's hardware-unfriendly access pattern?
Tested by porting the exact same atomic structure to row_sum:

| | p101 (cooperative tree) | p107 (atomic fan-out) |
|---|---|---|
| Total TGs | 262,144 | 1,048,576 (4×) |
| Threads / TG | 256 | 64 |
| Per-TG work | full row tree-reduce (8 stages) | K_CHUNK chunk tree-reduce (6 stages) |
| Atomic ops / output | 0 | 4 |
| kernel_ms | 3.01 | **2.89** |
| Improvement | — | 4% |

Conclusion (cleaner than either an unambiguous win or loss):
**atomics-with-more-TGs dominates only when the cooperative alternative
is poorly matched to the hardware**. For col_sum, "poorly matched"
meant stride-K loads + low TG count → naive cooperative version (p103)
was at 6.17ms, easy to beat by 4.5× with atomics. For row_sum, the
cooperative version (p101) was *already* well-matched — coalesced
loads, 256 cooperating threads, 262K TGs in flight — and the atomic
variant gives only a marginal 4% improvement.

The lesson is not "atomics are better" or "atomics are not better."
The lesson is: identify whether your baseline has structural slack
the alternative pattern can exploit. If yes (col_sum), switching
pays massively. If no (row_sum), it's roughly a wash.

### Small Metal-language find

When a kernel signature uses BOTH `thread_position_in_threadgroup`
and `threadgroup_position_in_grid` attributes, they must share their
vector width — mixing `uint` and `uint2` fails to compile with
"expecting input declarations with either all scalar types or all
vector types with the same number of elements". Fix is forced:
declare both as uint2 (or whatever shape the grid wants), even if
one dimension is degenerate. Documented inline in p107's scaffold
since this is a foot-gun for LLM-generated kernels.

### p108 row_argmax — paired reduction, first non-float output, biggest MPS win yet

Wrapped the session with the first non-float problem in the project.
Same one-TG-per-row, K=256-thread cooperative tree as p101/p102, but
each scratch slot now carries a (value, index) PAIR instead of one
float — two parallel threadgroup arrays kept in lockstep. Output is
int32 (the column index of the row maximum).

### What's new

- **Paired reduction.** Two threadgroup arrays propagating together
  through the tree. At each stage, the winning value's index moves
  with it. The convention for ties: use strict `>` instead of `>=`
  so equal values keep the existing (lower-index) survivor —
  matches torch.argmax's "first maximal" rule.
- **Non-float output.** First int32 output in the project. Harness
  handled it without changes: the runner sees bytes, execute.py
  applies the dtype on read-back, verify uses np.allclose with
  atol=rtol=0 to demand exact integer equality. Confirms the dtype
  plumbing is general for future int/uint problems.

### Result — and a phenomenon worth flagging

| metric | p102 row_max (value only) | p108 row_argmax (paired) |
|---|---|---|
| kernel_ms | 3.02 | 3.09 |
| reference_ms | 1.56 | **6.66** |
| speedup vs MPS | 0.52× | **2.16×** |
| max_abs_err | 0.0 (bit-exact) | 0.0 (bit-exact int) |

Two things at once:

1. **The kernel cost is essentially unchanged** by adding index
   tracking. Same tree, same number of barriers, two arrays instead
   of one — measurement bears this out (3.09 vs 3.02). So the
   "harder" version is functionally free.

2. **MPS pays 4× more for argmax than max** (6.66 vs 1.56). Likely
   reason: MPS's argmax goes through a more general path that may
   compute both the max value AND the index (or carries larger
   auxiliary state) regardless of which the caller asked for.
   Same story as the `torch.max` vs `torch.amax` trap we hit earlier
   in the day — MPS's general-purpose paths can pay for capabilities
   the caller doesn't use.

Net effect: the speedup jumps from 0.52× to 2.16× — a **4.1×**
improvement in relative position — purely from picking a problem
where MPS's overhead is higher than its base reduction cost.
**Kernels that look harder (more bookkeeping) can actually be
easier wins for purpose-built code.** Worth documenting as a
general principle in the Phase 4 report.

### Decisions made

- Output dtype is int32, not int64. torch.argmax's default is int64
  but K=256 fits in int8; int32 halves the output buffer vs torch's
  native and matches Metal's natural integer width. Spec's reference
  casts down before comparison.
- Tolerance set to atol=0, rtol=0. With randn inputs, true ties at
  the maximum have probability zero, so the integer match has no
  slack to give. If verification ever fails on this, the kernel has
  a real bug — not a numerics issue.

### Carry into next session

- **Eight Tier 2 problems** now shipped. Speedups: 0.204× (p105) to
  **2.16× (p108)**. Three real MPS wins (p104 softmax, p106
  col_sum_atomic, p108 row_argmax), two teaching artifacts (p103/p105
  col_sum lessons), three baselines (p101/p102/p107).
- The col_sum + row_sum atomic experiments together suggest a useful
  diagnostic principle to flag in the eventual report:
  **"the speedup from switching reduction idioms equals the structural
  slack of the baseline."** Worth testing on a Tier 3 problem to see
  if it holds with more arithmetic intensity.
- Next problem candidates: row_l2_norm (gentler, fills out the
  reduction taxonomy without new concepts), row_argmax (paired
  reduction, first non-float output), row_var/row_std (two-pass with
  intermediate — pairs structurally with softmax), or graduate to
  Tier 3 (tiled matrix ops — adds shared-memory tile-loading
  patterns).
- Open harness items: tempfile cleanup in run_problem.py:34,
  timing_noisy threshold tuning for short kernels, CPU-torch reference
  timing dynamics. Nothing blocking; queue for a focused maintenance
  session.

---

## 2026-06-23 — Tier 2 opener: p101 row_sum, p102 row_max, and the torch.max trap

Opened Tier 2 with the canonical learning reduction: row-wise sum of a
262144 × 256 float32 matrix, one threadgroup per row, K=256 threads
cooperating via threadgroup-shared memory and a tree reduction.

### What's new in this problem (concepts that didn't appear in Tier 1)

- **Threadgroup-shared memory** — `threadgroup float scratch[K]` allocates
  a per-group scratchpad visible to every thread in the group, much faster
  than device memory, invisible to other groups. Tier 1 never needed it
  because elementwise threads don't cooperate.
- **Barriers** — `threadgroup_barrier(mem_flags::mem_threadgroup)` is the
  synchronization point. Threads in a group don't run in lockstep (they're
  scheduled in SIMD-groups of 32 on Apple GPUs), so without barriers a
  later stage can read scratch slots before the earlier stage's writes
  land. The `mem_threadgroup` flag scopes the visibility guarantee to
  threadgroup memory, which is cheaper than `mem_device`.
- **Tree reduction** — log₂(K)=8 halving-stride stages, active threads
  halving each stage. The big "gotcha" is that the barrier MUST be
  outside the `if (tid < stride)` guard — every thread in the group has
  to hit every barrier, including the idle ones, or the behavior is
  undefined.

### Result

| metric | value |
|---|---|
| compiled | true |
| correct | true (max_abs_err 1.14e-5, atol 1e-4) |
| kernel_ms | 3.01 |
| reference_ms | 1.59 |
| speedup | **0.527× — we lose to MPS** |
| A/B/A delta | 0.25% (timing trustworthy) |
| timing_noisy | true (IQR > 15%) |

First problem in the entire project where the candidate kernel doesn't
beat MPS. Every Tier 1 elementwise problem won (1.10×–2.49×) because
MPS's per-dispatch overhead dominates trivial ops. Reductions flip that:
there's real shared algorithmic work, so an MPS-optimized kernel that
exploits SIMD-group primitives wins.

### Why the gap exists (left unfixed on purpose)

Three suspected contributors:

1. **No SIMD shuffles.** Apple GPUs run threads in SIMD-groups of 32.
   Once `stride <= 32`, the surviving threads are all in the same
   SIMD-group, and you can swap data via `simd_shuffle_down(value,
   offset)` directly between registers — no scratch, no barrier. MPS
   almost certainly does this for the last 5 stages. Ours uses scratch +
   barrier for all 8.
2. **One element of useful work per thread.** Each thread loads 1 float,
   does 1 add per stage, then mostly idles. MPS likely has each thread
   sum N inputs first (coalesced strided loads), amortizing the 8
   barriers across more useful arithmetic.
3. **Late-stage thread idleness.** At stride=1, 1 thread does work and
   255 wait on the barrier. Inherent to the tree pattern but a real cost
   on a wide threadgroup.

Did NOT optimize because the *point* of this kernel is to be a clean,
honest baseline that LLM-generated kernels will be asked to beat. If we
tune it ourselves, we're tuning the target away from where we want it.

### Decisions made

- p101 ships at 0.53× as the Tier 2 baseline reference, not as a
  performance achievement. Scaffold (`row_sum_scaffold.metal`)
  preserved separately so future readers see what was given vs. what was
  written.
- Confirmed the harness's CPU-torch reference is fine here despite the
  open concern from the previous session — atol of 1e-4 absorbs the
  CPU-sum vs tree-reduction summation-order divergence cleanly
  (observed 1.14e-5).

### p102 row_max — same pattern, different op, and a fairness trap

Followed p101 with the natural sibling: row-wise max, same launch
geometry, same tree reduction, swap `+=` for `max()`. Result mirrored
p101 almost exactly — kernel 3.03ms, reference 1.56ms, speedup 0.517×.
The structural prediction held: our tree-reduction kernel costs the
same regardless of combining op (~3.0ms), and MPS's optimized
reductions also cost the same for sum vs amax (~1.6ms).

`max_abs_err` was **exactly 0.0**, which is the theoretical prediction:
unlike `+`, the `max` operator is associative AND commutative for
non-NaN floats, so any reduction order returns the same bit pattern.
Confirmed empirically; tightened tolerance to 1e-6 for paranoia.

### The torch.max vs torch.amax trap

The first run of p102 used `torch.max(x, dim=1).values` as the
reference and reported `speedup: 2.876×` — almost 3× over MPS, after
losing 2× on p101. That gap was suspicious given the kernels are
structurally identical.

Root cause: `torch.max(x, dim=1)` returns a named tuple
`(values, indices)`. Even when you immediately discard `.indices`, the
MPS implementation has already computed the argmax. So we were
comparing "max + argmax" (MPS) against "max only" (our kernel) —
inflating reference time roughly 5×. Switching to `torch.amax`, which
returns values only, dropped reference_ms from 8.69 to 1.56 and the
speedup from 2.88× to 0.52× — consistent with p101.

Documented inline in `p102_row_max/spec.py` so future readers (and the
LLM-prompt generator that reads spec descriptions) understand the
choice. **General principle for reduction problems**: when the natural
torch API returns auxiliary state (indices, second-largest, etc.),
use the values-only variant for the reference, or you're benchmarking
the convenience-API surcharge rather than the op.

### Carry into next session

- `timing_noisy: true` is now firing on the 3ms reductions even when
  A/B/A is rock-solid. The current IQR threshold (15% of median) may be
  too tight for short kernels — worth revisiting once a few more
  reductions land, not now.
- The SIMD-shuffle vs scratch tradeoff is a candidate for an instructor
  problem later in Tier 2 ("p1XX_row_sum_simd") if we want a
  side-by-side that exposes the optimization to students/LLMs.
- Next reduction candidates: col_sum (introduces memory coalescing),
  row_l2_norm (pre-reduction transform), row_softmax (multi-pass).

---

## 2026-06-22 — Tier 1 fill-out: p007–p012, and the first-run MPS-compilation gotcha

Pushed Tier 1 from two confirmed problems (p001, p002) to ten confirmed
problems (p001–p008, p010, p012). Skipped p009 GELU when MSL turned out to
have no `erf` — deferred to a focused future session on polynomial erf as
its own numerics task, rather than letting it derail this batch.

### What got built

| ID | Op | Steady speedup | Bucket |
|---|---|---|---|
| p007 | sigmoid | 1.13× | low (MPS fuses) |
| p008 | tanh | 1.15× | low (MPS fuses) |
| ~~p009~~ | ~~gelu~~ | deferred | — |
| p010 | abs | 1.20× | low (MPS fuses, single ALU op) |
| p011 | exp | 1.13× | low (MPS fuses) |
| p012 | clamp | 1.16× | low (MPS fuses) |

Also verified end-to-end p003–p006 which had specs from the prior commit
but no calibration. All four are now sane: p003 1.10×, p004 1.15×, p005
1.12×, p006 axpby 2.49× (the one big-speedup problem of the entire tier —
real fusion win because torch eager dispatches `a*x + b*y` as three
separate MPS kernels).

### The headline finding: MPS lazily compiles MPSGraph shaders on the first
### (op, shape) invocation, and our warmup loop doesn't amortize it.

What I observed mid-session: re-running `p011_exp` in a fresh process gave
**1.13× speedup, vs. 1.90× on the first run**. Same kernel, same reference,
same shape, same machine — only difference was that the first run was the
first time MPS had ever compiled `exp` at shape `(2^25,)` on this machine.
Reproduced on sigmoid (1.33× → 1.13×) and clamp (1.86× → 1.16×). Did NOT
reproduce on axpby (2.44× → 2.49×, so its big speedup is real fusion, not
a compilation artifact).

Diagnosis: MPS compiles graph shaders lazily, and the compilation cost is
borne by the *first dispatch*, not amortized across the warmup loop. Even
worse, the cache survives across processes (lives in
`~/Library/Caches/com.apple.metal/...` at the OS level), so once a shape+op
combo is compiled on a machine, **every subsequent benchmark run sees the
fast path**. The first time you measure a new problem, you're measuring
compile-time + run-time; every time after, you're measuring run-time only.

Implications:
- The committed speedup numbers for p007 (1.33×) and p011 (1.898×) are
  inflated by this artifact. Steady-state values are both ~1.13×. Spec
  descriptions and commit messages aren't being amended (per project
  practice), but this log is the source of truth for the corrected numbers.
- axpby's 2.49× is real and stays.
- The benchmark's mental model is now simpler and cleaner than I thought:
  **MPS has a fused single-op kernel for every standard elementwise op**.
  Hand-written kernels only win by a small margin on those (dispatch
  overhead). The big wins come from expression-level fusion across
  primitives (axpby) — that's the interesting territory for an LLM.

### Harness gap to address before Phase 3 / Tier 2

The fix needs to be one of:
1. Add a "prime" run before warmup — dispatch the reference once, discard
   timing, then start the real warmup. Populates the OS cache cheaply.
2. Make `--calibrate` only succeed if the result agrees with a re-run
   within X% (forces the user to take a second measurement).
3. Detect the situation post-hoc: if reference_ms differs from the
   calibrated baseline by >30% in *the same direction every time*, warn
   "did you re-run after spec change?"

Option 1 is the cheapest and most general. Carrying as the first item on
the Phase 3 todo.

### The .item() trap

Sub-finding from clamp: my first reference passed `min=lo.item(), max=hi.item()`
inside the timing loop. `.item()` on an MPS-resident tensor forces a
GPU→CPU sync per call, inflating reference_ms by ~0.4ms (28%) — pure
measurement artifact. Fixed by following p005 leaky_relu's pattern: the
lo/hi tensors are bound for the kernel as buffers, but the reference uses
the same Python constants directly. Lesson: any `.item()`, `.tolist()`, or
device→host transfer inside a reference function poisons the timing.
Adding this to the LLM CONVENTIONS prompt should help future generated
references avoid the same mistake.

### Smaller observations

- **Cold-start A/B/A failure.** First GPU dispatch of a session can trip
  the 7% block1-vs-block3 threshold because the M2 Pro's clock hasn't
  ramped. Workaround: discard the first run of any session. A real fix
  would be a clock-stabilization gate in warmup.
- **`timing_noisy` correlates with per-element compute, not absolute
  kernel time.** sigmoid, tanh, exp (all transcendentals) tripped the 15%
  IQR threshold; abs, clamp (no transcendental) did not, despite running
  in the same ~1.4ms. Suggests transcendental table-lookup adds
  shot-noise that flat ALU ops don't. The threshold might want to be
  absolute, not percentage — open question.
- **`max_abs_err` is misleading for ops with wide output ranges.** exp at
  x≈5.7 has output ≈298, where 1 ULP ≈ 3.5e-5. The verify gate already
  handles this via combined atol + rtol, but the headline number
  (4.58e-5) looks scary without that context. Worth surfacing in the
  result JSON: "max_ulp_err" alongside "max_abs_err".

### Concept introduced this session: MSL math library is smaller than libm

The Metal Shading Language ships a leaner math library than libm. Found
out the hard way: `erf` and `erfc` are absent. `exp`, `log`, `tanh`,
`sin`, `cos`, `pow`, and the usual suspects are present. There's also a
`metal::fast::` namespace with ~2-3× faster variants of the transcendentals
at lower accuracy (omitted for benchmark fairness — torch uses precise
versions on MPS).

The takeaway for spec design: when picking new elementwise problems,
**check the MSL headers first** at `/private/var/run/com.apple.security.cryptexd/.../metal_math`
(the path can be found via `find / -name 'metal_math' 2>/dev/null`) before
committing to a formula. Saves a compile failure mid-stride.

### Decisions made

- p009 GELU deferred to a dedicated session. Either Abramowitz-Stegun
  polynomial or full Chebyshev; the goal will be implementing a libm-grade
  transcendental from scratch, not just adding GELU coverage.
- p012 clamp constants: `lo = -1.0, hi = 1.0` (constant init, not uniform).
  Reasoning: uniform init could produce `lo > hi` (degenerate) by chance;
  constants ensure ~32% of randn x is actually clamped, so the operation
  does meaningful work.
- Calibration JSON (`results/calibration.json`) confirmed correctly
  gitignored — it's per-machine timing state, not source. Don't try to
  commit it again.

### Open items for next session

- Implement the "prime run" warmup primer (Harness gap above).
- Re-measure p001–p012 in one batch with the primer in place to get a
  clean steady-state speedup table — current numbers are partly inflated
  by first-run compilation.
- Consider whether to extend Tier 1 with silu/swish, softplus, log, sqrt,
  rsqrt, or pivot to Tier 2 reductions.

---

## 2026-06-15 — Phase 2: decoupling N from kernel files, decoupling warmup from iteration count

Shipped three pieces that closed Phase 1's two named open items and unblocked
Tier 1 sizing: (1) launch config (grid + threadgroup) migrated out of `.metal`
files into `mkb/problems.launch_config()` with sensible defaults derived from
the output shape, (2) Tier 1 problem size bumped from N=2^20 (1M) to N=2^25
(32M) so MPS reference times comfortably exceed 1 ms, (3) warmup replaced from
count-based ("3 iterations") to time-based ("dispatch until cumulative GPU
time ≥ 50 ms") on both the Swift runner and Python `time_reference_mps`.

p001 and p002 calibration baselines re-recorded at the new N. A/B/A
`block_delta_frac` dropped from 8.7% / 14.8% (failing the 7% gate) to 0.2%
/ 3.4% (comfortably passing). Speedup numbers collapsed from Phase 1's
fictional 7.3× to an honest 1.1–1.2× — the bandwidth-bound elementwise
kernels modestly beat MPSGraph but no longer hide behind dispatch overhead.

### Lesson: a coupling stays invisible until the shared variable moves

The `MKB_GRID`/`MKB_TG` magic comments in golden kernels encoded N (the
problem size) inside the kernel file. This worked fine — nothing ever
changed N. The moment Phase 2 tried to bump N from 1M to 32M, the coupling
produced a silent-wrong-answer bug: change spec.py to N=32M, forget the
kernel comment, and the dispatch is 1M threads against a 32M buffer —
only the first 1M elements get written. Verify catches the difference,
but only by luck; if the test had checked compile + dispatch without
result comparison, it would have looked superficially fine.

The fix was structural, not editorial: instead of "remember to update both
files," move N's ownership entirely to spec.py and derive grid in the
harness via `launch_config()`. Goldens become pure algorithm with no
N-specific data — the CLAUDE.md "never modify golden_kernels" rule got
stronger as a side effect, because there's now nothing in those files to
drift in the first place.

**Principle:** couplings between files reveal themselves only when one of
the shared values changes. When you find one, the question to ask is "who
*owns* this value?" The owner stores it; everyone else derives or receives
it. Don't fix coupling bugs with discipline ("just remember to update
both"); fix them by removing the duplicate ownership.

### Lesson: measure the property you want, not a proxy for it

Phase 1 ended with two named open items. Both were the same shape:

- (1) Time-based warmup, because "3 iterations" was a proxy for "GPU
  clocks are awake" — and the proxy fails when iteration time changes.
- (2) Reference ≥1 ms, because the existing N was a proxy for "reference
  is GPU-bound" — and the proxy fails when MPS dispatch overhead
  dominates small kernels.

Open item (1) was non-blocking on day-of-Phase-1 because the kernel size
hadn't moved. The instant Phase 2 moved N to 32M, the warmup proxy broke
catastrophically (block 3 ran 14.8% *faster* than block 1 — clock
ramping, not thermal degradation). Filed → fixed in the same session.

The principle that solves both: **stop coding the proxy, code the
property.** "Until cumulative GPU time ≥ 50 ms" measures the actual ramp
condition; it self-adjusts to any kernel size from 50 µs to 100 ms per
dispatch with the same code. "Reference is sized so it takes ≥1 ms"
measures the actual overhead-vs-work tradeoff; it adapts as MPS
orchestration costs change. Proxies make the failure invisible until the
underlying variable shifts; property-based measurements stay correct
across changes.

### Lesson: honest speedups need GPU-work-dominated reference times

At N=1M, MPS reference was ~0.5 ms — but ~0.3 ms of that was Python+MPS
orchestration overhead and only ~0.2 ms was GPU work. A hand-written
kernel running on a dedicated Swift runner trivially "wins" by 7× because
it skips the orchestration cost — not because it does the math better.
Useless signal: every kernel an LLM ever writes would look "much faster
than MPS" on tiny problems.

At N=32M, MPS reference is ~2.4 ms. Of that, ~0.2 ms is overhead
(unchanged) and ~2.2 ms is GPU work (30× more). Overhead drops from ~60%
of the measurement to ~8%. The speedup ratio now compares candidate-GPU
vs MPSGraph-GPU and means something. The 1.1–1.2× numbers are the real
signal: we're competing on equal terms with MPSGraph on memory-bound ops
and modestly winning.

**Carry forward to Tier 2/3 authoring:** every spec must be sized so its
reference takes ≥1 ms. This is *not* a global N — for reductions the
right size will differ from elementwise, for tiled matmul different
again. The rule is "reference ≥1 ms," not "shape ≥ X."

### Concept: GPU clock ramping, revisited with the fix

Phase 1 explained the symptom (Apple Silicon GPUs idle at low clock,
take 10–100 ms of sustained load to ramp up). Phase 2 fixed the
consequence: warmup must be measured in *GPU time*, not iteration count,
because the only thing that wakes the clocks is sustained load — and
"sustained" is a time threshold, not a count threshold. Three iterations
of a 50 µs kernel buys 150 µs of load (nowhere near ramp); three of a
10 ms kernel buys 30 ms (close but not enough); three of a 100 ms kernel
buys 300 ms (overkill). The same constant ("3") gives wildly different
warmup quality at different kernel scales. Time-based warmup gives the
*same* quality (≥50 ms of GPU work, by construction) at every scale.

### Concept: two timers, kept where they belong

The candidate side uses GPU hardware timestamps (`cmd.gpuStartTime` /
`cmd.gpuEndTime`) inside Swift — pure GPU execution time, no
orchestration noise. The reference side uses CPU wall-clock
(`time.perf_counter()` brackets around `reference_fn(...)` with explicit
`torch.mps.synchronize()` barriers) — wall-clock measurement, includes
PyTorch+MPS orchestration *and* GPU work.

Asymmetric on purpose: we use the best available timer for each side.
We can't get per-op GPU timestamps from PyTorch's MPS backend (black
box), so reference is wall-clock by necessity. We could move candidate
to wall-clock too, but it would re-introduce ~100s of µs of Swift/IPC
overhead per dispatch — the very thing the runner was built to exclude.

The N=32M fix doesn't unify the timers; it sizes the problem so GPU
work dominates wall-clock on the reference side (~93% of the
measurement), making the two timers measure things that are *close
enough to* apples-to-apples that the ratio means kernel quality.

### Decisions made

- **Tier 1 global N = 2^25 (33,554,432).** Chosen by measuring relu (the
  lightest Tier 1 op) at N ∈ {1M, 2M, 4M, 8M, 16M} on MPS and seeing relu
  cross 1 ms only at 16M with no margin. 32M extrapolates to ~2 ms —
  comfortable margin, well within M2 Pro unified memory (3 × 128 MB
  buffers per vector_add invocation).
- **Launch override in spec is partial-allowed** — spec can declare just
  `grid`, just `threadgroup`, or both. Either omitted field uses the
  default. Simpler than all-or-nothing and matches the common case of
  Tier 2 problems that override grid geometry but keep TG=256.
- **Default threadgroup = (256, 1, 1)** for 1D dispatch. Apple's
  hardware-optimal sub-multiples are 32 (one SIMD group) and multiples
  thereof; 256 is the standard "fits any Apple GPU since A12" choice.
- **Three warmup exit conditions, AND-joined.** Floor (50 ms) is the
  goal; ceiling (500 ms) caps wall-clock cost during sweeps; iter cap
  (10,000) prevents infinite loop on a kernel whose GPU timer returns 0.
  Floor fires under normal operation; the other two are guardrails.
- **LLM prompt updated to not declare MKB_GRID.** Convention #4 in
  `mkb/llm/generate.py` now says "launch config is owned by the harness,
  do NOT declare it in the kernel file." Phase 3 prompt will need to
  surface the spec's launch override to the model for Tier 2+ problems
  (currently doesn't — known gap, see open items).
- **Swift warmup loop was a user-authored learning slice.** Python
  mirror was plumbing (Claude wrote it). Splitting that way kept the
  learning content in the conceptually-interesting language and avoided
  re-doing the same loop in two places — the Swift work taught the
  pattern; the Python mirror just applied it.

### Open items (carry into next session)

- **`build_prompt` does not surface launch override to the model.** For
  Tier 1 (default launch), the model can infer grid from the output
  shape stated in the prompt. For Tier 2+ where spec overrides launch
  (e.g., `p101_row_sum` with `grid=(B,1,1) tg=(K,1,1)`), the model
  needs to know that geometry to write correct index math. Required
  before any LLM sweep over Tier 2.
- **Tier 1 problems p003–p008 not yet authored.** Six elementwise
  problems on the punch list (elementwise_mul, scalar_mul, leaky_relu,
  saxpy, sigmoid, gelu) plus their golden kernels. Plumbing — ship in
  one batch.
- **`p101_row_sum` is the Tier 2 first reduction.** Hand-off planned:
  Claude writes spec + scaffold with decision points marked, user
  writes the kernel by hand as a learning slice.

### Resolved this session (no longer open)

- **Phase 1 open item (1) — time-based warmup.** Resolved.
- **Phase 1 open item (2) — Tier 1 sizing for ≥1 ms reference.** Resolved.

### Carried forward from earlier sessions

- `tempfile.mkdtemp(prefix="mkb_build_")` in `scripts/run_problem.py:40`
  still leaks. Per-problem, minor.
- Correctness reference still on CPU torch (`run_problem.py`) — will
  bite on reduction-order problems (Tier 2 territory).
- `metal-kernelbench-plan.md` still missing from repo despite CLAUDE.md
  reference.

---

## 2026-06-12 — Phase 1 timing trust (calibration discipline + A/B/A)

Shipped both sub-tasks: metadata-aware calibration baselines (so drift checks
fail loud when the environment, not just the temperature, has shifted) and
A/B/A interleaved timing per variant (c) — measure candidate, then reference,
then candidate again, and refuse to report a speedup if the two candidate
blocks disagree by >7%. The reference block in between has provably been
measured under whatever conditions held across the bracketing candidate
blocks; if the candidate drifts across the window, the reference number is
suspect by inference.

22/22 tests green, 8 new (5 for `check_stability`, 7 for calibration metadata).
Calibration baseline re-recorded under the new schema.

### Lesson: A/B/A catches more than thermal drift

The original framing was "detect thermal throttling during measurement." On
the very first slice run after implementing it, A/B/A fired with block 3
**faster** than block 1 by 36% — the opposite of thermal degradation. Real
cause: GPU clock ramping. Apple Silicon GPUs idle at low clock and only ramp
up under sustained load; 3 dispatches of a 50 µs kernel total 150 µs of GPU
work, nowhere near long enough to wake the clocks. So block 1 was measured
mid-ramp and block 3 at steady state.

The error message had to grow up: it now reports *direction* (block 3
slower vs faster) and points at the likely cause for each direction. Either
form of disagreement means the speedup ratio is untrustworthy, but the
remediation differs (cool the machine vs. bump warmup).

**Principle:** when a check fires for an unexpected reason, don't tighten
or loosen its threshold first — read what it's actually detecting. A/B/A
turned out to be a more general "machine state shifted during measurement"
detector than I designed it to be. That's a feature, not a bug, as long as
the error message names the actual finding.

### Lesson: don't tune thresholds from one session

The 7% stability threshold flagged ~25% of runs as untrustworthy on `make
slice`. Tempting to loosen — instead, leave it. Single-session noise is the
worst possible data for picking a threshold; we'll have a real distribution
to look at after Phase 2 sweeps run dozens of problems. Premature tuning
hides the signal we're trying to learn from.

### Concept: GPU clock ramping (a.k.a. why warmup matters)

Apple Silicon GPUs (and most modern GPUs) have dynamic frequency scaling:
the device idles at a low clock to save power and ramps up only when the
power manager sees sustained load. Ramp time is in the 10–100 ms range on
M-series. This has a sharp consequence for benchmarking tiny kernels:
**a few iterations of a microsecond-scale kernel cannot wake the GPU
clocks**. You'll measure the kernel mid-ramp, where each dispatch is on a
slightly faster clock than the last, and your "median" becomes a mix of
several clock states. The fix is time-based warmup: dispatch until the
*cumulative* GPU time hits some target (50–100 ms), so the device is
provably out of its low-power state before timing starts. Iteration-count
warmup can't solve this — for a 50 µs kernel, you'd need 1000+ iterations
to total 50 ms, at which point the warmup budget is naturally expressed in
time anyway. Filed as open item (1) below.

### Concept: tiny-problem speedups measure dispatch overhead, not kernels

Quick BOM on `vector_add` at N=1M (~the current Tier 1 size): M2 Pro memory
bandwidth is ~200 GB/s; the kernel reads 8 MB and writes 4 MB, so the
bandwidth-bound floor is ~60 µs. Observed `kernel_ms` is ~72 µs — meaning
the hand-written kernel is *already near optimal*. The PyTorch MPS
reference, meanwhile, comes in at ~530 µs. That ~7× speedup isn't telling
us "the candidate did the math better"; it's telling us "the candidate
avoided MPS's per-call orchestration overhead, which dominates total time
at this problem size." A model that wrote this kernel and a model that
wrote one half as fast would both come out "much faster than MPS." Useless
signal. The fix is sizing problems so the reference takes ≥1 ms — at that
point reference time is dominated by GPU work, and speedup measures kernel
quality vs MPSGraph quality, which is what we actually care about. Filed
as open item (2) below.

### Decisions made

- **Sub-task 1: no migration code.** The pre-Phase-1 `calibration.json`
  format (a flat `{kernel_id: median_ms}` dict) is detected via missing
  `schema_version` and triggers a re-record prompt rather than being
  silently upgraded. Cheaper than writing migration logic that runs once
  per machine ever.
- **Sub-task 2: variant (c) over (a) per-pair or (b) runner-level
  interleave.** (c) is detection-not-prevention — we don't fight to make
  every session fair, we just flag the unfair ones and force a retry.
  ~20 lines of orchestration logic, no Swift-side changes. Justification
  for upgrading to (b) later: persistent flag rates in Phase 2 sweeps
  would be evidence we need it.
- **Threshold (7%) chosen blind**, hoisted to a named constant
  (`STABILITY_THRESHOLD_FRAC` in `mkb/timing.py`) so it's tunable. Will
  re-evaluate after Phase 2 gives us a real distribution.
- **Wrong-answer kernels short-circuit after block 1.** No point timing a
  kernel we already know is wrong — saves GPU work and avoids spurious
  instability flags from a single timing block.
- **On A/B/A fail during `--calibrate`: refuse to record.** Better to force
  the user to retry on stable conditions than to corrupt every future
  drift check with an untrustworthy baseline.

### Open items (carry into next session)

- **(1) Time-based warmup.** Replace count-based warmup (currently 3
  iterations in the Swift runner) with "dispatch until cumulative GPU time
  ≥ ~50–100 ms, then start timing." Implementation lives in the Swift
  runner — it already reads `gpuEndTime - gpuStartTime` per dispatch and
  can loop until the budget is hit. Cap with a min/max bound (e.g. 50 ms
  floor, 500 ms ceiling) so multi-second kernels don't blow past the
  budget on a single iteration. Root cause: GPU clock ramping; iteration
  bumps can't fix it.
- **(2) Tier 1 problem sizing.** Spec problems such that the MPS reference
  takes ≥1 ms (median). At N=1M for `vector_add`, the candidate is already
  bandwidth-bound near-optimal and the reference is dominated by MPS
  dispatch overhead, so "speedup" measures overhead rather than kernel
  quality. Phase 2 spec language: "Tier 1 problems must be sized so the
  reference op takes ≥1 ms," not a global N. Probably ~16M elements for
  `vector_add`, varies per-problem.

Both (1) and (2) are to be filed as issues, not implemented yet. (1) is a
Swift+manifest change with cross-runner implications. (2) is a spec change
that will reshape Phase 2's problem authoring.

### Carried forward from earlier sessions

- `tempfile.mkdtemp(prefix="mkb_build_")` in `scripts/run_problem.py:34`
  still leaks. Now it leaks per-problem instead of per-run; minor.
- Correctness reference still on CPU torch; will bite on reduction-order
  problems.
- `metal-kernelbench-plan.md` still missing from repo despite CLAUDE.md
  reference.

---

## 2026-06-11 — Phase 0 vertical slice green

Got `make slice` and `make test-mac` (11/11) passing end-to-end on a fresh
Mac. Harness correctly accepts golden kernels, rejects wrong-answer kernels,
and surfaces `xcrun metal` diagnostics for non-compiling ones — i.e., both
directions of the Phase 0 exit criterion satisfied.

### What broke and how it got fixed

1. **Missing Metal toolchain.** Xcode 26.5 installs the `metal` compiler but
   not `metallib` (the linker) — they used to ship together, now `metallib`
   is part of an on-demand "Metal Toolchain" component. Fixed with
   `xcodebuild -downloadComponent MetalToolchain` (~688 MB). Future-Erika:
   if you see `xcrun: error: unable to find utility "metallib"` on a fresh
   Mac, that's the command.
2. **`pyproject.toml` had no package declaration.** Setuptools refused to
   auto-discover among four top-level dirs (`mkb/`, `problems/`, `results/`,
   `runner/`). Fixed by adding `[tool.setuptools] packages = ["mkb"]` —
   `problems/` is loaded via `importlib.util.spec_from_file_location` at
   runtime, not as an installed package, so it doesn't belong in the list.
3. **Python env: no `python` binary, no torch, Homebrew default was 3.14.**
   Solved by creating `.venv` from `python3.12` (torch 2.12 has reliable
   wheels for 3.12; skipped 3.14 as too new). Venv always provides `python`
   so the Makefile's bare `python scripts/...` line resolves inside it.

### Lesson: never `try?` something whose error you'd want to see

The Swift runner originally had:

```swift
guard let manifestData = try? Data(contentsOf: manifestURL),
      let m = try? JSONDecoder().decode(Manifest.self, ...) else {
    fail("could not read or parse manifest")
}
```

`try?` silently converts any thrown error into `nil`. If `JSONDecoder` had
choked on a missing field or a type mismatch, the actual `DecodingError`
(which includes the exact `CodingKey` path that failed) would have been
discarded — leaving us to guess at JSON keys from a flat "could not read or
parse manifest" message.

Patched preemptively to a `do/catch` block that interpolates the caught
error into `fail("...: \(error)")`. Never bit us in this session, but it's
the cheap kind of insurance: ~5 lines now saves an unbounded debugging cost
the first time the manifest format drifts.

**Principle to carry forward:** `try?` is only appropriate when the failure
itself is the signal and the error genuinely doesn't matter (e.g., "try to
read a cache file; if it's missing or corrupt, fall through to the slow
path"). For anything where a failure would require debugging — *use
`do/catch` and surface the error*.

### Concept: GPU timing is harder than CPU timing because there are two clocks

CPU timing is easy because there's only one timeline: `t0 = now(); foo();
t1 = now()` works because `foo()` blocks the CPU until done. GPU timing is
hard because `cmd.commit()` is **asynchronous** — it hands the command
buffer to the driver and returns to the CPU immediately, while the GPU is
still working.

`cmd.waitUntilCompleted()` is a barrier: "block this CPU thread until the
GPU finishes." After that barrier, the GPU's own clocks (`cmd.gpuStartTime`,
`cmd.gpuEndTime`, populated by the driver when the GPU actually starts and
finishes) are safe to read. Without the barrier, those properties may be
zero or carry stale values from a previous use — giving you a duration of
`0.0 ms`, which looks fine until you realize every kernel "takes zero ms."

Same problem, same fix on the PyTorch side: `time_reference_mps` in
`mkb/timing.py:53` calls `torch.mps.synchronize()` before stopping its
CPU-side timer. Without that sync, you'd time how long PyTorch took to
queue the work on the GPU, not how long the work actually ran.

**One-sentence frame to remember:** GPU timing is harder than CPU timing
because there are two clocks running in parallel, and you have to explicitly
synchronize before reading either one.

### Concept: `kernel_ms` and `reference_ms` aren't apples-to-apples

The headline 7.287× speedup in `make slice` output deserves a footnote.

- `kernel_ms` is read from `cmd.gpuStartTime`/`gpuEndTime` — **pure GPU
  execution time** for the dispatch.
- `reference_ms` is read from `time.perf_counter()` brackets around the
  PyTorch call — **CPU wall-clock**, which includes Python overhead,
  MPSGraph's op-fusion machinery, *and* the GPU time.

So part of the 7× is real (a hand-written kernel doing one fused
load-add-store really is faster than MPSGraph for tiny ops), but part is
just MPS's per-dispatch overhead being amortized poorly at 1M elements.
Worth keeping in mind when comparing LLM-generated kernels later: a "10×
speedup" on a single tiny op may flatter the kernel by hiding MPS overhead
rather than reflecting better GPU work. The honest comparison would put
both numbers on the same clock — either both wall-clock or both GPU-only —
and the harness currently doesn't.

### First-run failures vs. what I predicted

I predicted three before running anything: (1) Metal toolchain missing,
(2) Swift Codable swallowing errors, (3) SwiftPM symlink path mismatch.

- **(1)** hit hard, plus one variant I didn't predict: `metal` installed
  but `metallib` missing (separate Toolchain component in Xcode 26+).
- **(2)** patched preemptively, never bit. Still worth the patch.
- **(3)** didn't bite — SPM created both `.build/release/Runner` and the
  triple-prefixed path on Apple Silicon.

What I didn't predict at all: Python environment setup was its own
multi-step diagnosis chain (no `python` binary, no torch installed,
setuptools refusing to auto-discover packages). Worth remembering — Python
toolchain is its own debugging surface, not just "pip install" wallpaper
over a working setup.

### Decisions made

- Project moved from `~/Downloads/metal-kernelbench` into `~/MSL-Bench`
  (the git-tracked repo with the GitHub remote at
  `github.com/erika-goh/MSL-Bench`).
- Toolchain choice: full Xcode + Metal Toolchain (Option A), rather than
  standalone Metal Developer Tools (B) or runtime Swift compilation (C).
  Reason: this is a learning project, and Xcode's GPU debugger / Frame
  Capture / Instruments are tools we want by Phase 1. The 10–15 GB cost is
  a one-time tax.

### Open items (carry into next session)

- `tempfile.mkdtemp(prefix="mkb_build_")` in `scripts/run_problem.py:34`
  is never cleaned up — disk leak, low urgency.
- Correctness reference is built on CPU torch (`run_problem.py:55`) while
  timing uses MPS. Fine for `a+b`; will bite later on ops where MPS and
  CPU diverge in floating-point ordering (sums, means, reductions in
  general).
- `compile_metal` returns only stage-1 (`metal -c`) stderr as diagnostics —
  if a future kernel produces stage-2 (`metallib`) warnings, they'd be
  dropped silently. Not urgent until we see it happen.
- No `metal-kernelbench-plan.md` exists in the repo despite CLAUDE.md
  referencing it. Either the plan should be written or the CLAUDE.md
  reference removed.
