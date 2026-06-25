# MSL-Bench session notes

A running log of what got built, what broke, and what I learned. New entries
go on top. Concept explanations and lessons (not just changelog) are the point.

---

## 2026-06-25 — Tier 2 build-out: p103 col_sum (coalescing) and p104 row_softmax (first win)

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

### Carry into next session

- p104's A/B/A delta was 0.05% — the cleanest stability measurement
  we've gotten on any kernel. Suggests longer kernels (4ms+) escape
  whatever short-kernel jitter source is firing `timing_noisy` on the
  3ms problems. Still no harness change required; reductions naturally
  drift toward longer running times.
- Four reductions in: 0.527× (sum), 0.517× (max), 0.287× (col_sum),
  1.093× (softmax). The spread (0.287×–1.093×, ~3.8× range) is now
  large enough that the harness's `fast_p` metric will resolve real
  differences in LLM kernel quality.
- Next candidates: row_l2_norm (lighter, fills out the reduction
  taxonomy), coalesced col_sum sibling (the optimization side-by-side
  story), or jump to a different reduction shape — argmax (returns
  indices, not values), or row_var/row_std (two-pass with subtract).

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
