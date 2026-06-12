# MSL-Bench session notes

A running log of what got built, what broke, and what I learned. New entries
go on top. Concept explanations and lessons (not just changelog) are the point.

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
