# MSL-Bench session notes

A running log of what got built, what broke, and what I learned. New entries
go on top. Concept explanations and lessons (not just changelog) are the point.

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
