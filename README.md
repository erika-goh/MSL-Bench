# Metal KernelBench

A benchmark for evaluating LLM-generated **Metal compute kernels** on Apple Silicon —
the Metal-side counterpart to [KernelBench](https://github.com/ScalingIntelligence/KernelBench)'s
CUDA suite. Candidate kernels are compiled with the real Xcode toolchain, executed on
the GPU, verified against PyTorch (MPS) references, and timed with GPU timestamps.

**Status:** v0.1 scaffold — Phase 0 vertical slice (2 problems, full harness).

Runs entirely on a Mac you already own. Zero cloud, zero paid APIs:
model evaluation supports **Ollama (local)**, **Groq (free tier)**, and
**Gemini Flash (free tier)** out of the box.

## Requirements

- Apple Silicon Mac, macOS 13+
- Xcode command line tools (`xcode-select --install`) — provides `xcrun metal`
- Python 3.10+ with `numpy` and `torch` (`pip install -e ".[bench,dev]"`)
- Swift toolchain (ships with Xcode CLT)
- Optional, for model evaluation: [Ollama](https://ollama.com) and/or free API keys
  (`GROQ_API_KEY`, `GEMINI_API_KEY`)

## Quickstart (Phase 0 exit criterion)

```bash
make runner    # builds the Swift Metal runner
make slice     # golden vector_add kernel: compile → run → verify → time
```

Expected output: JSON with `"compiled": true, "correct": true` and a speedup number.
Then prove the harness catches failures:

```bash
make test-mac  # includes wrong-answer and non-compiling kernel tests
```

Record a calibration baseline (used to detect thermal drift in later sessions):

```bash
python scripts/run_problem.py p001_vector_add tests/golden_kernels/vector_add.metal --calibrate
```

## Evaluating a model (free)

```bash
# fully local
ollama pull qwen2.5-coder:14b
python scripts/run_suite.py --provider ollama --model qwen2.5-coder:14b --mode repair --k 5

# free API tiers
export GROQ_API_KEY=...
python scripts/run_suite.py --provider groq --model llama-3.3-70b-versatile --mode one_shot

export GEMINI_API_KEY=...
python scripts/run_suite.py --provider gemini --model gemini-2.0-flash --mode repair --k 5

python scripts/make_leaderboard.py   # -> results/tables/leaderboard.md
```

## How a candidate kernel is judged

1. **Compile** — `xcrun metal` → `.metallib`; diagnostics captured verbatim (they feed the repair loop)
2. **Execute** — Swift runner binds buffers, dispatches, reads `MTLCommandBuffer` GPU timestamps
3. **Verify** — `allclose` vs PyTorch MPS reference (per-problem atol/rtol); NaN/Inf is an automatic fail
4. **Time** — 3 warmups, median of 10; runs with IQR > 15% of median flagged noisy;
   calibration drift > 10% warns that the session is thermally suspect

Headline metric: **fast_p** — fraction of problems correct AND ≥ p× the MPS reference speed
(fast_0 = correctness, fast_1 = matches MPS, fast_2 = 2× MPS).

## Kernel conventions (what models are told)

- One self-contained `.metal` file; kernel function named per the problem's `entry_point`
- Buffers: inputs bound `[[buffer(0)]]..[[buffer(N-1)]]` in spec order, outputs continue the numbering
- Launch config declared via magic comments (dispatchThreads is used — grid need not
  be a multiple of the threadgroup):

```metal
// MKB_GRID: 1048576 1 1
// MKB_TG: 256 1 1
```

## Repository map

```
runner/     Swift Metal execution shim (deliberately dumb: one kernel, one dispatch)
mkb/        Python orchestration: compile, execute, verify, timing, scoring, LLM clients
problems/   tierN/pNNN_name/spec.py + README per problem
results/    raw per-run JSON (transcripts included) + generated leaderboard tables
scripts/    run_problem (debug loop), run_suite (sweep), make_leaderboard
tests/      portable tests (verify/score/timing) + Mac-only end-to-end harness tests
```

## Roadmap

- [x] Phase 0 — vertical slice: harness end to end on 2 problems
- [x] Phase 1 — timing trust: calibration discipline, interleaved A/B timing
- [ ] Phase 2 — problem suite: ~60 problems across 4 tiers (elementwise → reductions → tiled → fused) *(36/60 landed: T1 ×12, T2 ×9, T3 ×8, T4 ×7)*
- [ ] Phase 3 — LLM evaluation: one-shot vs repair@5 across free models *(one_shot done for Gemini + Groq; repair@5 in progress)*
- [ ] Phase 4 — analysis: failure taxonomy, difficulty cliff, report + article *(findings landing in NOTES.md; formal writeup pending)*
- [ ] Phase 5 — public demo: web leaderboard with kernel-vs-MPS side-by-side, live
      latency bars, failure-class heatmap, and a "watch the kernel run" view.
      Positioning: *"A benchmark that measures how well LLMs write GPU kernels —
      and where they break."* Stack TBD.
- [ ] Phase 6 — (separate repo) data flywheel: repair transcripts → SFT → re-benchmark

## Known limitations (v0.1, by design)

- fp32 only (a few deliberate fp16 problems arrive with Tier 4)
- Launch config via magic comments is crude — getting it right is part of the task
- Single dispatch per problem; multi-kernel pipelines are out of scope
- Standard ops appear in models' training data as CUDA — this benchmark measures
  **Metal translation/idiom competence**, not algorithm discovery, and says so
