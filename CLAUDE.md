# Project: mslbench — Metal KernelBench
A benchmark for LLM-generated Metal compute kernels on Apple Silicon.
Roadmap and phase status live in README.md's `## Roadmap` section —
single source of truth, don't duplicate it here.

## Working style (important — this is a learning project)
- I am using this project to learn Metal, Swift, and benchmark engineering.
  Do not just produce working code.
- Before any non-trivial change: explain WHAT you're about to do, WHY this
  approach over alternatives, and what could go wrong and explain it simply and easy to understand. Then do it.
- After fixing any bug: explain the root cause in 2-3 sentences before moving on.
- When you touch Metal/GPU concepts (threadgroups, buffer binding, GPU
  timestamps, storage modes), give me a one-paragraph explanation the first
  time each concept appears.
- Work in small steps. One logical change, verify it, explain, next step.
  Never make sweeping multi-file changes in one shot.
- If two reasonable approaches exist, present both with tradeoffs and let
  me choose rather than picking silently.
- Quiz me occasionally: after a milestone, ask me one question to check I
  understood the key concept. Keep it short.

## Commands
- Build runner: make runner
- Phase 0 exit criterion: make slice  (must print compiled:true, correct:true)
- Tests: make test-mac
- Never modify files in tests/golden_kernels/ — those are ground truth.
