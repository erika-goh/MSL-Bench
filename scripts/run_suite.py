#!/usr/bin/env python3
"""Run a model over the full problem suite (Phase 3).

Examples (all free):
    python scripts/run_suite.py --provider ollama --model qwen2.5-coder:14b --mode one_shot
    python scripts/run_suite.py --provider groq --model llama-3.3-70b-versatile --mode repair --k 5
    python scripts/run_suite.py --provider gemini --model gemini-2.0-flash --mode repair --k 5

Raw per-problem records land in results/raw/, transcripts included —
repair transcripts are the Phase-5 flywheel's seed trajectories.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkb import problems as P
from mkb.compile import compile_metal
from mkb.execute import run_kernel
from mkb.llm.generate import one_shot, repair_k
from mkb.timing import summarize, time_reference_mps
from mkb.verify import verify

RESULTS_RAW = Path(__file__).resolve().parents[1] / "results" / "raw"


def evaluate_kernel(kernel_src: str, problem: dict, reference) -> tuple[bool, str, dict]:
    """Compile + run + verify one candidate. Returns (success, feedback, metrics)."""
    work = Path(tempfile.mkdtemp(prefix="mkb_eval_"))
    src = work / "candidate.metal"
    src.write_text(kernel_src)

    comp = compile_metal(src, work)
    if not comp.ok:
        return False, f"COMPILE ERROR:\n{comp.diagnostics}", {"stage": "compile"}

    grid, threadgroup = P.launch_config(problem)
    inputs = P.make_inputs(problem)
    res = run_kernel(comp.metallib, problem["entry_point"], grid, threadgroup,
                     inputs, problem["outputs"],
                     zero_output_each_run=problem.get("zero_output_each_run", False))
    if not res.ok:
        return False, f"RUNTIME ERROR: {res.error}", {"stage": "runtime"}

    import torch
    t_inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
    ref_outputs = {problem["outputs"][0]["name"]: reference(**t_inputs).numpy()}
    v = verify(res.outputs, ref_outputs, **problem["tolerance"])
    if not v.correct:
        return False, f"WRONG ANSWER: {v.detail}", {"stage": "verify", "max_abs_err": v.max_abs_err}

    cand_t = summarize(res.gpu_times_ms)
    ref_t = time_reference_mps(reference, inputs)
    speedup = ref_t.median_ms / cand_t.median_ms if cand_t.median_ms > 0 else None
    return True, "ok", {
        "stage": "pass", "kernel_ms": cand_t.median_ms,
        "reference_ms": ref_t.median_ms, "speedup": speedup,
        "timing_noisy": cand_t.noisy or ref_t.noisy,
    }


def _preflight_deps() -> None:
    """Fail fast if the interpreter can't run the verify step.

    evaluate_kernel imports torch lazily (only after compile+run succeed), so a
    missing torch would otherwise be caught by the outer exception handler and
    silently recorded as `provider_error` — destroying the LLM output we paid
    for. Better to bail out before spending any provider TPM.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch is not importable in this interpreter.", file=sys.stderr)
        print("  run_suite.py needs torch for the verify step against PyTorch MPS references.", file=sys.stderr)
        print(f"  Current interpreter: {sys.executable}", file=sys.stderr)
        print("  Try:  .venv/bin/python scripts/run_suite.py ...", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    _preflight_deps()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["groq", "gemini", "ollama", "local"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=["one_shot", "repair"], default="one_shot")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tier", type=int, default=None, help="restrict to one tier")
    ap.add_argument("--only", default=None,
                    help="comma-separated problem ids to include (e.g. p001_vector_add,p013_gelu)")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds between problems (be gentle with free tiers)")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="max output tokens per generation (reasoning models need more)")
    args = ap.parse_args()
    only = set(x.strip() for x in args.only.split(",")) if args.only else None

    RESULTS_RAW.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.provider}_{(args.model or 'default').replace('/', '-').replace(':', '-')}_{args.mode}"
    # Non-default max_tokens becomes part of the tag so re-sweeps don't overwrite
    # earlier records and both budgets end up as separate leaderboard rows.
    if args.max_tokens != 4096:
        run_tag += f"_mt{args.max_tokens}"

    records = []
    for spec_path in P.discover():
        problem, reference = P.load(spec_path)
        if args.tier and problem["tier"] != args.tier:
            continue
        pid = problem["id"]
        if only is not None and pid not in only:
            continue
        print(f"[{run_tag}] {pid} ...", flush=True)

        gen = None  # so the except block can preserve the LLM output if we got that far
        try:
            if args.mode == "one_shot":
                gen = one_shot(args.provider, problem, args.model, max_tokens=args.max_tokens)
                if gen["kernel"] is None:
                    rec = {"success": False, "metrics": {"stage": "no_code"}, "attempts": 1}
                else:
                    ok, fb, metrics = evaluate_kernel(gen["kernel"], problem, reference)
                    rec = {"success": ok, "feedback": fb, "metrics": metrics, "attempts": 1}
                transcript = gen["transcript"]
            else:
                def feedback_fn(src: str):
                    ok, fb, metrics = evaluate_kernel(src, problem, reference)
                    feedback_fn.last_metrics = metrics  # type: ignore[attr-defined]
                    return ok, fb
                gen = repair_k(args.provider, problem, feedback_fn, k=args.k, model=args.model,
                               max_tokens=args.max_tokens)
                rec = {"success": gen["success"],
                       "metrics": getattr(feedback_fn, "last_metrics", {}),
                       "attempts": gen["attempts"]}
                transcript = gen["transcript"]
        except Exception as e:
            # A single provider / evaluator error should not terminate the whole
            # suite — record it as this problem's failure and continue.
            # If the LLM call already returned (`gen` is set), the exception
            # happened downstream (compile / verify / timing) and the transcript
            # we paid for is worth preserving as Phase-6 flywheel seed data.
            err = f"{type(e).__name__}: {str(e)[:500]}"
            print(f"    !! {err}", flush=True)
            rec = {"success": False,
                   "metrics": {"stage": "provider_error", "error": err},
                   "attempts": 1}
            err_msg = {"role": "error", "content": err}
            transcript = gen["transcript"] + [err_msg] if gen else [err_msg]

        record = {
            "run": run_tag, "problem": pid, "tier": problem["tier"],
            "correct": rec["success"],
            "speedup": rec["metrics"].get("speedup"),
            "fail_stage": None if rec["success"] else rec["metrics"].get("stage"),
            "attempts": rec["attempts"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "transcript": transcript,
        }
        records.append(record)
        out = RESULTS_RAW / f"{run_tag}__{pid}.json"
        out.write_text(json.dumps(record, indent=2))
        print(f"    -> {'PASS' if rec['success'] else 'FAIL'} "
              f"({rec['metrics'].get('stage', '?')}, attempts={rec['attempts']})")
        time.sleep(args.sleep)

    passed = sum(r["correct"] for r in records)
    print(f"\n{run_tag}: {passed}/{len(records)} correct")


if __name__ == "__main__":
    main()
