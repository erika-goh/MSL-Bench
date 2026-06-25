#!/usr/bin/env python3
"""Run ONE candidate kernel against ONE problem, end to end.

Phase 0 exit criterion:
    python scripts/run_problem.py p001_vector_add tests/golden_kernels/vector_add.metal

Prints a JSON result: compiled / correct / speedup / timings / diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkb import problems as P
from mkb.compile import compile_metal
from mkb.execute import run_kernel
from mkb.timing import (
    check_calibration,
    check_stability,
    record_calibration,
    summarize,
    time_reference_mps,
)
from mkb.verify import verify


def run_one(problem_id: str, kernel_path: Path, calibrate: bool = False) -> dict:
    spec_path = next((p for p in P.discover() if p.parent.name == problem_id), None)
    if spec_path is None:
        return {"error": f"problem '{problem_id}' not found"}
    problem, reference = P.load(spec_path)

    result: dict = {"problem": problem_id, "tier": problem["tier"], "kernel": str(kernel_path)}

    comp = compile_metal(kernel_path, Path(tempfile.mkdtemp(prefix="mkb_build_")))
    result["compiled"] = comp.ok
    result["diagnostics"] = comp.diagnostics
    if not comp.ok:
        return result

    grid, threadgroup = P.launch_config(problem)
    inputs = P.make_inputs(problem)

    # A/B/A timing: candidate (block 1) → reference (block 2) → candidate (block 3).
    # Block 1 also produces the outputs we verify against the reference; block 3
    # is timing-only. If candidate medians from blocks 1 and 3 disagree, the
    # machine state shifted across the reference block in between and the
    # speedup ratio is untrustworthy.
    zero_each = problem.get("zero_output_each_run", False)
    exec_1 = run_kernel(
        comp.metallib, problem["entry_point"], grid, threadgroup,
        inputs, problem["outputs"],
        zero_output_each_run=zero_each,
    )
    if not exec_1.ok:
        result["error"] = exec_1.error
        return result

    import torch  # lazy: only needed once execution succeeds
    t_inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
    ref_out = reference(**t_inputs)
    ref_outputs = {problem["outputs"][0]["name"]: ref_out.numpy()}

    v = verify(exec_1.outputs, ref_outputs, **problem["tolerance"])
    result["correct"] = v.correct
    result["max_abs_err"] = v.max_abs_err
    result["verify_detail"] = v.detail

    if not v.correct:
        # Wrong-answer kernels: timing is meaningless, skip blocks 2 and 3.
        result["speedup"] = None
        result["timing_trustworthy"] = None
        return result

    ref_t = time_reference_mps(reference, inputs)

    exec_3 = run_kernel(
        comp.metallib, problem["entry_point"], grid, threadgroup,
        inputs, problem["outputs"],
        zero_output_each_run=zero_each,
    )
    if not exec_3.ok:
        result["error"] = exec_3.error
        return result

    cand_t_1 = summarize(exec_1.gpu_times_ms)
    cand_t_3 = summarize(exec_3.gpu_times_ms)
    stab = check_stability(cand_t_1, cand_t_3)
    result["block1_median_ms"] = round(cand_t_1.median_ms, 4)
    result["block3_median_ms"] = round(cand_t_3.median_ms, 4)
    result["block_delta_frac"] = round(stab.delta_frac, 4)
    result["timing_trustworthy"] = stab.stable

    if not stab.stable:
        result["stability_error"] = stab.message
        result["speedup"] = None
        if calibrate:
            result["calibration"] = "refused: " + stab.message
        return result

    # A/B/A passed → combine 20 candidate samples for a tighter kernel_ms estimate.
    cand_t = summarize(exec_1.gpu_times_ms + exec_3.gpu_times_ms)
    result["kernel_ms"] = round(cand_t.median_ms, 4)
    result["reference_ms"] = round(ref_t.median_ms, 4)
    result["speedup"] = round(ref_t.median_ms / cand_t.median_ms, 3) if cand_t.median_ms > 0 else None
    result["timing_noisy"] = cand_t.noisy or ref_t.noisy

    if calibrate:
        record_calibration(problem_id, cand_t.median_ms)
        result["calibration"] = "recorded"
    else:
        warn = check_calibration(problem_id, cand_t.median_ms)
        if warn:
            result["calibration_warning"] = warn

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("problem_id")
    ap.add_argument("kernel", type=Path)
    ap.add_argument("--calibrate", action="store_true",
                    help="record this run as the calibration baseline for drift checks")
    args = ap.parse_args()
    print(json.dumps(run_one(args.problem_id, args.kernel, args.calibrate), indent=2))
