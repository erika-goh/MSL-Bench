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
from mkb.timing import check_calibration, record_calibration, summarize, time_reference_mps
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
    if comp.grid is None or comp.threadgroup is None:
        result["compiled"] = True
        result["error"] = "missing MKB_GRID / MKB_TG launch-config comments"
        return result

    inputs = P.make_inputs(problem)

    exec_res = run_kernel(
        comp.metallib, problem["entry_point"], comp.grid, comp.threadgroup,
        inputs, problem["outputs"],
    )
    if not exec_res.ok:
        result["error"] = exec_res.error
        return result

    import torch  # lazy: only needed once execution succeeds
    t_inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
    ref_out = reference(**t_inputs)
    ref_outputs = {problem["outputs"][0]["name"]: ref_out.numpy()}

    v = verify(exec_res.outputs, ref_outputs, **problem["tolerance"])
    result["correct"] = v.correct
    result["max_abs_err"] = v.max_abs_err
    result["verify_detail"] = v.detail

    cand_t = summarize(exec_res.gpu_times_ms)
    ref_t = time_reference_mps(reference, inputs)
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
