"""Execute a compiled kernel via the Swift runner.

Marshalling is deliberately boring: raw little-endian binaries on disk plus a
JSON manifest. No IPC, no shared memory — v0.1 optimizes for debuggability.
Buffer binding convention: inputs in spec order get buffer(0..n-1), outputs
follow in spec order. This convention is also stated in the LLM prompt.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_RUNNER = Path(__file__).resolve().parents[1] / "runner" / ".build" / "release" / "Runner"


@dataclass
class ExecResult:
    ok: bool
    error: str | None
    gpu_times_ms: list[float]
    outputs: dict[str, np.ndarray]


def _dtype(np_name: str) -> np.dtype:
    return np.dtype(np_name).newbyteorder("<")


def run_kernel(
    metallib: Path,
    entry_point: str,
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    inputs: dict[str, np.ndarray],
    output_specs: list[dict],
    warmup_ms_min: float = 50.0,
    warmup_ms_max: float = 500.0,
    warmup_iter_max: int = 10_000,
    runs: int = 10,
    runner_path: Path | None = None,
) -> ExecResult:
    runner = Path(os.environ.get("MKB_RUNNER", runner_path or DEFAULT_RUNNER))
    if not runner.exists():
        return ExecResult(False, f"runner binary not found at {runner} — run `make runner` first", [], {})

    with tempfile.TemporaryDirectory(prefix="mkb_") as td:
        tdir = Path(td)
        buffers = []

        for name, arr in inputs.items():
            p = tdir / f"in_{name}.bin"
            arr.astype(arr.dtype.newbyteorder("<"), copy=False).tofile(p)
            buffers.append({"path": str(p), "bytes": int(arr.nbytes), "mode": "in"})

        out_paths: list[tuple[str, Path, tuple, np.dtype]] = []
        for spec in output_specs:
            dt = _dtype(spec["dtype"])
            nbytes = int(np.prod(spec["shape"])) * dt.itemsize
            p = tdir / f"out_{spec['name']}.bin"
            buffers.append({"path": str(p), "bytes": nbytes, "mode": "out"})
            out_paths.append((spec["name"], p, tuple(spec["shape"]), dt))

        manifest = {
            "metallib": str(metallib),
            "entry_point": entry_point,
            "grid": list(grid),
            "threadgroup": list(threadgroup),
            "buffers": buffers,
            "warmup_ms_min": warmup_ms_min,
            "warmup_ms_max": warmup_ms_max,
            "warmup_iter_max": warmup_iter_max,
            "runs": runs,
        }
        mpath = tdir / "manifest.json"
        mpath.write_text(json.dumps(manifest))

        proc = subprocess.run([str(runner), str(mpath)], capture_output=True, text=True, timeout=300)
        try:
            res = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return ExecResult(False, f"runner produced no JSON. stderr: {proc.stderr.strip()[:2000]}", [], {})

        if not res.get("ok", False):
            return ExecResult(False, res.get("error", "unknown runner error"), [], {})

        outputs = {}
        for name, p, shape, dt in out_paths:
            outputs[name] = np.fromfile(p, dtype=dt).reshape(shape)

        return ExecResult(True, None, res["gpu_times_ms"], outputs)
