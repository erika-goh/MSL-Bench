"""Problem discovery and loading."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

PROBLEMS_DIR = Path(__file__).resolve().parents[1] / "problems"

_DEFAULT_TG_1D = (256, 1, 1)


def discover() -> list[Path]:
    """All spec.py files, sorted by problem id."""
    return sorted(PROBLEMS_DIR.glob("tier*/p*/spec.py"))


def load(spec_path: Path):
    """Import a spec.py as a module. Returns (PROBLEM dict, reference fn)."""
    mod_name = f"mkb_problem_{spec_path.parent.name}"
    spec = importlib.util.spec_from_file_location(mod_name, spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.PROBLEM, mod.reference


def launch_config(problem: dict) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Resolve (grid, threadgroup) for a problem.

    Default: total threads = product of the first output's shape (flattened
    1D dispatch), threadgroup = (256, 1, 1). Suits the per-element pattern
    `uint i [[thread_position_in_grid]]; out[i] = f(in[i])`.

    Override via PROBLEM["launch"] = {"grid": (...), "threadgroup": (...)}.
    Either key may be omitted to keep the default for that field.
    """
    override = problem.get("launch", {})

    if "grid" in override:
        grid = tuple(override["grid"])
    else:
        out_shape = problem["outputs"][0]["shape"]
        n = int(math.prod(out_shape))
        grid = (n, 1, 1)

    tg = tuple(override.get("threadgroup", _DEFAULT_TG_1D))

    if len(grid) != 3 or len(tg) != 3:
        raise ValueError(
            f"launch_config for {problem.get('id', '?')}: grid/threadgroup "
            f"must be 3-tuples, got grid={grid} tg={tg}"
        )
    return grid, tg


def make_inputs(problem: dict, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for x in problem["inputs"]:
        shape, dtype = tuple(x["shape"]), np.dtype(x["dtype"])
        init = x.get("init", "randn")
        if init == "randn":
            arr = rng.standard_normal(shape).astype(dtype)
        elif init == "uniform":
            arr = rng.uniform(-1, 1, shape).astype(dtype)
        elif init == "zeros":
            arr = np.zeros(shape, dtype=dtype)
        elif init == "constant":
            arr = np.full(shape, x["value"], dtype=dtype)
        else:
            raise ValueError(f"unknown init '{init}'")
        out[x["name"]] = arr
    return out
