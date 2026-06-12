"""Problem discovery and loading."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

PROBLEMS_DIR = Path(__file__).resolve().parents[1] / "problems"


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
        else:
            raise ValueError(f"unknown init '{init}'")
        out[x["name"]] = arr
    return out
