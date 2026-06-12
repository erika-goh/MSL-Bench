"""Numerical verification of candidate outputs against the reference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VerifyResult:
    correct: bool
    max_abs_err: float
    max_rel_err: float
    detail: str


def verify(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray],
           atol: float, rtol: float) -> VerifyResult:
    worst_abs, worst_rel, correct, notes = 0.0, 0.0, True, []
    for name, ref in reference.items():
        if name not in candidate:
            return VerifyResult(False, float("inf"), float("inf"), f"missing output '{name}'")
        cand = candidate[name]
        if cand.shape != ref.shape:
            return VerifyResult(False, float("inf"), float("inf"),
                                f"output '{name}' shape {cand.shape} != reference {ref.shape}")
        if not np.all(np.isfinite(cand)):
            return VerifyResult(False, float("inf"), float("inf"),
                                f"output '{name}' contains NaN/Inf")
        abs_err = np.abs(cand.astype(np.float64) - ref.astype(np.float64))
        denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-12)
        rel_err = abs_err / denom
        worst_abs = max(worst_abs, float(abs_err.max()))
        worst_rel = max(worst_rel, float(rel_err.max()))
        if not np.allclose(cand, ref, atol=atol, rtol=rtol):
            correct = False
            bad = int((abs_err > atol + rtol * np.abs(ref)).sum())
            notes.append(f"'{name}': {bad}/{ref.size} elements out of tolerance, "
                         f"max_abs_err={abs_err.max():.3e}")
    return VerifyResult(correct, worst_abs, worst_rel, "; ".join(notes) or "ok")
