"""Scoring: the fast_p metric, mirroring KernelBench's framing.

fast_p = fraction of problems where the kernel is correct AND achieves
speedup >= p over the PyTorch MPS reference.
fast_0 reduces to plain correctness rate.
"""
from __future__ import annotations


def fast_p(records: list[dict], p: float) -> float:
    """records: [{"correct": bool, "speedup": float|None}, ...]"""
    if not records:
        return 0.0
    hits = sum(1 for r in records
               if r.get("correct") and (p == 0 or (r.get("speedup") or 0.0) >= p))
    return hits / len(records)


def tier_table(records: list[dict], ps=(0.0, 1.0, 2.0)) -> dict:
    """Group records by tier and compute fast_p columns."""
    tiers: dict[int, list[dict]] = {}
    for r in records:
        tiers.setdefault(r["tier"], []).append(r)
    return {
        tier: {f"fast_{p:g}": round(fast_p(rs, p), 4) for p in ps} | {"n": len(rs)}
        for tier, rs in sorted(tiers.items())
    }
