import math

_M = 64    # sequence length
_D = 64    # head dimension
_SCALE = 1.0 / math.sqrt(_D)  # = 0.125

PROBLEM = {
    "id": "p303_attention_head",
    "tier": 4,
    "title": "Single-head scaled dot-product attention (fused matmul+softmax+matmul)",
    "description": (
        "Compute out = softmax(Q @ K^T / sqrt(D)) @ V. Shapes: "
        f"Q ({_M}, {_D}), K ({_M}, {_D}), V ({_M}, {_D}), out ({_M}, {_D}). "
        "Three composite operations fused into one kernel: "
        "(1) scores = Q @ K^T scaled by 1/sqrt(D), (2) row-wise softmax "
        "over the scores, (3) output = probs @ V. One threadgroup per "
        f"query row ({_M} TGs total), {_M} threads per TG. The most "
        "ambitious composite in the project — MPS dispatches attention "
        "as at least three kernels (matmul, softmax, matmul), so the "
        "fusion savings should be substantial if our per-op cost is "
        "competitive."
    ),
    "inputs": [
        {"name": "q", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
        {"name": "k", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
        {"name": "v", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_M, _D), "dtype": "float32"},
    ],
    # Error budget: softmax's max-subtract is exact; exp + sum +
    # divide accumulate ~few ULPs; phase-1 and phase-3 dot products
    # are length D=64, error ~sqrt(D)*eps*|sum|. Total worst-case
    # abs err around 1e-4 for outputs of magnitude ~1.
    "tolerance": {"atol": 1e-3, "rtol": 1e-3},
    "entry_point": "attention_head",
    "launch": {
        "grid":        (_M * _M, 1, 1),
        "threadgroup": (_M,      1, 1),
    },
    "notes": (
        f"M = D = {_M}; one thread per output column AND per score "
        "position (clean 1:1 mapping in every phase). Scratch reuse: "
        f"a single threadgroup float[{_M}] holds scores (phase 1), "
        "then partial reduction values (phase 2 max-reduce + sum-"
        "reduce), then normalized probabilities (phase 3). Five "
        "barriers total. q_row is staged once at the top so phase 1's "
        "dot product reads it from threadgroup memory; K is read "
        "directly from device (one row per thread, coalesced within "
        "the row); V is read directly from device in phase 3 "
        "(uncoalesced — leaves room for a staged-V follow-up problem)."
    ),
}


def reference(q, k, v):
    import torch
    import math
    s = q @ k.transpose(-1, -2)
    s = s * (1.0 / math.sqrt(_D))
    p = torch.softmax(s, dim=-1)
    return p @ v
