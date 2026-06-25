# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

_B = 262144   # 2^18 rows  — same matrix shape as p101/p102/p103
_K = 256      # row width — also threadgroup size

PROBLEM = {
    "id": "p104_row_softmax",
    "tier": 2,
    "title": "Row-wise softmax",
    "description": (
        "Row-wise softmax of a 2D float32 matrix. For each row b: "
        "out[b, k] = exp(x[b, k] - max_j(x[b, j])) / sum_j(exp(x[b, j] - max_j(x[b, j]))). "
        f"Input and output shape ({_B}, {_K}). One threadgroup per row, "
        f"{_K} threads per group. Numerical stability via max-subtraction "
        "is REQUIRED — naive softmax overflows on inputs with magnitude > ~88. "
        "Multi-pass within a single kernel: (1) tree-reduce for row max, "
        "(2) tree-reduce for row sum of exp(x - max), (3) per-element divide."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        # Same shape as input — unlike p101/p102/p103 which output (B,) or (K,).
        {"name": "out", "shape": (_B, _K), "dtype": "float32"},
    ],
    # Softmax outputs are in [0, 1] and sum to 1 per row. Error compounds
    # through max (exact) → exp (~1 ulp) → sum-of-K (~sqrt(K) eps) → divide
    # (1 ulp). Expect max_abs_err in the 1e-6 range; atol=1e-5 gives slack.
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "entry_point": "row_softmax",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": (
        f"K = {_K} threads per row. Each thread owns one column. The "
        "kernel runs three phases against a single threadgroup-shared "
        "scratch array, which is reused across both reductions. After "
        "each tree-reduce, every thread reads scratch[0] into a thread-"
        "local register, then barriers before phase-2 overwrites scratch. "
        "(The scaffold also documents an alternate broadcast-scalar "
        "pattern using `threadgroup float row_max;` — it works but "
        "trips a spurious -Wsometimes-uninitialized warning under "
        "Metal's per-thread flow analysis.)"
    ),
}


def reference(x):
    import torch
    return torch.softmax(x, dim=1)
