_M = 1024
_N = 1024
_K = 1024

PROBLEM = {
    "id": "p215_matmul_scaled",
    "tier": 3,
    "title": "Scaled matrix multiply: C = alpha * (A @ B)",
    "description": (
        f"Compute C = alpha * (A @ B) for ({_M}, {_K}) @ ({_K}, {_N}) -> ({_M}, {_N}). "
        f"alpha is a scalar (single-element buffer). Tests fusing a scale into a "
        f"tiled matmul (attention-style scaling is a common workload)."
    ),
    "inputs": [
        {"name": "a",     "shape": (_M, _K), "dtype": "float32", "init": "randn"},
        {"name": "b",     "shape": (_K, _N), "dtype": "float32", "init": "randn"},
        {"name": "alpha", "shape": (1,),     "dtype": "float32", "init": "constant", "value": 0.125},
    ],
    "outputs": [
        {"name": "c", "shape": (_M, _N), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "matmul_scaled",
    "launch": {
        "grid":        (_N, _M, 1),
        "threadgroup": (16, 16, 1),
    },
    "notes": "alpha=0.125 (attention scaling for head_dim=64). Multiply once at write time.",
}


def reference(a, b, alpha):
    return alpha * (a @ b)
