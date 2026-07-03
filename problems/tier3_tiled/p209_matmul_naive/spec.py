_M = 1024
_N = 1024
_K = 1024

PROBLEM = {
    "id": "p209_matmul_naive",
    "tier": 3,
    "title": "Naive matrix multiply (one thread per output)",
    "description": (
        f"Compute C = A @ B for ({_M}, {_K}) @ ({_K}, {_N}) -> ({_M}, {_N}). "
        f"One thread computes one element of C by looping over the K axis. "
        f"Baseline for tiled variants — expect MPS to beat this by 10-50x."
    ),
    "inputs": [
        {"name": "a", "shape": (_M, _K), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_K, _N), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "c", "shape": (_M, _N), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "matmul_naive",
    "launch": {
        "grid":        (_N, _M, 1),
        "threadgroup": (16, 16, 1),
    },
    "notes": "One thread = one output element. K-loop in the kernel. No tiling.",
}


def reference(a, b):
    return a @ b
