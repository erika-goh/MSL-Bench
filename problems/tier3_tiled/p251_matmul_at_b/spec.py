_M    = 1024   # rows of C  = cols of A
_N    = 1024   # cols of C  = cols of B
_K    = 1024   # inner dim  = rows of A = rows of B
_TILE = 16

assert _M % _TILE == 0 and _N % _TILE == 0 and _K % _TILE == 0

PROBLEM = {
    "id": "p251_matmul_at_b",
    "tier": 3,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Tiled A^T @ B (transposed-A matmul, 16×16 tiles)",
    "description": (
        f"Compute C = A^T @ B where A is ({_K}, {_M}), B is ({_K}, {_N}), "
        f"C is ({_M}, {_N}), all float32. Same {_TILE}×{_TILE} cooperative "
        "tile-staging as the plain tiled matmul, but A is stored "
        "K-major so each thread loads A[k, m] (a strided/transposed read) "
        "into the A tile. One threadgroup computes one output tile; the K "
        "dimension is walked in tile-sized steps with a threadgroup barrier "
        "around each shared-tile load and use."
    ),
    "inputs": [
        {"name": "a", "shape": (_K, _M), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_K, _N), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "c", "shape": (_M, _N), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "matmul_at_b",
    "launch": {
        "grid":        (_N,    _M,    1),
        "threadgroup": (_TILE, _TILE, 1),
    },
    "notes": "A is (K, M): element A[k, m] lives at a[k*M + m]. The A-tile "
             "load is the only change vs p201 — B and the inner loop match.",
}


def reference(a, b):
    import torch
    return a.t() @ b
