_M    = 1024   # rows of C = rows of A
_N    = 1024   # cols of C = rows of B
_K    = 1024   # inner dim = cols of A = cols of B
_TILE = 16

assert _M % _TILE == 0 and _N % _TILE == 0 and _K % _TILE == 0

PROBLEM = {
    "id": "p252_matmul_a_bt",
    "tier": 3,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Tiled A @ B^T (transposed-B matmul, 16×16 tiles)",
    "description": (
        f"Compute C = A @ B^T where A is ({_M}, {_K}), B is ({_N}, {_K}), "
        f"C is ({_M}, {_N}), all float32. Same {_TILE}×{_TILE} cooperative "
        "tile-staging as the plain tiled matmul, but B is stored row-major "
        "as (N, K), so the B tile is filled by reading B[n, k] — a "
        "transposed load relative to the standard (K, N) layout. One "
        "threadgroup computes one output tile, walking K in tile steps "
        "with a barrier around each shared-tile load and use."
    ),
    "inputs": [
        {"name": "a", "shape": (_M, _K), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_N, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "c", "shape": (_M, _N), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "matmul_a_bt",
    "launch": {
        "grid":        (_N,    _M,    1),
        "threadgroup": (_TILE, _TILE, 1),
    },
    "notes": "B is (N, K): element B[n, k] lives at b[n*K + k]. The B-tile "
             "load is the only change vs p201 — A and the inner loop match.",
}


def reference(a, b):
    import torch
    return a @ b.t()
