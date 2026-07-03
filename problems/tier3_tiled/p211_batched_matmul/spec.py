_BATCH = 16
_M = 256
_N = 256
_K = 256

PROBLEM = {
    "id": "p211_batched_matmul",
    "tier": 3,
    "title": "Batched matrix multiply",
    "description": (
        f"Compute C[i] = A[i] @ B[i] for each of {_BATCH} batches. "
        f"Shapes: A ({_BATCH}, {_M}, {_K}), B ({_BATCH}, {_K}, {_N}), "
        f"C ({_BATCH}, {_M}, {_N})."
    ),
    "inputs": [
        {"name": "a", "shape": (_BATCH, _M, _K), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_BATCH, _K, _N), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "c", "shape": (_BATCH, _M, _N), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "batched_matmul",
    "launch": {
        # Grid: (N, M, BATCH) — one threadgroup handles a tile of one batch element.
        "grid":        (_N, _M, _BATCH),
        "threadgroup": (16, 16,  1),
    },
    "notes": "Use z-dim of the grid for batch. Same tile structure as p201 within each batch.",
}


def reference(a, b):
    import torch
    return torch.bmm(a, b)
