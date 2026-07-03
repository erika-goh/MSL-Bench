_B = 4096
_K = 4096

PROBLEM = {
    "id": "p113_col_max",
    "tier": 2,
    "title": "Column-wise maximum",
    "description": (
        f"out[k] = max(x[:, k]) for a ({_B}, {_K}) float32 matrix. "
        f"Reduces along the strided axis (bad for coalescing) — testing "
        f"whether the model reaches for a transpose or does a scatter-reduce."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_K,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 0.0, "rtol": 0.0},
    "entry_point": "col_max",
    "notes": (
        "Column reductions on row-major storage are the classic memory-"
        "access-pattern test. Naive one-thread-per-column is memory-strided; "
        "tiling with threadgroup memory recovers coalescing."
    ),
}


def reference(x):
    import torch
    return torch.max(x, dim=0).values
