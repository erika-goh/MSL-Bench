_B = 262144
_K = 256

PROBLEM = {
    "id": "p154_row_range",
    "tier": 2,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Row-wise range (max - min)",
    "description": (
        f"out[b] = max_j(x[b, j]) - min_j(x[b, j]) for a ({_B}, {_K}) float32 "
        f"matrix. One threadgroup per row, {_K} threads per group. Two "
        "reductions (max and min) carried together, then subtracted."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-4},
    "entry_point": "row_range",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": "Dual tree reduction: max and min accumulate in parallel "
             "threadgroup arrays; lane 0 writes max - min.",
}


def reference(x):
    import torch
    return torch.amax(x, dim=1) - torch.amin(x, dim=1)
