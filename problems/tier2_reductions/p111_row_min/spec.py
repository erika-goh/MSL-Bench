_B = 262144
_K = 256

PROBLEM = {
    "id": "p111_row_min",
    "tier": 2,
    "title": "Row-wise minimum",
    "description": (
        f"out[b] = min(x[b, :]) for a ({_B}, {_K}) float32 matrix. "
        f"One threadgroup per row, {_K} threads per group."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 0.0, "rtol": 0.0},
    "entry_point": "row_min",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": "Halving-tree reduce with min() at each step. Identity = +inf.",
}


def reference(x):
    import torch
    return torch.min(x, dim=1).values
