_B = 262144
_K = 256

PROBLEM = {
    "id": "p114_row_l2_norm",
    "tier": 2,
    "title": "Row-wise L2 norm",
    "description": (
        f"out[b] = sqrt(sum(x[b, :]^2)) for a ({_B}, {_K}) float32 matrix. "
        f"One threadgroup per row, {_K} threads per group."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "row_l2_norm",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": "Square each element in-register during load, then reduce, then sqrt.",
}


def reference(x):
    import torch
    return torch.norm(x, p=2, dim=1)
