_B = 262144
_K = 256

PROBLEM = {
    "id": "p110_row_mean",
    "tier": 2,
    "title": "Row-wise mean",
    "description": (
        f"out[b] = sum(x[b, :]) / K for a ({_B}, {_K}) float32 matrix. "
        f"One threadgroup per row, {_K} threads per group."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-4},
    "entry_point": "row_mean",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": "Like row_sum but divide by K at the end.",
}


def reference(x):
    import torch
    return torch.mean(x, dim=1)
