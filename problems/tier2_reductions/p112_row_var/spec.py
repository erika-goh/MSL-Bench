_B = 262144
_K = 256

PROBLEM = {
    "id": "p112_row_var",
    "tier": 2,
    "title": "Row-wise variance (population)",
    "description": (
        f"out[b] = mean((x[b, :] - mean(x[b, :]))^2). Population variance "
        f"(divisor K, not K-1). Input shape ({_B}, {_K}), output ({_B},). "
        f"One threadgroup per row, {_K} threads per group."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "row_var",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": (
        "Two-pass or one-pass with sum+sumsq. One-pass identity: "
        "var = sumsq/K - (sum/K)^2. Numerically fine for K=256 randn."
    ),
}


def reference(x):
    import torch
    return torch.var(x, dim=1, unbiased=False)
