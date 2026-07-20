_B = 262144
_K = 256

PROBLEM = {
    "id": "p151_row_l1_norm",
    "tier": 2,
    "split": "heldout",  # Phase-2.5 test set — excluded from train runs + SFT
    "title": "Row-wise L1 norm",
    "description": (
        f"out[b] = sum_j |x[b, j]| for a ({_B}, {_K}) float32 matrix. "
        f"One threadgroup per row, {_K} threads per group."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-4},
    "entry_point": "row_l1_norm",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": "Same SIMD-reduce idiom as row_sum, but reduce |x| instead of x.",
}


def reference(x):
    import torch
    return torch.sum(torch.abs(x), dim=1)
