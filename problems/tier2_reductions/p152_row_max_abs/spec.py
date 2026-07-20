_B = 262144
_K = 256

PROBLEM = {
    "id": "p152_row_max_abs",
    "tier": 2,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Row-wise max-abs (L-infinity norm)",
    "description": (
        f"out[b] = max_j |x[b, j]| for a ({_B}, {_K}) float32 matrix. "
        f"One threadgroup per row, {_K} threads per group."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-4},
    "entry_point": "row_max_abs",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": "SIMD-shuffle reduce like row_sum, but reduce with max(|x|). "
             "0.0 is a safe padding identity since |x| >= 0.",
}


def reference(x):
    import torch
    return torch.amax(torch.abs(x), dim=1)
