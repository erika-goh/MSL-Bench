_B = 262144
_K = 256

PROBLEM = {
    "id": "p115_row_dot",
    "tier": 2,
    "title": "Row-wise dot product of two matrices",
    "description": (
        f"out[b] = sum(a[b, :] * b[b, :]) for two ({_B}, {_K}) float32 matrices. "
        f"One threadgroup per row, {_K} threads per group."
    ),
    "inputs": [
        {"name": "a", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "row_dot",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": "Multiply-then-reduce. Fused MAD (fma) into reduction is the target.",
}


def reference(a, b):
    return (a * b).sum(dim=1)
