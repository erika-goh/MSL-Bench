_B = 262144
_K = 256

PROBLEM = {
    "id": "p311_softmax_stable",
    "tier": 4,
    "title": "Numerically stable row-wise softmax",
    "description": (
        f"For each row b: out[b, k] = exp(x[b, k] - max(x[b, :])) / sum(exp(...)). "
        f"Input x ({_B}, {_K}), output same shape. Fuses three reductions "
        f"(max, sum-of-exp, and normalize) into one kernel."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B, _K), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "entry_point": "softmax_stable",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": (
        "Three-pass fused: (1) find max via tree reduce, (2) compute exp(x-max) "
        "and sum via tree reduce, (3) divide each element by the sum. "
        "Can be done in one kernel with two threadgroup barriers."
    ),
}


def reference(x):
    import torch
    import torch.nn.functional as F
    return F.softmax(x, dim=1)
