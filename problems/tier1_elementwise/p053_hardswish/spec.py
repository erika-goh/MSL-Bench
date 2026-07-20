_N = 33554432  # 2^25

PROBLEM = {
    "id": "p053_hardswish",
    "tier": 1,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Hard Swish",
    "description": (
        "Element-wise hardswish: out = x * clamp(x + 3, 0, 6) / 6, "
        "for x of length 2^25 (float32)."
    ),
    "inputs": [
        {"name": "x", "shape": (_N,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_N,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "hardswish",
    "notes": "relu6(x+3)/6 gating; no transcendentals, so tight tolerance.",
}


def reference(x):
    import torch.nn.functional as F
    return F.hardswish(x)
