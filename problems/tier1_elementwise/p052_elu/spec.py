_N = 33554432  # 2^25
_ALPHA = 1.0

PROBLEM = {
    "id": "p052_elu",
    "tier": 1,
    "split": "heldout",  # Phase-2.5 test set
    "title": "ELU",
    "description": (
        "Element-wise ELU with alpha=1.0: out = x if x > 0 else alpha*(exp(x) - 1), "
        "for x of length 2^25 (float32)."
    ),
    "inputs": [
        {"name": "x", "shape": (_N,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_N,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-4},
    "entry_point": "elu",
    "notes": "alpha fixed at 1.0 as an in-kernel constant, matching the reference.",
}


def reference(x):
    import torch.nn.functional as F
    return F.elu(x, alpha=_ALPHA)
