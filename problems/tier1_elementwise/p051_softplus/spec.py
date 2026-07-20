_N = 33554432  # 2^25

PROBLEM = {
    "id": "p051_softplus",
    "tier": 1,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Softplus",
    "description": (
        "Element-wise softplus: out = log(1 + exp(x)) for x of length 2^25 "
        "(float32). Use the numerically stable form."
    ),
    "inputs": [
        {"name": "x", "shape": (_N,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_N,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-4},
    "entry_point": "softplus",
    "notes": "Stable form: max(x, 0) + log(1 + exp(-|x|)) avoids overflow for large x.",
}


def reference(x):
    import torch.nn.functional as F
    return F.softplus(x)
