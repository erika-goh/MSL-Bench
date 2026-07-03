PROBLEM = {
    "id": "p009_neg",
    "tier": 1,
    "title": "Element-wise negation",
    "description": "out[i] = -x[i] for a 1D float32 tensor of length 2^25.",
    "inputs": [
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "neg",
    "notes": "",
}


def reference(x):
    return -x
