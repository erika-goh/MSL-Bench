PROBLEM = {
    "id": "p014_square",
    "tier": 1,
    "title": "Element-wise square",
    "description": "out[i] = x[i] * x[i] for a 1D float32 tensor of length 2^25.",
    "inputs": [
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "square",
    "notes": "",
}


def reference(x):
    return x * x
