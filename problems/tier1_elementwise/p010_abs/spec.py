# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).
PROBLEM = {
    "id": "p010_abs",
    "tier": 1,
    "title": "Absolute value",
    "description": (
        "Element-wise absolute value: out = |x|. "
        "x is a 1D float32 tensor of length 2^25."
    ),
    "inputs": [
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "abs_kernel",
    "notes": "",
}


def reference(x):
    import torch
    return torch.abs(x)
