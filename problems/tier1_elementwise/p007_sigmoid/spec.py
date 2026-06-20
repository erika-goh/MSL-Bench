# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).
PROBLEM = {
    "id": "p007_sigmoid",
    "tier": 1,
    "title": "Sigmoid",
    "description": (
        "Element-wise sigmoid: out = 1 / (1 + exp(-x)). "
        "x is a 1D float32 tensor of length 2^25."
    ),
    "inputs": [
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "sigmoid_kernel",
    "notes": (
        "Use precise `exp` (the default in <metal_stdlib>), not metal::fast::exp; "
        "the reference is torch.sigmoid which uses high-precision exp."
    ),
}


def reference(x):
    import torch
    return torch.sigmoid(x)
