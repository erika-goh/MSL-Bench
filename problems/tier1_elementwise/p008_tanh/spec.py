# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).
PROBLEM = {
    "id": "p008_tanh",
    "tier": 1,
    "title": "Tanh",
    "description": (
        "Element-wise hyperbolic tangent: out = tanh(x). "
        "x is a 1D float32 tensor of length 2^25."
    ),
    "inputs": [
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "tanh_kernel",
    "notes": (
        "Use precise `tanh` (the default in <metal_stdlib>), not metal::fast::tanh; "
        "the reference is torch.tanh which uses high-precision tanh."
    ),
}


def reference(x):
    import torch
    return torch.tanh(x)
