# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).
PROBLEM = {
    "id": "p011_exp",
    "tier": 1,
    "title": "Exponential",
    "description": (
        "Element-wise exponential: out = exp(x). "
        "x is a 1D float32 tensor of length 2^25, drawn from N(0,1) so values "
        "stay well inside float32's exp range (max |x| ~5.7 statistically, far "
        "from the ~88 overflow boundary)."
    ),
    "inputs": [
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "exp_kernel",
    "notes": (
        "Use precise `exp` (the default in <metal_stdlib>), not metal::fast::exp; "
        "the reference is torch.exp which uses high-precision exp."
    ),
}


def reference(x):
    import torch
    return torch.exp(x)
