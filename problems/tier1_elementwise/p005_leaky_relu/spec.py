# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

# Single source of truth: spec init AND reference both use this constant.
_NEGATIVE_SLOPE = 0.01

PROBLEM = {
    "id": "p005_leaky_relu",
    "tier": 1,
    "title": "Leaky ReLU",
    "description": (
        "Element-wise leaky ReLU: out = x if x > 0 else alpha * x. "
        "x has length 2^25; alpha is a single-element float32 buffer (uniform) "
        "fixed at the standard negative slope of 0.01."
    ),
    "inputs": [
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
        {"name": "alpha", "shape": (1,), "dtype": "float32",
         "init": "constant", "value": _NEGATIVE_SLOPE},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "leaky_relu",
    "notes": "`alpha` is bound as a one-element buffer — read it as `alpha[0]`.",
}


def reference(x, alpha):
    import torch.nn.functional as F
    # `alpha` is bound for the kernel; reference uses the fused MPS primitive
    # parameterized by the same constant the spec uses for init.value above.
    return F.leaky_relu(x, negative_slope=_NEGATIVE_SLOPE)
