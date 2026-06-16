# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).
PROBLEM = {
    "id": "p004_scalar_mul",
    "tier": 1,
    "title": "Scalar multiplication",
    "description": (
        "Element-wise multiply of a 1D float32 tensor by a scalar: out = alpha * x. "
        "x has length 2^25; alpha is a single-element float32 buffer (uniform)."
    ),
    "inputs": [
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
        {"name": "alpha", "shape": (1,), "dtype": "float32", "init": "uniform"},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "scalar_mul",
    "notes": (
        "`alpha` is bound as a one-element buffer — read it as `alpha[0]` in MSL. "
        "Every thread reads the same scalar; this is the canonical way to pass a "
        "uniform parameter."
    ),
}


def reference(x, alpha):
    return alpha * x
