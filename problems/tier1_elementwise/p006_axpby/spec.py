# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).
PROBLEM = {
    "id": "p006_axpby",
    "tier": 1,
    "title": "AXPBY (linear combination)",
    "description": (
        "Out-of-place linear combination of two 1D float32 tensors: "
        "out = a * x + b * y. x and y have length 2^25; a and b are "
        "one-element float32 buffers (uniforms)."
    ),
    "inputs": [
        {"name": "a", "shape": (1,), "dtype": "float32", "init": "uniform"},
        {"name": "b", "shape": (1,), "dtype": "float32", "init": "uniform"},
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
        {"name": "y", "shape": (33554432,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "entry_point": "axpby",
    "notes": (
        "Scalars `a` and `b` are bound as one-element buffers (buffers 0 and 1); "
        "read them as `a[0]` and `b[0]`. Out-of-place — `y` is read-only, not "
        "the BLAS in-place form."
    ),
}


def reference(a, b, x, y):
    return a * x + b * y
