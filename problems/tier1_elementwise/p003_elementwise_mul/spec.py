# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).
PROBLEM = {
    "id": "p003_elementwise_mul",
    "tier": 1,
    "title": "Element-wise multiplication",
    "description": "Element-wise multiplication of two 1D float32 tensors of length 2^25.",
    "inputs": [
        {"name": "a", "shape": (33554432,), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (33554432,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "elementwise_mul",
    "notes": "",
}


def reference(a, b):
    return a * b
