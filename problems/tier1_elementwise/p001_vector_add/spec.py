# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).
PROBLEM = {
    "id": "p001_vector_add",
    "tier": 1,
    "title": "Vector addition",
    "description": "Element-wise addition of two 1D float32 tensors of length 2^20.",
    "inputs": [
        {"name": "a", "shape": (1048576,), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (1048576,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (1048576,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "vector_add",
    "notes": "",
}


def reference(a, b):
    return a + b
