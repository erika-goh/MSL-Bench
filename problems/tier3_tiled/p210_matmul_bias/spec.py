_M = 1024
_N = 1024
_K = 1024

PROBLEM = {
    "id": "p210_matmul_bias",
    "tier": 3,
    "title": "Matrix multiply with bias broadcast",
    "description": (
        f"Compute C[m, n] = sum_k A[m, k] * B[k, n] + bias[n]. "
        f"Shapes: A ({_M}, {_K}), B ({_K}, {_N}), bias ({_N},), C ({_M}, {_N}). "
        f"Fuses a bias-add into a tiled matmul."
    ),
    "inputs": [
        {"name": "a",    "shape": (_M, _K), "dtype": "float32", "init": "randn"},
        {"name": "b",    "shape": (_K, _N), "dtype": "float32", "init": "randn"},
        {"name": "bias", "shape": (_N,),    "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "c", "shape": (_M, _N), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "matmul_bias",
    "launch": {
        "grid":        (_N, _M, 1),
        "threadgroup": (16, 16, 1),
    },
    "notes": "Add bias[n] to the accumulator before writing C[m, n].",
}


def reference(a, b, bias):
    return a @ b + bias
