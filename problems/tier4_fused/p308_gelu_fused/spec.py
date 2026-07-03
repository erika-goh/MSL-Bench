_M = 1024
_N = 1024
_K = 1024

PROBLEM = {
    "id": "p308_gelu_fused",
    "tier": 4,
    "title": "Fused linear + GELU (matmul + activation)",
    "description": (
        f"Compute C = gelu(A @ B) for ({_M}, {_K}) @ ({_K}, {_N}) -> ({_M}, {_N}). "
        f"Fuses the GELU activation into the matmul output — avoids a full "
        f"round-trip through device memory between matmul and activation."
    ),
    "inputs": [
        {"name": "a", "shape": (_M, _K), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_K, _N), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "c", "shape": (_M, _N), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "gelu_fused",
    "launch": {
        "grid":        (_N, _M, 1),
        "threadgroup": (16, 16, 1),
    },
    "notes": (
        "Apply GELU to the accumulator before writing to C. "
        "Uses the tanh approximation matching torch's default: "
        "0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."
    ),
}


def reference(a, b):
    import torch
    import torch.nn.functional as F
    return F.gelu(a @ b)
