PROBLEM = {
    "id": "p013_gelu",
    "tier": 1,
    "title": "Gaussian Error Linear Unit (tanh approximation)",
    "description": (
        "Element-wise GELU using the tanh approximation: "
        "out = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))). "
        "x is a 1D float32 tensor of length 2^25, drawn from N(0,1). "
        "PyTorch's F.gelu defaults to the EXACT (erf-based) form, so "
        "the reference uses approximate='tanh' to match the kernel. "
        "Apple's Metal does not expose `erf` in metal_stdlib (CUDA "
        "does) — a port of an exact-GELU CUDA kernel will not compile "
        "on Metal. The tanh approximation is the portable fallback."
    ),
    "inputs": [
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    # Matched approximation form on both sides; remaining error is
    # purely from `tanh` rounding (1-2 ULPs on Apple GPU). Per-element
    # error well under 1e-5 in practice.
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "entry_point": "gelu_kernel",
    "notes": (
        "Coalesced 1:1 read/write. The kernel is memory-bound (1 read + "
        "1 write per element ≈ 256 MB total at this length), so the "
        "absolute speedup vs MPS should be modest — MPS's eager F.gelu "
        "is a single fused dispatch, not a sequence like RMSNorm's "
        "torch path. Comparison closer to parity than to the 8× wins."
    ),
}


def reference(x):
    import torch
    import torch.nn.functional as F
    return F.gelu(x, approximate="tanh")
