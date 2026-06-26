_M  = 1024
_N  = 1024
_K  = 1024
_SG = 8
_TGS = 32

PROBLEM = {
    "id": "p302_fused_linear_relu",
    "tier": 4,
    "title": "Fused linear + ReLU (matmul + bias + activation in one kernel)",
    "description": (
        "Compute out = relu(x @ w + b). Shapes: x (1024, 1024), "
        "w (1024, 1024), b (1024,), out (1024, 1024). Single-kernel "
        "fusion of a matmul (using the simdgroup matrix unit, p202 "
        "style), a per-column bias add, and a ReLU activation. MPS "
        "dispatches this as at least two kernels (addmm + relu), so "
        "we save one dispatch worth of overhead — the question is "
        "whether that recovery offsets our matmul deficit vs MPS's "
        "tuned BLAS."
    ),
    "inputs": [
        {"name": "x", "shape": (_M, _K), "dtype": "float32", "init": "randn"},
        {"name": "w", "shape": (_K, _N), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_N,),    "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_M, _N), "dtype": "float32"},
    ],
    # Matmul-level accuracy slack (sum of K=1024 products) plus the
    # bias add (no new error) plus ReLU (clamp to zero — no error).
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "fused_linear_relu",
    "launch": {
        "grid":        (_N // _SG * _TGS, _M // _SG, 1),  # (4096, 128, 1)
        "threadgroup": (_TGS,             1,         1),  # (32,   1,   1)
    },
    "notes": (
        "Same launch geometry as p202: one SIMD-group per 8x8 output "
        "tile. Post-matmul epilogue: simdgroup_store to threadgroup "
        "memory, then 32 threads each handle 2 of the 64 output "
        "elements (add bias, ReLU, store to device). Reference uses "
        "the natural torch idiom: torch.relu(x @ w + b)."
    ),
}


def reference(x, w, b):
    import torch
    return torch.relu(x @ w + b)
