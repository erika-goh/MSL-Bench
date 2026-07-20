_M    = 1024
_N    = 1024
_K    = 1024
_TILE = 16

assert _M % _TILE == 0 and _N % _TILE == 0 and _K % _TILE == 0

PROBLEM = {
    "id": "p352_fused_linear_sigmoid",
    "tier": 4,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Fused linear + sigmoid (tiled matmul + bias + activation)",
    "description": (
        f"Compute out = sigmoid(x @ w + b). Shapes: x ({_M}, {_K}), "
        f"w ({_K}, {_N}), b ({_N},), out ({_M}, {_N}), float32. A single "
        f"kernel fuses a {_TILE}×{_TILE} tiled matmul with a per-column "
        "bias add and a sigmoid activation in the store epilogue — one "
        "dispatch instead of the addmm + sigmoid that MPS would issue "
        "separately."
    ),
    "inputs": [
        {"name": "x", "shape": (_M, _K), "dtype": "float32", "init": "randn"},
        {"name": "w", "shape": (_K, _N), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_N,),    "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_M, _N), "dtype": "float32"},
    ],
    # Pre-activation magnitude ~sqrt(K)=32, so sigmoid mostly saturates to
    # 0/1 and post-activation error is small; matmul-scale slack still ample.
    "tolerance": {"atol": 1e-3, "rtol": 1e-3},
    "entry_point": "fused_linear_sigmoid",
    "launch": {
        "grid":        (_N,    _M,    1),
        "threadgroup": (_TILE, _TILE, 1),
    },
    "notes": "Tiled-matmul skeleton (p201) with a sigmoid(acc + b[n]) store.",
}


def reference(x, w, b):
    import torch
    return torch.sigmoid(x @ w + b)
