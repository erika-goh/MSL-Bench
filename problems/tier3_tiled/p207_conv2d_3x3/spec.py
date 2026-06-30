_IN_H  = 1026
_IN_W  = 1026
_K     = 3
_OUT_H = _IN_H - _K + 1   # 1024
_OUT_W = _IN_W - _K + 1   # 1024
_TILE  = 16

assert _OUT_H % _TILE == 0 and _OUT_W % _TILE == 0, \
    "output dims must divide TILE=16 for clean dispatch"

PROBLEM = {
    "id": "p207_conv2d_3x3",
    "tier": 3,
    "title": "2D convolution, 3×3 valid (no padding, stride 1)",
    "description": (
        f"Compute y = conv2d(x, w) where x is ({_IN_H}, {_IN_W}), w is "
        f"({_K}, {_K}), and y is ({_OUT_H}, {_OUT_W}) — \"valid\" mode, "
        "no padding, stride 1. PyTorch's F.conv2d is cross-correlation "
        "(no kernel flip), matching the natural loop ordering used here. "
        "Each thread computes one output element via a 9-element 2D dot "
        "product over the input window. Input shape is chosen so the "
        "output divides evenly by the threadgroup edge."
    ),
    "inputs": [
        {"name": "x", "shape": (_IN_H, _IN_W), "dtype": "float32", "init": "randn"},
        {"name": "w", "shape": (_K,    _K),    "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "y", "shape": (_OUT_H, _OUT_W), "dtype": "float32"},
    ],
    # 9-element dot product. Float32 error per element ~sqrt(9)*eps*|y|
    # ≈ 3 * 1.2e-7 * O(3) ≈ 1.1e-6. atol=1e-4 has plenty of slack.
    "tolerance": {"atol": 1e-4, "rtol": 1e-5},
    "entry_point": "conv2d_3x3_kernel",
    "launch": {
        "grid":        (_OUT_W, _OUT_H, 1),   # one thread per output element
        "threadgroup": (_TILE,  _TILE,  1),   # 16×16 = 256 threads
    },
    "notes": (
        f"IN_H = IN_W = {_IN_H}, K = {_K}, OUT_H = OUT_W = {_OUT_H}. "
        f"{(_OUT_H // _TILE) * (_OUT_W // _TILE)} TGs × {_TILE * _TILE} "
        "threads = output element count. Coalescing: at any (ki, kj) "
        "iteration, adjacent threads (varying j_out) read addresses one "
        "apart in x — coalesced. The 3×3 weight matrix is read by every "
        "thread; since w is 36 bytes, every load hits L1 (or even the "
        "constant cache after the first access). No need to stage w."
    ),
}


def reference(x, w):
    import torch
    import torch.nn.functional as F
    # F.conv2d expects (N, C_in, H, W) and (C_out, C_in, K_h, K_w).
    # For single-channel single-output, wrap both with two unit dims.
    x_b = x.unsqueeze(0).unsqueeze(0)   # (1, 1, IN_H, IN_W)
    w_b = w.unsqueeze(0).unsqueeze(0)   # (1, 1, K, K)
    return F.conv2d(x_b, w_b).squeeze(0).squeeze(0)
