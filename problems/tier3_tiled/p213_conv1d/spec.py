_N = 65536   # signal length
_KW = 15     # kernel width (odd for symmetric padding)

PROBLEM = {
    "id": "p213_conv1d",
    "tier": 3,
    "title": "1D convolution, single channel, valid padding",
    "description": (
        f"Compute out[i] = sum_k x[i + k] * w[k] for k in [0, {_KW}). "
        f"Input length {_N}, kernel width {_KW}, output length {_N - _KW + 1}. "
        f"Valid (no) padding, stride 1, single channel in and out."
    ),
    "inputs": [
        {"name": "x", "shape": (_N,),  "dtype": "float32", "init": "randn"},
        {"name": "w", "shape": (_KW,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_N - _KW + 1,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "conv1d",
    "notes": (
        f"KW = {_KW} weights loaded once per threadgroup into threadgroup memory "
        "is the typical optimization. Naive: one thread per output does its own "
        "KW-loop."
    ),
}


def reference(x, w):
    import torch
    import torch.nn.functional as F
    # F.conv1d computes cross-correlation (no flip), which matches our
    # spec: out[i] = sum_k x[i + k] * w[k].
    x_ = x.unsqueeze(0).unsqueeze(0)      # (1, 1, N)
    w_ = w.unsqueeze(0).unsqueeze(0)      # (1, 1, KW)
    out = F.conv1d(x_, w_)
    return out.squeeze(0).squeeze(0)
