_H = 256
_W = 256
_KH = 3
_KW = 3
_STRIDE = 2

# Output spatial size for valid padding, stride 2:
_OH = (_H - _KH) // _STRIDE + 1
_OW = (_W - _KW) // _STRIDE + 1

PROBLEM = {
    "id": "p214_conv2d_stride2",
    "tier": 3,
    "title": "2D convolution, 3x3 kernel, stride 2, single channel",
    "description": (
        f"Compute out[oh, ow] = sum_kh,kw x[oh*{_STRIDE} + kh, ow*{_STRIDE} + kw] * w[kh, kw]. "
        f"Input ({_H}, {_W}), kernel ({_KH}, {_KW}), stride {_STRIDE}, valid padding. "
        f"Output ({_OH}, {_OW})."
    ),
    "inputs": [
        {"name": "x", "shape": (_H, _W),   "dtype": "float32", "init": "randn"},
        {"name": "w", "shape": (_KH, _KW), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_OH, _OW), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "conv2d_stride2",
    "notes": (
        f"Stride {_STRIDE} halves the spatial resolution — output is ({_OH}, {_OW}). "
        "Naive: one thread per output pixel, KH*KW loop each."
    ),
}


def reference(x, w):
    import torch
    import torch.nn.functional as F
    x_ = x.unsqueeze(0).unsqueeze(0)
    w_ = w.unsqueeze(0).unsqueeze(0)
    out = F.conv2d(x_, w_, stride=_STRIDE)
    return out.squeeze(0).squeeze(0)
