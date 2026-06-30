_IN_H  = 1028
_IN_W  = 1028
_K     = 5
_OUT_H = _IN_H - _K + 1   # 1024
_OUT_W = _IN_W - _K + 1   # 1024
_TILE  = 16               # output-tile edge
_HALO  = _K - 1           # 4 — halo size on TWO sides total, splitting (K-1)/2 each
_IN_TILE = _TILE + _HALO  # 20 — input-tile edge per TG

assert _OUT_H % _TILE == 0 and _OUT_W % _TILE == 0, \
    "output dims must divide TILE=16"

PROBLEM = {
    "id": "p208_conv2d_5x5_tiled",
    "tier": 3,
    "title": "2D convolution 5×5 with tile-with-halo input staging",
    "description": (
        f"Same kernel shape as p207 (3×3 conv) extended to K = {_K}, "
        f"but with the canonical tile-with-halo optimization: each TG "
        f"covers a {_TILE}×{_TILE} block of output, and cooperatively "
        f"stages the ({_IN_TILE}×{_IN_TILE}) = {_IN_TILE * _IN_TILE} "
        "input pixels that block depends on into threadgroup memory "
        "ONCE before any output is computed. After the barrier, every "
        f"output thread reads its {_K}×{_K} window from threadgroup "
        "memory — zero redundant device reads, vs the naïve version "
        f"where each input pixel could be read by up to {_K * _K} "
        "different output threads (and the L1 cache has to absorb that). "
        "Pedagogical question: does the halo pattern actually beat L1 "
        "on Apple Silicon?"
    ),
    "inputs": [
        {"name": "x", "shape": (_IN_H, _IN_W), "dtype": "float32", "init": "randn"},
        {"name": "w", "shape": (_K,    _K),    "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "y", "shape": (_OUT_H, _OUT_W), "dtype": "float32"},
    ],
    # 25-element dot product. Error per output ~sqrt(K^2)*eps*|y|
    # ≈ 5 * 1.2e-7 * O(5) ≈ 3e-6. atol=1e-4 has plenty of slack.
    "tolerance": {"atol": 1e-4, "rtol": 1e-5},
    "entry_point": "conv2d_5x5_tiled_kernel",
    "launch": {
        "grid":        (_OUT_W, _OUT_H, 1),   # one thread per output pixel
        "threadgroup": (_TILE,  _TILE,  1),   # 16×16 = 256 threads
    },
    "notes": (
        f"K = {_K}, TILE = {_TILE}, IN_TILE = {_IN_TILE}. "
        f"Threadgroup memory: {_IN_TILE * _IN_TILE * 4} bytes "
        "(1600 bytes) for the staged input + ~108 bytes for w if "
        "we staged it (we don't — w fits in the constant cache). "
        "Cooperative load: 256 threads need to cover 400 input "
        "pixels, so a strided linear loop covers exactly 1 or 2 "
        "pixels per thread (144 threads load 2, 112 load 1)."
    ),
}


def reference(x, w):
    import torch
    import torch.nn.functional as F
    x_b = x.unsqueeze(0).unsqueeze(0)
    w_b = w.unsqueeze(0).unsqueeze(0)
    return F.conv2d(x_b, w_b).squeeze(0).squeeze(0)
