_H = 2048   # input height
_W = 2048   # input width
_HO = _H // 2
_WO = _W // 2
_TILE = 16  # output-tile edge (input tile is 2*TILE = 32)

PROBLEM = {
    "id": "p253_avg_pool2d",
    "tier": 3,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Tiled 2×2 average pooling (threadgroup-staged)",
    "description": (
        f"2×2 average pooling (stride 2, no padding) of a ({_H}, {_W}) "
        f"float32 image into a ({_HO}, {_WO}) output. Each {_TILE}×{_TILE} "
        f"threadgroup computes one {_TILE}×{_TILE} output block, first "
        f"cooperatively staging the corresponding 2·{_TILE} × 2·{_TILE} "
        "input tile into threadgroup memory (each of the 256 threads loads "
        "4 elements), barriering, then averaging its own 2×2 window from "
        "the staged tile."
    ),
    "inputs": [
        {"name": "x", "shape": (_H, _W), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_HO, _WO), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-4},
    "entry_point": "avg_pool2d",
    "launch": {
        "grid":        (_WO,   _HO,   1),
        "threadgroup": (_TILE, _TILE, 1),
    },
    "notes": "Input tile is 32×32 = 1024 floats, loaded by 256 threads at "
             "4 each via a linearized thread id.",
}


def reference(x):
    import torch
    import torch.nn.functional as F
    return F.avg_pool2d(x.view(1, 1, _H, _W), kernel_size=2).view(_HO, _WO)
