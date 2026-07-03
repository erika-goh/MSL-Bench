_M = 4096
_N = 4096
_TILE = 32

PROBLEM = {
    "id": "p212_transpose_tiled",
    "tier": 3,
    "title": "Tiled matrix transpose (32x32 tiles)",
    "description": (
        f"Compute out[n, m] = x[m, n] for a ({_M}, {_N}) float32 matrix. "
        f"Naive transpose is memory-strided in one direction; tiled version "
        f"loads a {_TILE}x{_TILE} block into threadgroup memory, then "
        f"writes it back transposed with coalesced accesses."
    ),
    "inputs": [
        {"name": "x", "shape": (_M, _N), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_N, _M), "dtype": "float32"},
    ],
    "tolerance": {"atol": 0.0, "rtol": 0.0},
    "entry_point": "transpose_tiled",
    "launch": {
        "grid":        (_N,    _M,    1),
        "threadgroup": (_TILE, _TILE, 1),
    },
    "notes": (
        "Classic memory-access-pattern problem. Naive one-thread-per-output "
        "has coalesced writes but strided reads (or vice versa). Tile via "
        "threadgroup memory to get coalesced access on both sides."
    ),
}


def reference(x):
    return x.transpose(0, 1).contiguous()
