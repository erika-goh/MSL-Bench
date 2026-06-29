_M    = 2048
_N    = 2048
_TILE = 16

assert _M % _TILE == 0 and _N % _TILE == 0, \
    "both dims must divide TILE=16 for the tiled transpose"

PROBLEM = {
    "id": "p205_transpose",
    "tier": 3,
    "title": "Matrix transpose via threadgroup-staged tile",
    "description": (
        f"Transpose a float32 matrix of shape ({_M}, {_N}) into "
        f"({_N}, {_M}). B[i, j] = A[j, i]. "
        "The naïve thread-per-element kernel reads A coalesced "
        "(adjacent threads → adjacent A addresses) but writes B "
        "uncoalesced (adjacent threads → addresses M floats apart). "
        f"The golden kernel uses a {_TILE}x{_TILE} threadgroup-memory "
        "tile so both reads and writes are coalesced: read A's tile "
        "in natural order into TG memory, barrier, then write to B "
        "by reading the TG tile with swapped indices."
    ),
    "inputs": [
        {"name": "a", "shape": (_M, _N), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "b", "shape": (_N, _M), "dtype": "float32"},
    ],
    # Pure data movement, no arithmetic. fp32 in → fp32 out, no
    # accumulation, no precision drift. Bit-exact in practice.
    "tolerance": {"atol": 1e-7, "rtol": 0.0},
    "entry_point": "transpose_kernel",
    "launch": {
        "grid":        (_N,    _M,    1),    # one thread per A element
        "threadgroup": (_TILE, _TILE, 1),
    },
    "notes": (
        f"M = N = {_M}, TILE = {_TILE}. "
        f"{_M // _TILE} × {_N // _TILE} = {(_M // _TILE) * (_N // _TILE)} "
        f"TGs × {_TILE * _TILE} threads = {_M * _N} total threads "
        "(one per input element). Threadgroup memory per TG: "
        f"{_TILE * _TILE * 4} bytes = 1 KB. "
        "MPS transpose is a tuned reference; expect speedup ≤ 1× at this "
        "size. The point of p205 is to introduce the coalesced-write "
        "lesson into Tier 3 (which until now is only matmul variants)."
    ),
}


def reference(a):
    import torch
    return a.t().contiguous()
