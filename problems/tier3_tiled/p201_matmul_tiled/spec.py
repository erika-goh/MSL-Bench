# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

_M    = 1024   # rows of C and A
_N    = 1024   # cols of C and B
_K    = 1024   # inner dim — cols of A, rows of B
_TILE = 16     # tile size (and threadgroup edge)

assert _M % _TILE == 0 and _N % _TILE == 0 and _K % _TILE == 0, \
    "all three matrix dims must divide cleanly by TILE for p201"

PROBLEM = {
    "id": "p201_matmul_tiled",
    "tier": 3,
    "title": "Tiled matrix multiply (16×16 tiles, cooperative load)",
    "description": (
        "Compute C = A @ B for three square float32 matrices of shape "
        f"({_M}, {_K}), ({_K}, {_N}), ({_M}, {_N}). Tiled kernel: each "
        f"threadgroup is {_TILE}×{_TILE} threads and computes a single "
        f"{_TILE}×{_TILE} block of C. The K dimension is walked in "
        f"{_K // _TILE} tile-sized steps; at each step the threadgroup "
        "cooperatively loads one tile of A and one tile of B into "
        "threadgroup memory, barriers, accumulates the partial dot "
        "products into a per-thread float, then barriers before the "
        "next K-tile overwrites the shared tiles. Each thread carries "
        "one float accumulator across the entire K-loop and writes "
        "one element of C at the end."
    ),
    "inputs": [
        {"name": "a", "shape": (_M, _K), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_K, _N), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "c", "shape": (_M, _N), "dtype": "float32"},
    ],
    # Sum of K=1024 products of randn samples. Each product is ~O(1),
    # |C[m,n]| ~ sqrt(K) = 32 by CLT. Float32 pairwise vs sequential
    # summation differs by ~sqrt(K) * eps * |C| ≈ 1.2e-4. Add some
    # slack for fma vs separate mul-add ordering: atol=1e-2.
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "matmul_tiled",
    "launch": {
        # Grid is TOTAL threads. With one thread per output element of C:
        "grid":        (_N,    _M,    1),  # x = col, y = row
        "threadgroup": (_TILE, _TILE, 1),  # 16x16 = 256 threads per block
    },
    "notes": (
        f"TILE = {_TILE} chosen for: (a) clean SIMD-group division "
        "(16x16 = 256 threads = 8 SIMD-groups), (b) modest shared "
        f"memory footprint (2 * {_TILE}*{_TILE} * 4 bytes = "
        f"{2 * _TILE * _TILE * 4} bytes per TG), (c) clean divisibility "
        f"of M=N=K={_M} by TILE. MPS will likely win this comparison "
        "by 5–20× — Apple's BLAS is decades of tuning and is AMX-aware "
        "on M-series. The point of p201 is to establish the naive "
        "tiled baseline; optimization variants follow."
    ),
}


def reference(a, b):
    import torch
    return a @ b
