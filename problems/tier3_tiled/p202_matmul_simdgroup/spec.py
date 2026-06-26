# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

_M = 1024
_N = 1024
_K = 1024
_SG = 8        # simdgroup matrix dim (Metal 3 fixes float SG matrices at 8x8)
_TGS = 32      # threads per threadgroup = one SIMD-group

assert _M % _SG == 0 and _N % _SG == 0 and _K % _SG == 0, \
    "all three matrix dims must divide cleanly by SG=8"

PROBLEM = {
    "id": "p202_matmul_simdgroup",
    "tier": 3,
    "title": "Matrix multiply via simdgroup_matrix (Apple GPU matrix unit)",
    "description": (
        "Compute C = A @ B using Apple's per-SIMD-group matrix unit. "
        f"Three square 1024x1024 float32 matrices. Each SIMD-group "
        f"(32 threads, one TG) computes one 8x8 tile of C using "
        f"`simdgroup_matrix<float, 8, 8>`. The K dimension is walked "
        f"in K/SG = {_K // _SG} steps of `simdgroup_multiply_accumulate`. "
        "No explicit threadgroup_barriers in the hot loop — matrix-unit "
        "ops are SIMD-group-synchronous. Tests how much of p201's gap "
        "to MPS closes when we let the hardware matrix unit do the "
        "work the manual tile kernel does in software."
    ),
    "inputs": [
        {"name": "a", "shape": (_M, _K), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_K, _N), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "c", "shape": (_M, _N), "dtype": "float32"},
    ],
    # Realistic slack: the matrix unit's accumulation order differs from
    # CPU BLAS (which p201 happened to match bit-exactly). Expect ~1e-4
    # to ~1e-3 abs error on magnitude ~32 outputs.
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "matmul_simdgroup",
    "launch": {
        # One SIMD-group per output tile: (N/SG) * SG_threads in x,
        # (M/SG) * 1 in y. TG = (32, 1, 1).
        "grid":        (_N // _SG * _TGS, _M // _SG, 1),  # = (4096, 128, 1)
        "threadgroup": (_TGS,             1,         1),  # = (32,   1,   1)
    },
    "notes": (
        "Apple's simdgroup_matrix functions are Metal 3+ only. The "
        "matrix dimensions are fixed at 8x8 for float; the SIMD-group "
        "of 32 threads collectively owns the 64 elements. Load/store "
        "operations work directly on device or threadgroup pointers — "
        "no need to first stage into threadgroup memory. The kernel is "
        "much shorter than p201's because the heavy lifting moves into "
        "the matrix-unit intrinsics."
    ),
}


def reference(a, b):
    import torch
    return a @ b
