# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

_M       = 1024
_N       = 1024
_K       = 1024
_TG_M    = 16     # output rows per TG  (= 2 matrix-unit row tiles)
_TG_N    = 16     # output cols per TG  (= 2 matrix-unit col tiles)
_K_STAGE = 32     # K columns staged into threadgroup memory per outer iter
_SG      = 8      # matrix-unit fragment size
_TGS     = 32     # threads per TG = one SIMD-group

assert _M % _TG_M == 0 and _N % _TG_N == 0 and _K % _K_STAGE == 0, \
    "matrix dims must divide cleanly by tile/stage sizes"
assert _K_STAGE % _SG == 0, "K_STAGE must be a multiple of SG=8"

PROBLEM = {
    "id": "p203_matmul_simdgroup_staged",
    "tier": 3,
    "title": "Matmul: simdgroup matrix + threadgroup-staged loads + 4 outputs/TG",
    "description": (
        "Compute C = A @ B for 1024x1024 float32 matrices. Each TG "
        f"(one SIMD-group, {_TGS} threads) produces a {_TG_M}x{_TG_N} "
        f"patch of C, made of {(_TG_M // _SG) * (_TG_N // _SG)} matrix-"
        f"unit 8x8 tiles accumulated in registers. Outer K-loop stages "
        f"a {_K_STAGE}-deep slab of A and B into threadgroup memory; "
        f"inner K-loop runs {_K_STAGE // _SG} simdgroup_multiply_"
        "accumulate calls per output tile, reading operands from "
        "threadgroup memory. Each staged A tile feeds both column "
        "outputs and each staged B tile feeds both row outputs — 2x "
        "input reuse vs p202."
    ),
    "inputs": [
        {"name": "a", "shape": (_M, _K), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_K, _N), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "c", "shape": (_M, _N), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "matmul_simdgroup_staged",
    "launch": {
        # Grid is TOTAL threads. With TGS=32 threads per TG and one TG per
        # 16x16 patch of C: grid.x = (N/TG_N) * TGS, grid.y = (M/TG_M).
        "grid":        (_N // _TG_N * _TGS, _M // _TG_M, 1),  # (2048, 64, 1)
        "threadgroup": (_TGS,                1,          1),  # (32,   1,  1)
    },
    "notes": (
        f"4096 TGs ({_M // _TG_M} x {_N // _TG_N}), 4x fewer than p202. "
        "Per output element, device reads drop from p202's ~256 floats "
        "to ~128 — the input-reuse win. Two barriers per outer iter; "
        "no barrier inside the inner K-loop because matrix-unit ops are "
        f"SIMD-group-synchronous. Threadgroup memory used: 2 x {_TG_M} "
        f"x {_K_STAGE} x 4 = {2 * _TG_M * _K_STAGE * 4} bytes per TG."
    ),
}


def reference(a, b):
    import torch
    return a @ b
