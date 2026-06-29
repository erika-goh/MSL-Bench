import math

_M = 512   # sequence length
_D = 512   # head dimension
_SCALE = 1.0 / math.sqrt(_D)  # ≈ 0.04419417

# Per-TG block of 8 query rows (one TG = one 8-row block of output).
_ROWS_PER_TG = 8
_SG = 8            # simdgroup_matrix tile dim (Metal 3 fixes float at 8x8)
_TG_THREADS = 256  # 8 SIMD groups × 32 lanes

assert _M % _SG == 0 and _D % _SG == 0, "M, D must divide SG=8"
assert _M % _ROWS_PER_TG == 0, "M must divide ROWS_PER_TG"
assert _TG_THREADS == (_ROWS_PER_TG * 32), \
    "TG must hold one SIMD-group per output row"

PROBLEM = {
    "id": "p305_attention_simdmatmul",
    "tier": 4,
    "title": "Fused attention using simdgroup_matrix matmul (Apple matrix engine)",
    "description": (
        "Same fused attention as p304 (M = D = 512), but the two matmul "
        "phases (QK^T and PV) run on Apple's per-SIMD-group matrix unit "
        "via `simdgroup_matrix<float, 8, 8>` instead of scalar dot-product "
        "loops. Tests whether the ~10–16× compute-density bump of the "
        "matrix engine closes p304's 2:1 deficit vs MPS, which is itself "
        "BLAS-tuned and uses the matrix unit. "
        "Layout: one TG per block of 8 query rows (64 TGs total). Per TG: "
        "8 SIMD groups, 256 threads. Scores live in TG memory between the "
        "matmul and softmax phases so 8 SIMD groups can each reduce one "
        "row independently using `simd_max` / `simd_sum` (lane reductions, "
        "no barriers, no tree)."
    ),
    "inputs": [
        {"name": "q", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
        {"name": "k", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
        {"name": "v", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_M, _D), "dtype": "float32"},
    ],
    # Looser than p304's 1e-3. simdgroup_matrix accumulates in a different
    # order than CPU BLAS or scalar fp32 (p202 documented this); attention
    # then compounds with exp + divide. Realistic worst case ~few e-3 abs.
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "attention_simdmatmul",
    "launch": {
        "grid":        (_M // _ROWS_PER_TG * _TG_THREADS, 1, 1),  # 64 * 256 = 16384
        "threadgroup": (_TG_THREADS,                       1, 1),  # 256
    },
    "notes": (
        f"M = D = {_M}, ROWS_PER_TG = {_ROWS_PER_TG}, SG = {_SG}. "
        f"{_M // _ROWS_PER_TG} TGs × {_TG_THREADS} threads = "
        f"{_M // _ROWS_PER_TG * _TG_THREADS} total threads. "
        "Threadgroup memory: scores[8 × 512] = 16 KB (well under 32 KB). "
        "Q and V are read from device directly through `simdgroup_load`; "
        "no explicit Q-row staging because each tile load reuses Q "
        "within the SG. K is loaded as K[c0:c0+8, k0:k0+8] (treating K "
        "as the transposed operand B in C = A @ B^T). Phase 3 uses "
        "`simdgroup_load` from threadgroup memory (the probs scratch). "
        "If timing falls short of MPS, the most likely next levers are "
        "(a) staging K tiles into TG memory once per K-strip and (b) "
        "blocking the 8-row work so the same Q tile feeds multiple "
        "score columns before being evicted from L1."
    ),
}


def reference(q, k, v):
    import torch
    import math
    s = q @ k.transpose(-1, -2)
    s = s * (1.0 / math.sqrt(_D))
    p = torch.softmax(s, dim=-1)
    return p @ v
