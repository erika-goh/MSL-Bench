_M = 4096
_N = 4096
_TG_THREADS = 256
_SGS_PER_TG = 8     # 256 / 32

assert _N % _TG_THREADS == 0, "N must divide TG_THREADS for clean strip-mining"

PROBLEM = {
    "id": "p206_sgemv",
    "tier": 3,
    "title": "Single-precision matrix-vector multiply (sgemv) with two-stage reduction",
    "description": (
        f"Compute y = A @ x for A shape ({_M}, {_N}) and x shape ({_N},). "
        f"One threadgroup per output row m: {_M} TGs of {_TG_THREADS} "
        f"threads each. Each thread strip-mines {_N // _TG_THREADS} "
        "multiply-adds of A's row against x, then the TG reduces the "
        f"{_TG_THREADS} partial sums to a single scalar via a two-stage "
        f"`simd_sum`: 32 lanes per SIMD-group collapse to 1, then {_SGS_PER_TG} "
        "SG partials are gathered through TG memory and reduced again by "
        "one SG. The reduction pattern is distinct from p101 row_sum "
        "(which tree-reduces in TG memory) because we use the lane-level "
        "intrinsic — no scratch array for the within-SG step, no barrier."
    ),
    "inputs": [
        {"name": "a", "shape": (_M, _N), "dtype": "float32", "init": "randn"},
        {"name": "x", "shape": (_N,),    "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "y", "shape": (_M,), "dtype": "float32"},
    ],
    # |y[m]| ~ sqrt(N) = 64 by CLT (sum of N=4096 randn × randn products).
    # Float32 pairwise sum vs simd_sum reduction order differs by
    # ~sqrt(N) * eps * |y| ≈ 1.5e-4. Matches p201's tolerance budget.
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "sgemv_kernel",
    "launch": {
        "grid":        (_M * _TG_THREADS, 1, 1),   # = (1048576, 1, 1)
        "threadgroup": (_TG_THREADS,      1, 1),   # = (256,     1, 1)
    },
    "notes": (
        "Memory pattern: A read row-by-row coalesced (adjacent threads "
        "→ adjacent columns within the row). x read coalesced (same "
        "pattern). For sgemv specifically, x is reused across every "
        f"row m, so all {_M} TGs re-read the same x; whether L1 catches "
        "that depends on x's footprint ({_N} * 4 = 16 KB) fitting in "
        "L1, which it likely does. A staged-x variant would only help "
        "if the L1 hit rate proves low — an empirical question for a "
        "future p207."
    ),
}


def reference(a, x):
    import torch
    return a @ x
