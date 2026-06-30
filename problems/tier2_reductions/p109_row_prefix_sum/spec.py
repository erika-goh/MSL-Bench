_M = 1024
_N = 4096
_TG_THREADS = 256
_SGS_PER_TG = 8     # 256 / 32
_PER_THREAD = _N // _TG_THREADS   # 16

assert _N % _TG_THREADS == 0, "row length must divide TG_THREADS"
assert _TG_THREADS % 32 == 0, "TG must contain whole SIMD-groups"

PROBLEM = {
    "id": "p109_row_prefix_sum",
    "tier": 2,
    "title": "Row-wise inclusive prefix sum (parallel scan)",
    "description": (
        f"Compute y[m, j] = sum_{{i=0..j}} x[m, i] for an ({_M}, {_N}) "
        "float32 matrix. Same shape in and out. One TG per row "
        f"({_M} TGs of {_TG_THREADS} threads). Three-stage scan: "
        f"(1) each thread sequentially scans its {_PER_THREAD} "
        "contiguous elements and records its total; (2) within each "
        "SIMD-group, `simd_prefix_inclusive_sum` gives each thread "
        "the sum of preceding threads' totals in the SG; (3) the 8 "
        "SG totals are published to TG memory and SG 0 runs an "
        "exclusive scan to give each SG its prefix offset. Each "
        "thread then adds (sg_prefix + within_sg_prefix) to its 16 "
        "local results and writes them. The full scan is parallelized "
        "with O(log N) cross-thread coordination depth, vs the "
        "sequential reference's O(N)."
    ),
    "inputs": [
        {"name": "x", "shape": (_M, _N), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "y", "shape": (_M, _N), "dtype": "float32"},
    ],
    # Cumsum of N=4096 randn values. Output magnitude grows as
    # sqrt(j) for position j → max ~sqrt(N)=64. Parallel scan's
    # different summation order vs torch's sequential cumsum can
    # diverge by ~few * sqrt(N) * eps * |output| ≈ 1e-4 at the tail.
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "row_prefix_sum_kernel",
    "launch": {
        "grid":        (_M * _TG_THREADS, 1, 1),
        "threadgroup": (_TG_THREADS,      1, 1),
    },
    "notes": (
        "First catalog problem using SIMD-group scan intrinsics "
        "(`simd_prefix_inclusive_sum`, `simd_prefix_exclusive_sum`, "
        "`simd_broadcast`). Memory pattern: each thread reads its 16 "
        "CONTIGUOUS row elements rather than a strip-mined slice. "
        "Total cache-line traffic per row is the same either way "
        "(16 lines either via 1-per-iter strip-mine or 16-per-iter "
        "contiguous-with-L1-reuse), but contiguous is algorithmically "
        "cleaner — each thread does a straightforward sequential scan "
        "of its slice before the cross-thread coordination starts."
    ),
}


def reference(x):
    import torch
    return torch.cumsum(x, dim=-1)
