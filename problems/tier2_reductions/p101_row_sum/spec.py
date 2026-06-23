# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

_B = 262144   # 2^18 rows
_K = 256      # row width — also threadgroup size (one thread per column)

PROBLEM = {
    "id": "p101_row_sum",
    "tier": 2,
    "title": "Row-wise sum",
    "description": (
        "Sum along axis=1 of a 2D float32 matrix: out[b] = sum(x[b, :]). "
        f"Input shape ({_B}, {_K}); output shape ({_B},). One threadgroup "
        f"per row, {_K} threads per threadgroup."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "float32"},
    ],
    # Loosened from Tier 1's 1e-6: float32 sums of 256 randn samples have
    # rounding error proportional to sqrt(K) * eps * |sum|. With K=256, eps
    # ~1e-7, magnitudes ~16, the expected divergence is in the 1e-5 range.
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "row_sum",
    # Override the default per-element launch — we want one threadgroup per
    # row, K threads per threadgroup (so threads can cooperate via
    # threadgroup-shared memory and barriers).
    "launch": {
        "grid":        (_B * _K, 1, 1),  # total threads; Metal divides by tg to get group count
        "threadgroup": (_K,      1, 1),  # K threads per group → B groups → one per row
    },
    "notes": (
        f"K = {_K} is a power of 2 (clean tree reduction). The kernel should "
        "load one element per thread into threadgroup-shared memory, then "
        "reduce via halving-stride tree, then have thread 0 write the result."
    ),
}


def reference(x):
    import torch
    return torch.sum(x, dim=1)
