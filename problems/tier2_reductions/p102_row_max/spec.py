# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

_B = 262144   # 2^18 rows  — same as p101 for direct comparability
_K = 256      # row width — also threadgroup size (one thread per column)

PROBLEM = {
    "id": "p102_row_max",
    "tier": 2,
    "title": "Row-wise max",
    "description": (
        "Max along axis=1 of a 2D float32 matrix: out[b] = max(x[b, :]). "
        f"Input shape ({_B}, {_K}); output shape ({_B},). One threadgroup "
        f"per row, {_K} threads per threadgroup."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "float32"},
    ],
    # Tight: unlike sum, max is order-independent for non-NaN floats — the
    # tree reduction returns bit-exact the same value as a sequential scan.
    # Keep a tiny atol for paranoia rather than the 1e-4 we needed for sum.
    "tolerance": {"atol": 1e-6, "rtol": 1e-6},
    "entry_point": "row_max",
    # Same override as p101: one threadgroup per row, K threads per group.
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": (
        f"K = {_K} is a power of 2. Same tree-reduction pattern as p101 "
        "row_sum, but the combining op is max() instead of +. Identity "
        "element is -INFINITY (only matters if grid > input — here it does "
        "not, since every thread loads exactly one element)."
    ),
}


def reference(x):
    import torch
    # torch.amax returns values only; torch.max(x, dim=1) returns
    # (values, indices) and computes the argmax even if we discard it,
    # which makes that path ~5× slower on MPS. Use amax for a pure
    # max-vs-max comparison against the kernel.
    return torch.amax(x, dim=1)
