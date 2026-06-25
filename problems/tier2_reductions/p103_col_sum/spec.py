# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

_B = 262144   # 2^18 rows — same matrix shape as p101 for direct comparability
_K = 256      # row width / number of columns / number of output elements
_TG = 256     # threads per threadgroup (each thread sums _B / _TG strided rows)

assert _B % _TG == 0, "B must divide cleanly by threads-per-threadgroup"

PROBLEM = {
    "id": "p103_col_sum",
    "tier": 2,
    "title": "Column-wise sum (uncoalesced baseline)",
    "description": (
        "Sum along axis=0 of a 2D float32 matrix: out[k] = sum(x[:, k]). "
        f"Input shape ({_B}, {_K}); output shape ({_K},). One threadgroup "
        f"per column ({_TG} threads per group, each summing {_B // _TG} "
        "strided rows), then a tree reduction across threads within the "
        "group. The natural translation of row_sum's geometry — but because "
        "adjacent threads end up reading addresses K floats apart instead "
        "of 1 apart, loads are uncoalesced."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_K,), "dtype": "float32"},
    ],
    # Each output is a sum of B=262144 randn samples. Rounding error grows
    # as ~sqrt(B) * eps * |sum| ≈ sqrt(262144) * 1.2e-7 * sqrt(B) ≈ ~3e-2
    # in the worst case but typically much smaller (the partial sums via
    # tree reduction cluster magnitudes near the true |sum| ~ sqrt(B)).
    # Use a loose absolute tolerance and let rtol carry the verification.
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "col_sum",
    "launch": {
        # K threadgroups × _TG threads = K * _TG total threads.
        "grid":        (_K * _TG, 1, 1),
        "threadgroup": (_TG,      1, 1),
    },
    "notes": (
        f"One threadgroup per column ({_K} groups). Each thread "
        f"accumulates {_B // _TG} strided row values of its column, then "
        "the group tree-reduces. Stride-K addresses across adjacent threads "
        "make the global loads uncoalesced — this is the teaching point. "
        "Sibling problem (future) will explore the coalesced/transpose fix."
    ),
}


def reference(x):
    import torch
    return torch.sum(x, dim=0)
