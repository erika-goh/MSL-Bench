# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

_B = 262144   # 2^18 rows — same matrix shape as p101/p102 for direct comparison
_K = 256      # row width — also threadgroup size

PROBLEM = {
    "id": "p108_row_argmax",
    "tier": 2,
    "title": "Row-wise argmax (paired reduction, int32 output)",
    "description": (
        "For each row b, find the column index of the maximum value: "
        "out[b] = argmax_j(x[b, j]). Input shape "
        f"({_B}, {_K}) float32; output shape ({_B},) int32. One "
        f"threadgroup per row, {_K} threads cooperate via a paired tree "
        "reduction carrying BOTH the candidate value AND the candidate "
        "index. Ties resolve to the lowest index, matching torch's "
        "convention. First problem in the project with a non-float "
        "output dtype."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        # int32, not int64: matches Metal's native uint width and halves
        # the output buffer size. The reference casts torch's default
        # int64 result down before comparison.
        {"name": "out", "shape": (_B,), "dtype": "int32"},
    ],
    # Integer outputs: exact equality required. With randn inputs the
    # probability of a tie at the maximum is effectively zero, so the
    # candidate must match the reference at every position. atol/rtol=0
    # collapses np.allclose to strict equality.
    "tolerance": {"atol": 0, "rtol": 0},
    "entry_point": "row_argmax",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": (
        f"K = {_K} is a power of 2. Same tree shape as p101/p102, but "
        "now each scratch slot is a (value, index) pair — two parallel "
        "threadgroup arrays. At each reduction stage, the survivor's "
        "value AND index must propagate together. Strict-greater-than "
        "comparison (`>`, not `>=`) means ties keep the lower-index "
        "survivor, matching torch.argmax."
    ),
}


def reference(x):
    import torch
    # torch.argmax returns int64 by default; cast down so the dtype
    # matches the kernel output without an implicit promote in numpy.
    return torch.argmax(x, dim=1).to(torch.int32)
