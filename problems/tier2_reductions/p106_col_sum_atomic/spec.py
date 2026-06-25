# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

_B         = 262144   # 2^18 rows — same matrix shape as p103/p105
_K         = 256      # number of columns / output size
_TG_X      = 32       # threadgroup width — cols per TG, AND SIMD-group width
_TG_Y      = 8        # threadgroup height — threads cooperating per col
_ROW_CHUNK = 1024     # rows handled by one TG (per-thread: ROW_CHUNK/TG_Y = 128)

assert _B % _ROW_CHUNK == 0, "B must divide cleanly by ROW_CHUNK"
assert _ROW_CHUNK % _TG_Y == 0, "ROW_CHUNK must divide cleanly by TG_Y"
assert _K % _TG_X == 0,         "K must divide cleanly by TG_X"

_N_TG_X = _K // _TG_X              # = 8  TGs in x
_N_TG_Y = _B // _ROW_CHUNK         # = 256 TGs in y
_TOTAL_TGS = _N_TG_X * _N_TG_Y     # = 2048

PROBLEM = {
    "id": "p106_col_sum_atomic",
    "tier": 2,
    "title": "Column-wise sum (coalesced + atomic cross-TG reduction)",
    "description": (
        "Sum along axis=0 of a 2D float32 matrix: out[k] = sum(x[:, k]). "
        f"Input shape ({_B}, {_K}); output shape ({_K},). The real fix "
        "for p103/p105: keep p105's (32, 8) tile for coalesced loads, but "
        f"split the row dimension into {_N_TG_Y} chunks so the kernel "
        f"launches {_TOTAL_TGS} threadgroups instead of 8. Each TG "
        "computes 32 per-column partial sums; thread (tx, 0) of each TG "
        "then does an atomic_fetch_add into out[col] to combine "
        "partials across TGs. Should finally beat MPS on column sum."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_K,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "col_sum_atomic",
    # REQUIRED for any atomic-accumulator kernel: the harness re-zeros
    # output buffers before every dispatch. Without this, repeat
    # dispatches in the warmup + timed loops accumulate on top of each
    # other and correctness fails.
    "zero_output_each_run": True,
    "launch": {
        "grid":        (_K,    _N_TG_Y * _TG_Y, 1),  # = (256, 2048, 1)
        "threadgroup": (_TG_X, _TG_Y,           1),  # = (32,  8,    1)
    },
    "notes": (
        f"Atomic contention per output slot: {_N_TG_Y} ops "
        f"(one per row-block TG). With {_K} outputs and {_N_TG_Y} "
        "atomics each, total atomic ops is "
        f"{_K * _N_TG_Y} but they distribute across {_K} slots so "
        "the serial bottleneck is bounded. Uses Metal 3 `atomic_float` "
        "and `atomic_fetch_add_explicit` with memory_order_relaxed "
        "(we need atomicity, not ordering)."
    ),
}


def reference(x):
    import torch
    return torch.sum(x, dim=0)
