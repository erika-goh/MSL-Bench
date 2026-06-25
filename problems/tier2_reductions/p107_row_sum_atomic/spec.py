# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

_B       = 262144   # 2^18 rows — same matrix shape as p101 for direct comparison
_K       = 256      # row width
_K_CHUNK = 64       # cols per TG (one row split across K/K_CHUNK = 4 TGs)

assert _K % _K_CHUNK == 0, "K must divide cleanly by K_CHUNK"

_N_TG_X = _K // _K_CHUNK   # = 4  TGs spanning the K dimension per row
_TOTAL_TGS = _N_TG_X * _B  # = 1,048,576

PROBLEM = {
    "id": "p107_row_sum_atomic",
    "tier": 2,
    "title": "Row-wise sum (atomic-combine variant, vs p101's tree reduce)",
    "description": (
        "Sum along axis=1 of a 2D float32 matrix: out[b] = sum(x[b, :]). "
        f"Input shape ({_B}, {_K}); output shape ({_B},). The atomic-"
        "combine analog of p101: each row is split into K/K_CHUNK = "
        f"{_N_TG_X} column chunks, each TG handles one chunk of one row, "
        "does a tree reduction of its K_CHUNK threads, then atomic-adds "
        f"the partial to out[row]. Total TGs: {_TOTAL_TGS:,}. Tests "
        "whether the atomics-instead-of-cooperation pattern that won "
        "on p106 col_sum also wins on row_sum."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "float32"},
    ],
    # Same tolerance as p101.
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "row_sum_atomic",
    "zero_output_each_run": True,
    "launch": {
        # Grid is TOTAL threads. With K_CHUNK threads in x per TG, the
        # K dimension contributes K = N_TG_X * K_CHUNK threads. The B
        # dimension contributes B threads (one TG per row, TG_Y = 1).
        "grid":        (_K,       _B, 1),
        "threadgroup": (_K_CHUNK, 1,  1),
    },
    "notes": (
        f"Atomic contention per output: {_N_TG_X} ops. Across {_B} "
        f"outputs, that is {_TOTAL_TGS:,} total atomic adds; they "
        "distribute across rows so the per-slot serial cost is bounded "
        f"to {_N_TG_X} atomics. Coalescing: 32 threads of a SIMD-group "
        "share row and have consecutive col → 32 consecutive addresses. "
        "Direct comparison to p101: same op, same input, same output, "
        "two different kernel idioms."
    ),
}


def reference(x):
    import torch
    return torch.sum(x, dim=1)
