# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

_B    = 262144   # 2^18 rows — same matrix shape as p103 for direct comparison
_K    = 256      # number of columns / output size
_TG_X = 32       # threadgroup width — cols per TG, AND SIMD-group width
_TG_Y = 8        # threadgroup height — row stripes per TG

assert _B % _TG_Y == 0, "B must divide cleanly by TG_Y row stripes"
assert _K % _TG_X == 0, "K must divide cleanly by TG_X cols per TG"

PROBLEM = {
    "id": "p105_col_sum_tiled_naive",
    "tier": 2,
    "title": "Column-wise sum (coalesced tile — but parallelism-starved)",
    "description": (
        "Sum along axis=0 of a 2D float32 matrix: out[k] = sum(x[:, k]). "
        f"Input shape ({_B}, {_K}); output shape ({_K},). Designed as the "
        "coalesced fix for p103: threadgroup is 2D "
        f"({_TG_X} × {_TG_Y}), 32-wide along x so each SIMD-group reads "
        "32 consecutive columns at once (coalesced). HOWEVER the chosen "
        "geometry collapses TG count from p103's 256 to just K/TG_X = 8, "
        "leaving the GPU heavily under-utilized. Wall-clock is WORSE than "
        "p103 despite the coalescing fix — an intentional teaching "
        "artifact for the coalescing-vs-parallelism tradeoff."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_K,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "col_sum_tiled_naive",
    "launch": {
        "grid":        (_K,    _TG_Y, 1),
        "threadgroup": (_TG_X, _TG_Y, 1),
    },
    "notes": (
        "Lesson kernel. The (32, 8) thread layout fixes p103's coalescing "
        "problem at the per-SIMD-group level, but reduces total TGs from "
        "256 to 8 — the GPU has many cores expecting hundreds of TGs in "
        "flight and gets starved. Real fix requires multiple TGs PER "
        "column block plus an atomic or two-pass cross-TG reduction. "
        "Future problem (p1XX_col_sum_atomic or two-pass) will tackle "
        "that. Keep this one as a benchmark target: if an LLM proposes "
        "this same design, the harness will show it's not the win."
    ),
}


def reference(x):
    import torch
    return torch.sum(x, dim=0)
