import math

_M = 512   # sequence length
_D = 512   # head dimension
_SCALE = 1.0 / math.sqrt(_D)  # ≈ 0.04419417

PROBLEM = {
    "id": "p304_attention_large",
    "tier": 4,
    "title": "Single-head scaled dot-product attention at M=D=512 (compute-bound regime)",
    "description": (
        "Same kernel structure as p303 (one TG per query row, M threads "
        "per TG, scratch reuse across phases), but at production-scale "
        f"sequence length: M = D = {_M}. Compute grows as M·M·D ≈ {_M*_M*_D/1e6:.0f}M "
        "ops vs ~0.5M at p303 — a ~500× scale-up — while MPS's per-dispatch "
        "overhead stays constant. The p303 fusion advantage (~8×) came from "
        "amortizing 3× dispatch overhead over tiny compute; at this size, "
        "compute should dominate and the speedup should collapse toward 1× "
        "(or below, if our single-thread-per-output dot products lose to "
        "MPS's BLAS-tuned matmul). This is the other end of the fusion "
        "curve from p303."
    ),
    "inputs": [
        {"name": "q", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
        {"name": "k", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
        {"name": "v", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_M, _D), "dtype": "float32"},
    ],
    # Error budget: length-D dot products (D=512) accumulate ~sqrt(D)*eps
    # ≈ 2.7e-6 relative error. Softmax denominator is a length-M sum of
    # exp values, similar order. Output is prob-weighted sum of length M.
    # Worst-case abs err around few * 1e-5 for outputs of magnitude ~1.
    # Loosened vs p303 (which had 1e-3) — same budget; the larger N just
    # uses more of it.
    "tolerance": {"atol": 1e-3, "rtol": 1e-3},
    "entry_point": "attention_large",
    "launch": {
        "grid":        (_M * _M, 1, 1),
        "threadgroup": (_M,      1, 1),
    },
    "notes": (
        f"M = D = {_M}. 512 TGs × 512 threads/TG = 262144 total threads. "
        "Same scratch layout as p303: scratch[M] reused across scores → "
        "max-reduce → sum-reduce → probs, plus q_row[D] staged once. "
        "Threadgroup memory used: 2·M·4 = 4 KB (well under the 32 KB "
        "budget). Reduction tree depth grows from 6 (p303) to 9 levels. "
        "Phase-3 V reads are still uncoalesced (stride-D across threads "
        "at fixed j); at D=512 this may bite harder than at D=64 — a "
        "future staged-V variant could test that."
    ),
}


def reference(q, k, v):
    import torch
    import math
    s = q @ k.transpose(-1, -2)
    s = s * (1.0 / math.sqrt(_D))
    p = torch.softmax(s, dim=-1)
    return p @ v
