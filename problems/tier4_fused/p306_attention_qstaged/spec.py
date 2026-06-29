import math

_M = 512
_D = 512
_SCALE = 1.0 / math.sqrt(_D)

_ROWS_PER_TG = 8
_SG = 8
_TG_THREADS = 256

PROBLEM = {
    "id": "p306_attention_qstaged",
    "tier": 4,
    "title": "p305 + Q staged into threadgroup memory (eliminating 64× Q redundancy)",
    "description": (
        "Same kernel as p305 with one change: the 8 query rows that a "
        "TG works on are loaded into threadgroup memory once at the "
        "top, cooperatively across all 256 threads. Phase 1's Q load "
        "then reads from threadgroup memory instead of re-issuing the "
        "same device read on every (SG, tile, kt) iteration. Tests "
        "whether the 64× Q redundancy in p305's inner loop showed up "
        "as a real bandwidth cost or was absorbed by L1. "
        "TG memory: 16 KB scores + 16 KB Q-stage = 32 KB (at the cap)."
    ),
    "inputs": [
        {"name": "q", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
        {"name": "k", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
        {"name": "v", "shape": (_M, _D), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_M, _D), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "attention_qstaged",
    "launch": {
        "grid":        (_M // _ROWS_PER_TG * _TG_THREADS, 1, 1),
        "threadgroup": (_TG_THREADS,                       1, 1),
    },
    "notes": (
        "Cooperative Q load: 256 threads each load 16 elements "
        "(4096 / 256 = 16). Coalesced because adjacent threads in a "
        "SIMD group read adjacent device addresses at each step. "
        "One barrier after the load, then the original p305 kernel "
        "body runs unchanged except for the Q load source (TG instead "
        "of device). K and V are still device-only — they have no "
        "intra-TG redundancy to eliminate without a work restructure."
    ),
}


def reference(q, k, v):
    import torch
    import math
    s = q @ k.transpose(-1, -2)
    s = s * (1.0 / math.sqrt(_D))
    p = torch.softmax(s, dim=-1)
    return p @ v
