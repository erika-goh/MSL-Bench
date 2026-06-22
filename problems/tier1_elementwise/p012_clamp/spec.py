# torch is imported lazily inside reference() so problem specs
# load on any machine (prompt building does not require torch).

# Single source of truth: spec init AND reference both use these constants.
# Chosen so that meaningful fraction (~32%) of standard normal x is clamped.
_LO = -1.0
_HI = 1.0

PROBLEM = {
    "id": "p012_clamp",
    "tier": 1,
    "title": "Clamp",
    "description": (
        "Element-wise clamp: out = min(max(x, lo), hi). "
        "x has length 2^25; lo and hi are one-element float32 buffers (uniforms) "
        "fixed at -1.0 and 1.0 — about 32% of standard-normal x falls outside "
        "this range, so the operation does meaningful work per element."
    ),
    "inputs": [
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
        {"name": "lo", "shape": (1,), "dtype": "float32",
         "init": "constant", "value": _LO},
        {"name": "hi", "shape": (1,), "dtype": "float32",
         "init": "constant", "value": _HI},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "clamp_kernel",
    "notes": (
        "`lo` and `hi` are bound as one-element buffers (buffers 1 and 2); read "
        "them as `lo[0]` and `hi[0]`. Use metal::clamp for the fused min/max."
    ),
}


def reference(x, lo, hi):
    import torch
    # `lo`/`hi` are bound for the kernel; reference uses the same Python
    # constants directly to avoid a GPU→CPU sync from .item() inside the
    # timing loop (same pattern as p005 leaky_relu).
    return torch.clamp(x, min=_LO, max=_HI)
