_B = 4096
_K = 4096

PROBLEM = {
    "id": "p314_swiglu",
    "tier": 4,
    "title": "SwiGLU gate (elementwise, dual-input)",
    "description": (
        f"out[b, k] = silu(gate[b, k]) * up[b, k]. Two ({_B}, {_K}) input "
        f"matrices. This is the elementwise gating step inside the SwiGLU "
        f"FFN block used in Llama, PaLM, Gemma — a fused sigmoid + multiply."
    ),
    "inputs": [
        {"name": "gate", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
        {"name": "up",   "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B, _K), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "entry_point": "swiglu",
    "notes": (
        "Elementwise: read two inputs, compute silu of one, multiply by the other, write. "
        "Perfect fusion candidate — no dependencies across threads."
    ),
}


def reference(gate, up):
    import torch
    import torch.nn.functional as F
    return F.silu(gate) * up
