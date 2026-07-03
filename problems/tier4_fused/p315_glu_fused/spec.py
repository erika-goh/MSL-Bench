_B = 4096
_K = 4096

PROBLEM = {
    "id": "p315_glu_fused",
    "tier": 4,
    "title": "GLU gate (sigmoid gating, dual-input)",
    "description": (
        f"out[b, k] = sigmoid(gate[b, k]) * up[b, k]. Two ({_B}, {_K}) input "
        f"matrices. Classic GLU (Gated Linear Unit) elementwise op — same "
        f"shape as SwiGLU but with sigmoid instead of silu."
    ),
    "inputs": [
        {"name": "gate", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
        {"name": "up",   "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B, _K), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "entry_point": "glu_fused",
    "notes": "Companion to p314_swiglu; measures the cost difference between silu and sigmoid.",
}


def reference(gate, up):
    import torch
    return torch.sigmoid(gate) * up
