PROBLEM = {
    "id": "p015_silu",
    "tier": 1,
    "title": "SiLU (Swish) activation",
    "description": "out[i] = x[i] * sigmoid(x[i]) for a 1D float32 tensor of length 2^25.",
    "inputs": [
        {"name": "x", "shape": (33554432,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (33554432,), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "entry_point": "silu",
    "notes": "SiLU / Swish: x * sigmoid(x). Used in Llama, Gemma, PaLM.",
}


def reference(x):
    import torch
    return torch.nn.functional.silu(x)
