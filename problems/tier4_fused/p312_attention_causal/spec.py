_SEQ = 512
_DIM = 64

PROBLEM = {
    "id": "p312_attention_causal",
    "tier": 4,
    "title": "Causal self-attention (single head)",
    "description": (
        f"Compute out = softmax(mask(Q @ K^T / sqrt(dim))) @ V. "
        f"Q, K, V each ({_SEQ}, {_DIM}). Causal mask: positions i can only "
        f"attend to positions j <= i (upper triangle masked to -inf before softmax). "
        f"Output ({_SEQ}, {_DIM})."
    ),
    "inputs": [
        {"name": "q", "shape": (_SEQ, _DIM), "dtype": "float32", "init": "randn"},
        {"name": "k", "shape": (_SEQ, _DIM), "dtype": "float32", "init": "randn"},
        {"name": "v", "shape": (_SEQ, _DIM), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_SEQ, _DIM), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-3, "rtol": 1e-3},
    "entry_point": "attention_causal",
    "notes": (
        "Autoregressive attention pattern (GPT-style). Scale = 1/sqrt(dim) = 0.125. "
        "Upper-triangular of the attention matrix set to -inf pre-softmax. "
        "Flash-attention-style tiling is the target optimization."
    ),
}


def reference(q, k, v):
    import torch
    import torch.nn.functional as F
    # Use PyTorch's SDPA with is_causal=True for the reference.
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
