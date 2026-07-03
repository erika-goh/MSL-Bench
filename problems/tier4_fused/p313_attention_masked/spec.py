_SEQ = 512
_DIM = 64

PROBLEM = {
    "id": "p313_attention_masked",
    "tier": 4,
    "title": "Attention with explicit additive mask (single head)",
    "description": (
        f"Compute out = softmax(Q @ K^T / sqrt(dim) + mask) @ V. "
        f"Q, K, V each ({_SEQ}, {_DIM}). mask is ({_SEQ}, {_SEQ}) additive "
        f"(zeros = attend, -inf = block; padding-mask pattern). "
        f"Output ({_SEQ}, {_DIM})."
    ),
    "inputs": [
        {"name": "q",    "shape": (_SEQ, _DIM), "dtype": "float32", "init": "randn"},
        {"name": "k",    "shape": (_SEQ, _DIM), "dtype": "float32", "init": "randn"},
        {"name": "v",    "shape": (_SEQ, _DIM), "dtype": "float32", "init": "randn"},
        {"name": "mask", "shape": (_SEQ, _SEQ), "dtype": "float32", "init": "zeros"},
    ],
    "outputs": [
        {"name": "out", "shape": (_SEQ, _DIM), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-3, "rtol": 1e-3},
    "entry_point": "attention_masked",
    "notes": (
        "Same as p312 but with an explicit additive mask passed as input. "
        "All-zeros mask makes it equivalent to unmasked attention, so this "
        "measures mask-handling overhead when the mask is neutral. "
        "A real mask (padding, sliding-window) would exercise more paths."
    ),
}


def reference(q, k, v, mask):
    import torch
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
