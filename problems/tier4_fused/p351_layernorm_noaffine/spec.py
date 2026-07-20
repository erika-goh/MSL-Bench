_B = 262144
_K = 256

PROBLEM = {
    "id": "p351_layernorm_noaffine",
    "tier": 4,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Row-wise layernorm, no affine (fused mean+var reduction)",
    "description": (
        f"For each row b: out[b, k] = (x[b, k] - mean) / sqrt(var + eps), "
        f"with mean and var computed across the K={_K} columns. Input x "
        f"({_B}, {_K}) float32, output same shape. No gamma/beta. Fuses a "
        "combined sum + sum-of-squares reduction with the per-element "
        "normalize in one dispatch. eps = 1e-5."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B, _K), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "layernorm_noaffine",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": "Like p301 but without the gamma/beta affine. var via the "
             "identity sumsq/K - mean^2, so one combined tree-reduce.",
}


def reference(x):
    import torch.nn.functional as F
    return F.layer_norm(x, [_K], eps=1e-5)
