_B = 262144
_K = 256

PROBLEM = {
    "id": "p309_batchnorm",
    "tier": 4,
    "title": "BatchNorm inference (fused normalize + affine)",
    "description": (
        f"out[b, k] = gamma[k] * (x[b, k] - mean[k]) / sqrt(var[k] + eps) + beta[k]. "
        f"Inference mode (running stats given, not computed). "
        f"Input x ({_B}, {_K}), mean/var/gamma/beta each ({_K},), output same shape as x."
    ),
    "inputs": [
        {"name": "x",     "shape": (_B, _K), "dtype": "float32", "init": "randn"},
        {"name": "mean",  "shape": (_K,),    "dtype": "float32", "init": "zeros"},
        {"name": "var",   "shape": (_K,),    "dtype": "float32", "init": "constant", "value": 1.0},
        {"name": "gamma", "shape": (_K,),    "dtype": "float32", "init": "constant", "value": 1.0},
        {"name": "beta",  "shape": (_K,),    "dtype": "float32", "init": "zeros"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B, _K), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "batchnorm",
    "notes": (
        "Inference-mode batch norm (no reduction — running stats are given). "
        "Pure element-wise with a broadcast lookup over the K-axis. "
        "eps = 1e-5 matches PyTorch default."
    ),
}


def reference(x, mean, var, gamma, beta):
    import torch
    import torch.nn.functional as F
    return F.batch_norm(x, mean, var, weight=gamma, bias=beta, training=False, eps=1e-5)
