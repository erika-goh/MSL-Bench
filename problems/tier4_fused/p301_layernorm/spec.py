_B = 262144   # 2^18 rows
_K = 256      # row width — also TG size

PROBLEM = {
    "id": "p301_layernorm",
    "tier": 4,
    "title": "Row-wise layernorm (fused reduction + elementwise affine)",
    "description": (
        "For each row b: compute mean and variance across the K=256 "
        "columns, then output (x[b, k] - mean) / sqrt(var + eps) * "
        "gamma[k] + beta[k]. Input x has shape (B, K) = (262144, 256). "
        "gamma and beta are per-column scale and offset of shape (K,). "
        "Output has the same shape as x. The kernel fuses two reductions "
        "(sum + sumsq, combined into one tree-reduce carrying both) with "
        "a per-element affine transform — all in one dispatch. "
        "Tests the project's fusion thesis: MPS may dispatch layernorm "
        "as separate kernels (mean → variance → normalize), and our "
        "single-kernel fusion would win; or MPS already has it fused, "
        "in which case we land roughly even."
    ),
    "inputs": [
        {"name": "x",     "shape": (_B, _K), "dtype": "float32", "init": "randn"},
        {"name": "gamma", "shape": (_K,),    "dtype": "float32", "init": "constant", "value": 1.0},
        {"name": "beta",  "shape": (_K,),    "dtype": "float32", "init": "zeros"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B, _K), "dtype": "float32"},
    ],
    # Errors compound through mean + var + division + affine. Per-element
    # output is roughly N(0, 1) after layernorm; mean reduction error
    # ~sqrt(K)*eps*|sum| ~1e-5; var slightly more. atol=1e-4 is comfortable.
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "row_layernorm",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": (
        "Combined sum+sumsq reduction: two threadgroup-shared scratch "
        "arrays (or a paired-value array) accumulated in one tree pass "
        "instead of two separate reductions. Var = sumsq/K - mean^2 "
        "uses the algebraic identity rather than a second pass over "
        "(x-mean)^2. eps = 1e-5 to match torch.nn.functional.layer_norm "
        "default."
    ),
}


def reference(x, gamma, beta):
    import torch
    import torch.nn.functional as F
    return F.layer_norm(x, [_K], weight=gamma, bias=beta, eps=1e-5)
