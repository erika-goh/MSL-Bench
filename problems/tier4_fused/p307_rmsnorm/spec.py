_M = 1024
_N = 4096
_EPS = 1e-6
_TG_THREADS = 256
_SGS_PER_TG = 8
_PER_THREAD = _N // _TG_THREADS    # 16

assert _N % _TG_THREADS == 0, "row length must divide TG_THREADS"

PROBLEM = {
    "id": "p307_rmsnorm",
    "tier": 4,
    "title": "RMSNorm (root-mean-square layer normalization)",
    "description": (
        f"Compute y[m, :] = x[m, :] * g / sqrt(mean(x[m, :]^2) + eps) "
        f"for x shape ({_M}, {_N}) and learnable scale g shape ({_N},). "
        f"Used in LLaMA, Mistral, Gemma, and most other modern "
        "decoder LLMs — replaces LayerNorm with a simpler statistic "
        "(no mean subtraction, only the per-row RMS). One TG per row, "
        f"{_TG_THREADS} threads each. Three phases: (1) two-stage "
        "simd_sum reduction of x[m, :]^2; (2) one thread computes "
        "inv_rms = rsqrt(sum_sq / N + eps) and broadcasts it via "
        "TG memory; (3) every thread writes its 16 outputs as "
        "x[m, j] * g[j] * inv_rms."
    ),
    "inputs": [
        {"name": "x", "shape": (_M, _N), "dtype": "float32", "init": "randn"},
        {"name": "g", "shape": (_N,),    "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "y", "shape": (_M, _N), "dtype": "float32"},
    ],
    # |y| ~ O(1) (x ~ N(0,1), g ~ N(0,1), rms ~ 1). Sum-of-squares
    # error ~sqrt(N)*eps*sum ≈ 4e-4 absolute on a sum of ~N=4096.
    # rsqrt preserves relative error. Final per-element error bound
    # ~1e-5; 1e-3 atol has two decades of slack.
    "tolerance": {"atol": 1e-3, "rtol": 1e-4},
    "entry_point": "rmsnorm_kernel",
    "launch": {
        "grid":        (_M * _TG_THREADS, 1, 1),
        "threadgroup": (_TG_THREADS,      1, 1),
    },
    "notes": (
        f"EPS = {_EPS}. M = {_M}, N = {_N}. {_M} TGs of {_TG_THREADS} "
        "threads (8 SGs of 32 lanes). Memory pattern: strip-mined "
        "x reads (thread t reads x[m, t], x[m, t+256], ..., x[m, "
        "t+15*256]). g[j] read by every thread at the corresponding "
        f"j — small enough (16 KB) to stay in L1 across the {_M} TGs."
    ),
}


def reference(x, g):
    import torch
    eps = 1e-6
    # x: (M, N), g: (N,)
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * g
