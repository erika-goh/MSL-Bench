_B = 262144
_K = 256

PROBLEM = {
    "id": "p354_log_softmax",
    "tier": 4,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Row-wise log-softmax (fused max + sum-exp reduction)",
    "description": (
        "For each row b: out[b, k] = (x[b, k] - m) - log(sum_j exp(x[b, j] - m)), "
        f"where m = max_j x[b, j]. Input x ({_B}, {_K}) float32, output same "
        "shape. Fuses two reductions (max, then sum-of-exp) with the "
        "elementwise finish in one dispatch; the max shift keeps exp finite."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B, _K), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "log_softmax",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": "Two tree reductions with a barrier between: max first (read into "
             "a register), then sum of exp(x-max). out = (x-max) - log(sum).",
}


def reference(x):
    import torch.nn.functional as F
    return F.log_softmax(x, dim=1)
