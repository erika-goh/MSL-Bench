_B = 4096
_K = 4096

PROBLEM = {
    "id": "p310_bias_add_relu",
    "tier": 4,
    "title": "Fused bias-add + ReLU",
    "description": (
        f"out[b, k] = max(0, x[b, k] + bias[k]). Input x ({_B}, {_K}), "
        f"bias ({_K},) broadcast along the batch dim. Fuses two passes "
        f"(add-bias then relu) into one kernel."
    ),
    "inputs": [
        {"name": "x",    "shape": (_B, _K), "dtype": "float32", "init": "randn"},
        {"name": "bias", "shape": (_K,),    "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B, _K), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "bias_add_relu",
    "notes": "Element-wise with broadcast lookup. Fusion test: MPS may or may not fuse this pair.",
}


def reference(x, bias):
    import torch
    import torch.nn.functional as F
    return F.relu(x + bias)
