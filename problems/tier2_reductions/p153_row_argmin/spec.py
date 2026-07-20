_B = 262144
_K = 256

PROBLEM = {
    "id": "p153_row_argmin",
    "tier": 2,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Row-wise argmin (paired reduction, int32 output)",
    "description": (
        "For each row b, find the column index of the minimum value: "
        f"out[b] = argmin_j(x[b, j]). Input shape ({_B}, {_K}) float32; "
        f"output shape ({_B},) int32. One threadgroup per row, {_K} threads "
        "cooperate via a paired tree reduction carrying BOTH the candidate "
        "value AND its index. Ties resolve to the lowest index, matching "
        "torch.argmin."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _K), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B,), "dtype": "int32"},
    ],
    # Integer output: strict equality (randn makes ties ~impossible).
    "tolerance": {"atol": 0, "rtol": 0},
    "entry_point": "row_argmin",
    "launch": {
        "grid":        (_B * _K, 1, 1),
        "threadgroup": (_K,      1, 1),
    },
    "notes": "Same paired tree as p108_row_argmax but with strict `<`, so a "
             "tie keeps the existing (lower-index) survivor = torch.argmin.",
}


def reference(x):
    import torch
    return torch.argmin(x, dim=1).to(torch.int32)
