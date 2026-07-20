_B = 262144
_C = 256   # channels
_G = 8     # groups -> 32 channels per group

PROBLEM = {
    "id": "p353_groupnorm",
    "tier": 4,
    "split": "heldout",  # Phase-2.5 test set
    "title": "GroupNorm over channels, no affine (segmented fused reduction)",
    "description": (
        f"For each sample b and each of {_G} contiguous channel groups "
        f"(C={_C} channels, {_C // _G} per group): normalize the group's "
        "channels to zero mean / unit variance: out = (x - mean_g) / "
        "sqrt(var_g + eps). Input x "
        f"({_B}, {_C}) float32, output same shape, no affine. One "
        f"threadgroup per sample ({_C} threads); each group reduces "
        "independently via a segmented tree reduction within its "
        f"{_C // _G}-thread block. eps = 1e-5."
    ),
    "inputs": [
        {"name": "x", "shape": (_B, _C), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "out", "shape": (_B, _C), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "entry_point": "groupnorm",
    "launch": {
        "grid":        (_B * _C, 1, 1),
        "threadgroup": (_C,      1, 1),
    },
    "notes": "Segmented reduction: groups are 32-aligned contiguous blocks, "
             "so a stride<32 tree reduce keeps partners within one group; "
             "each thread reads its group's totals from the block base.",
}


def reference(x):
    import torch.nn.functional as F
    return F.group_norm(x, _G, eps=1e-5)
