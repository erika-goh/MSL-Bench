_M    = 4096   # length of u  = rows of C
_N    = 4096   # length of v  = cols of C
_TILE = 16

PROBLEM = {
    "id": "p254_outer_product",
    "tier": 3,
    "split": "heldout",  # Phase-2.5 test set
    "title": "Tiled outer product u ⊗ v (threadgroup-staged vectors)",
    "description": (
        f"Compute C = u ⊗ v, i.e. C[i, j] = u[i] * v[j], for u of length "
        f"{_M} and v of length {_N}, C of shape ({_M}, {_N}), float32. "
        f"Each {_TILE}×{_TILE} threadgroup computes one output tile and "
        f"first stages the {_TILE} u-values and {_TILE} v-values its block "
        "needs into threadgroup memory (loaded by the first row / first "
        "column of threads), barriers, then every thread multiplies one "
        "staged u by one staged v. Reuses each loaded value TILE times."
    ),
    "inputs": [
        {"name": "u", "shape": (_M,), "dtype": "float32", "init": "randn"},
        {"name": "v", "shape": (_N,), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "c", "shape": (_M, _N), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "entry_point": "outer_product",
    "launch": {
        "grid":        (_N,    _M,    1),
        "threadgroup": (_TILE, _TILE, 1),
    },
    "notes": "Staging is the point: without it each element reloads u[i]/v[j] "
             "from device memory; with it, TILE u's and TILE v's are shared.",
}


def reference(u, v):
    import torch
    return torch.outer(u, v)
