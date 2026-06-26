_M       = 1024
_N       = 1024
_K       = 1024
_TG_M    = 16
_TG_N    = 16
_K_STAGE = 32
_SG      = 8
_TGS     = 32

assert _K % _K_STAGE == 0, "K must be a multiple of K_STAGE"

PROBLEM = {
    "id": "p204_matmul_double_buffered_backfires",
    "tier": 3,
    "title": "Matmul: textbook double-buffering — and why it makes things worse",
    "description": (
        "Compute C = A @ B for 1024x1024 float32 matrices. Layers "
        "textbook double-buffering on top of p203: two sets of A/B "
        "stage buffers, so the next K-slab loads from device memory "
        "in parallel with matrix-unit ops on the current slab. The "
        "kernel is structurally textbook-correct and verifies bit-"
        "exact, but it runs ~45% SLOWER than p203. Two likely "
        "causes: (1) Apple's Metal compiler already pipelined p203's "
        "loads with compute implicitly (the dependency graph allowed "
        "it), so explicit double-buffering adds no new information; "
        "(2) 2x threadgroup-memory footprint (8KB vs 4KB per TG) "
        "reduces GPU occupancy, weakening the cores' ability to hide "
        "load latency by switching threadgroups. Shipped as a "
        "teaching artifact like p105 — kernels that an LLM might "
        "propose with confidence but that don't actually win."
    ),
    "inputs": [
        {"name": "a", "shape": (_M, _K), "dtype": "float32", "init": "randn"},
        {"name": "b", "shape": (_K, _N), "dtype": "float32", "init": "randn"},
    ],
    "outputs": [
        {"name": "c", "shape": (_M, _N), "dtype": "float32"},
    ],
    "tolerance": {"atol": 1e-2, "rtol": 1e-3},
    "entry_point": "matmul_double_buffered_backfires",
    "launch": {
        "grid":        (_N // _TG_N * _TGS, _M // _TG_M, 1),
        "threadgroup": (_TGS,                1,          1),
    },
    "notes": (
        "Measured: kernel_ms 1.48 (p203: 1.02), bit-exact correctness. "
        "Lesson: textbook GPU optimizations are contextual. An "
        "optimization that helps when the compiler can't see the "
        "opportunity (CUDA, older GPUs) can be net-negative when the "
        "compiler already does it AND the explicit version costs "
        "occupancy resources the implicit version didn't."
    ),
}


def reference(a, b):
    import torch
    return a @ b
