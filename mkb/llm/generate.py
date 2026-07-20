"""Prompt construction and generation loops (one-shot and repair@k)."""
from __future__ import annotations

import re

from .. import problems as _problems
from . import providers

__all__ = ["build_prompt", "extract_metal", "one_shot", "repair_k"]

SYSTEM = """You are an expert Metal Shading Language (MSL) programmer writing compute \
kernels for Apple Silicon GPUs. You output a single, complete, self-contained .metal \
file and nothing else."""

CONVENTIONS = """Conventions (required):
1. Output exactly one ```metal code block containing the full kernel file.
2. The kernel function must be named `{entry_point}`.
3. Buffer bindings: inputs in the order listed get [[buffer(0)]] .. [[buffer(N-1)]]; \
outputs follow, continuing the numbering.
4. The harness dispatches with the grid and threadgroup dimensions given \
under "Dispatch geometry" — do NOT declare launch config inside the kernel \
file. The harness uses `dispatchThreads`, so grid need not be a multiple of \
threadgroup.
5. float32 unless stated otherwise. No host code — kernel file only."""

TEMPLATE = """Write a Metal compute kernel for the following problem.

Problem: {title}
{description}

Inputs (bound in this order):
{inputs}

Outputs (bound after inputs, in this order):
{outputs}

Dispatch geometry (fixed by the harness):
{launch}

{notes}

{conventions}"""


def build_prompt(problem: dict) -> list[dict]:
    inputs = "\n".join(
        f"  - buffer({i}) {x['name']}: shape {tuple(x['shape'])}, {x['dtype']}"
        for i, x in enumerate(problem["inputs"])
    )
    n_in = len(problem["inputs"])
    outputs = "\n".join(
        f"  - buffer({n_in + i}) {x['name']}: shape {tuple(x['shape'])}, {x['dtype']}"
        for i, x in enumerate(problem["outputs"])
    )
    grid, tg = _problems.launch_config(problem)
    n_groups = tuple(-(-g // t) for g, t in zip(grid, tg))  # ceil-div per axis
    launch = (
        f"  - grid        = {grid}   (total threads dispatched, per axis)\n"
        f"  - threadgroup = {tg}   (threads per group, per axis)\n"
        f"  - -> {n_groups[0] * n_groups[1] * n_groups[2]} threadgroups total "
        f"({' x '.join(str(n) for n in n_groups)})"
    )
    user = TEMPLATE.format(
        title=problem["title"],
        description=problem["description"],
        inputs=inputs,
        outputs=outputs,
        launch=launch,
        notes=f"Notes: {problem['notes']}" if problem.get("notes") else "",
        conventions=CONVENTIONS.format(entry_point=problem["entry_point"]),
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_BLOCK_RE = re.compile(r"```(?:metal|objc|cpp|c\+\+|c)?\s*\n(.*?)```", re.DOTALL)


def extract_metal(text: str) -> str | None:
    m = _BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: model ignored fencing but output looks like a kernel file
    if "kernel void" in text:
        return text.strip()
    return None


def one_shot(provider: str, problem: dict, model: str | None = None,
             max_tokens: int = providers.DEFAULT_MAX_TOKENS) -> dict:
    messages = build_prompt(problem)
    raw = providers.complete(provider, messages, model, max_tokens=max_tokens)
    return {"mode": "one_shot", "attempts": 1,
            "kernel": extract_metal(raw), "transcript": messages + [{"role": "assistant", "content": raw}]}


def repair_k(provider: str, problem: dict, feedback_fn, k: int = 5, model: str | None = None,
             max_tokens: int = providers.DEFAULT_MAX_TOKENS) -> dict:
    """Iterative repair loop.

    feedback_fn(kernel_src: str) -> (success: bool, feedback: str)
    The caller owns compile/run/verify; this loop only manages the conversation.
    The full transcript is saved — these transcripts are the Phase-5 flywheel's
    seed trajectories, so never throw them away.
    """
    messages = build_prompt(problem)
    kernel = None
    for attempt in range(1, k + 1):
        try:
            raw = providers.complete(provider, messages, model, max_tokens=max_tokens)
        except Exception as e:
            # Provider error mid-loop (e.g. Groq TPM 413 once the transcript has
            # grown past the per-minute cap). Preserve the attempts collected so
            # far — a partial fail->feedback->retry trajectory is still a valid
            # flywheel seed. Never let the error discard it (the codebase's
            # "never throw transcripts away" rule). Record it so callers can tell
            # a provider abort from an honest repair@k exhaustion.
            messages.append({"role": "error", "content": f"{type(e).__name__}: {str(e)[:300]}"})
            return {"mode": f"repair@{k}", "attempts": attempt - 1, "kernel": kernel,
                    "success": False, "transcript": messages, "provider_error": str(e)[:300]}
        messages.append({"role": "assistant", "content": raw})
        kernel = extract_metal(raw)
        if kernel is None:
            messages.append({"role": "user",
                             "content": "Your reply contained no ```metal code block. "
                                        "Output the complete kernel file in one ```metal block."})
            continue
        success, feedback = feedback_fn(kernel)
        if success:
            return {"mode": f"repair@{k}", "attempts": attempt, "kernel": kernel,
                    "success": True, "transcript": messages}
        messages.append({"role": "user",
                         "content": f"That kernel failed. Feedback:\n{feedback}\n\n"
                                    f"Fix the kernel. Output the complete corrected file "
                                    f"in one ```metal block."})
    return {"mode": f"repair@{k}", "attempts": k, "kernel": kernel,
            "success": False, "transcript": messages}
