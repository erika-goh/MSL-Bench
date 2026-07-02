#!/usr/bin/env python3
"""Preflight check for Phase-3 providers.

Pings each provider with a minimal prompt and reports auth / reachability.
Skips providers whose credentials aren't set (so you can preflight subsets).

Example:
    python scripts/preflight.py                              # all providers, default models
    python scripts/preflight.py --provider groq gemini       # subset
    python scripts/preflight.py --model qwen2.5-coder:14b    # override ollama model
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkb.llm import providers as P

PING = [{"role": "user", "content": "Reply with exactly one word: pong."}]

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "ollama": "qwen2.5-coder:14b",
}

CREDS = {
    "groq": ("GROQ_API_KEY", "env var"),
    "gemini": ("GEMINI_API_KEY", "env var"),
    "ollama": ("OLLAMA_HOST", "env var (optional, defaults to http://localhost:11434)"),
}


def check(provider: str, model: str) -> tuple[str, str, float]:
    """Returns (status, detail, elapsed_seconds). status in {ok, fail, skip}."""
    env_name, _ = CREDS[provider]
    if provider != "ollama" and not os.environ.get(env_name):
        return "skip", f"{env_name} not set", 0.0
    t0 = time.perf_counter()
    try:
        reply = P.complete(provider, PING, model)
    except Exception as e:
        return "fail", f"{type(e).__name__}: {str(e)[:200]}", time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    snippet = reply.strip().replace("\n", " ")[:60]
    return "ok", f'model={model} reply="{snippet}"', elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", nargs="+", choices=list(DEFAULT_MODELS),
                    default=list(DEFAULT_MODELS))
    ap.add_argument("--model", default=None,
                    help="override the model for whichever provider matches; otherwise defaults are used")
    args = ap.parse_args()

    any_fail = False
    for provider in args.provider:
        model = args.model or DEFAULT_MODELS[provider]
        status, detail, elapsed = check(provider, model)
        marker = {"ok": "[OK]  ", "fail": "[FAIL]", "skip": "[SKIP]"}[status]
        print(f"{marker} {provider:8s} {elapsed:5.2f}s  {detail}")
        if status == "fail":
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
