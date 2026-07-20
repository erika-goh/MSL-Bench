"""LLM provider clients — free-tier and local only.

Providers:
- groq:   free tier, OpenAI-compatible API.  env: GROQ_API_KEY
- gemini: free tier (Gemini Flash).           env: GEMINI_API_KEY
- ollama: fully local, no key.                env: OLLAMA_HOST (optional)
- local:  MLX-LM server, OpenAI-compatible.   env: MLX_HOST (optional)

All providers expose the same call signature:
    complete(messages: list[{"role","content"}], model: str) -> str

Adding a paid provider later is one more function here — nothing else changes.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


USER_AGENT = "MSL-Bench/0.1 (+https://github.com/erika-goh/MSL-Bench)"


def _post_json(url: str, payload: dict, headers: dict, retries: int = 3) -> dict:
    # Groq is fronted by Cloudflare, which 403s the default `Python-urllib/x.y`
    # UA as a bot signature (error 1010). A real UA gets us through.
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": USER_AGENT, **headers})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            transient = e.code == 429 or 500 <= e.code < 600
            if transient and attempt < retries - 1:
                # 429 = rate-limited, 5xx = transient upstream error;
                # both are worth retrying with backoff before giving up.
                time.sleep(15 * (attempt + 1))
                continue
            safe_url = url.split("?", 1)[0]  # Gemini puts the API key in the query string
            raise RuntimeError(f"HTTP {e.code} from {safe_url}: {e.read().decode()[:500]}") from e
    raise RuntimeError("unreachable")


DEFAULT_MAX_TOKENS = 4096


def groq(messages: list[dict], model: str = "llama-3.3-70b-versatile",
         max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    res = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens},
        {"Authorization": f"Bearer {key}"},
    )
    return res["choices"][0]["message"]["content"]


def gemini(messages: list[dict], model: str = "gemini-2.0-flash",
           max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in messages if m["role"] != "system"
    ]
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    payload: dict = {"contents": contents,
                     "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    res = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        payload, {},
    )
    return res["candidates"][0]["content"]["parts"][0]["text"]


def ollama(messages: list[dict], model: str = "qwen2.5-coder:14b",
           max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    res = _post_json(
        f"{host}/api/chat",
        {"model": model, "messages": messages, "stream": False,
         "options": {"temperature": 0.2, "num_predict": max_tokens}},
        {},
    )
    return res["message"]["content"]


def local(messages: list[dict], model: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
          max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """MLX-LM server (OpenAI-compatible) serving a base model + LoRA adapter.

    Start it with, e.g.:
        mlx_lm.server --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \\
                      --adapter-path phase6/adapters_iter50
    The `model` field MUST match the server's --model: the server keys on it to
    reuse the preloaded (adapter-applied) instance. An unknown value is treated
    as a HF repo id and download-attempted. env: MLX_HOST (default :8080).
    """
    host = os.environ.get("MLX_HOST", "http://localhost:8080")
    res = _post_json(
        f"{host}/v1/chat/completions",
        {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens},
        {},
    )
    return res["choices"][0]["message"]["content"]


PROVIDERS = {"groq": groq, "gemini": gemini, "ollama": ollama, "local": local}


def complete(provider: str, messages: list[dict], model: str | None = None,
             max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    fn = PROVIDERS[provider]
    kwargs = {"max_tokens": max_tokens}
    if model:
        return fn(messages, model, **kwargs)
    return fn(messages, **kwargs)
