"""Tests for LLM response parsing. Pure Python — no Mac or torch required."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkb.llm.generate import extract_metal

KERNEL = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void foo(device float* out [[buffer(0)]],\n"
    "                uint gid [[thread_position_in_grid]]) {\n"
    "    out[gid] = 1.0f;\n"
    "}"
)


def test_extract_metal_fence():
    assert extract_metal(f"prose\n```metal\n{KERNEL}\n```\ntrailing") == KERNEL


def test_extract_cpp_fence():
    assert extract_metal(f"```cpp\n{KERNEL}\n```") == KERNEL


def test_extract_c_plus_plus_fence():
    assert extract_metal(f"```c++\n{KERNEL}\n```") == KERNEL


def test_extract_c_fence():
    assert extract_metal(f"```c\n{KERNEL}\n```") == KERNEL


def test_extract_objc_fence():
    assert extract_metal(f"```objc\n{KERNEL}\n```") == KERNEL


def test_extract_unlabeled_fence():
    assert extract_metal(f"```\n{KERNEL}\n```") == KERNEL


def test_extract_fallback_when_no_fence():
    assert extract_metal(KERNEL) == KERNEL


def test_extract_returns_none_on_pure_prose():
    assert extract_metal("Sorry, I can't help with that request.") is None


def test_extract_first_of_multiple_fences():
    first = "kernel void first() {}"
    second = "kernel void second() {}"
    out = extract_metal(f"```metal\n{first}\n```\nexplanation\n```metal\n{second}\n```")
    assert "first" in out and "second" not in out
