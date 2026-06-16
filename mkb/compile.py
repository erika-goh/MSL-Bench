"""Compile a .metal source file into a .metallib via the Xcode toolchain.

Compiler diagnostics are captured verbatim — they are fed back to the model
in repair@k mode, so do not truncate or prettify them.

Launch configuration (grid + threadgroup) is owned by the problem spec, not
the kernel file — see mkb.problems.launch_config. The kernel is pure algorithm.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompileResult:
    ok: bool
    metallib: Path | None
    diagnostics: str


def compile_metal(src_path: Path, work_dir: Path) -> CompileResult:
    work_dir.mkdir(parents=True, exist_ok=True)

    air = work_dir / (src_path.stem + ".air")
    lib = work_dir / (src_path.stem + ".metallib")

    p1 = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(src_path), "-o", str(air)],
        capture_output=True, text=True,
    )
    if p1.returncode != 0:
        return CompileResult(False, None, p1.stderr.strip())

    p2 = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(lib)],
        capture_output=True, text=True,
    )
    if p2.returncode != 0:
        return CompileResult(False, None, p2.stderr.strip())

    return CompileResult(True, lib, (p1.stderr or "").strip())
