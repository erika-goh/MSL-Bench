"""Compile a .metal source file into a .metallib via the Xcode toolchain.

Compiler diagnostics are captured verbatim — they are fed back to the model
in repair@k mode, so do not truncate or prettify them.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompileResult:
    ok: bool
    metallib: Path | None
    diagnostics: str
    grid: tuple[int, int, int] | None
    threadgroup: tuple[int, int, int] | None


_GRID_RE = re.compile(r"//\s*MKB_GRID:\s*(\d+)\s+(\d+)\s+(\d+)")
_TG_RE = re.compile(r"//\s*MKB_TG:\s*(\d+)\s+(\d+)\s+(\d+)")


def parse_launch_config(src: str) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    """Candidate kernels declare their own launch config via magic comments.

    Getting the grid right is part of the task — a kernel with no MKB_GRID
    comment fails with a clear error rather than a guessed default.
    """
    g = _GRID_RE.search(src)
    t = _TG_RE.search(src)
    grid = tuple(int(x) for x in g.groups()) if g else None
    tg = tuple(int(x) for x in t.groups()) if t else None
    return grid, tg  # type: ignore[return-value]


def compile_metal(src_path: Path, work_dir: Path) -> CompileResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    src = src_path.read_text()
    grid, tg = parse_launch_config(src)

    air = work_dir / (src_path.stem + ".air")
    lib = work_dir / (src_path.stem + ".metallib")

    p1 = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(src_path), "-o", str(air)],
        capture_output=True, text=True,
    )
    if p1.returncode != 0:
        return CompileResult(False, None, p1.stderr.strip(), grid, tg)

    p2 = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(lib)],
        capture_output=True, text=True,
    )
    if p2.returncode != 0:
        return CompileResult(False, None, p2.stderr.strip(), grid, tg)

    return CompileResult(True, lib, (p1.stderr or "").strip(), grid, tg)
