"""End-to-end harness tests — REQUIRE macOS + Xcode + built runner.

Run on your Mac after `make runner`:
    pytest tests/test_harness.py -v
"""
import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

needs_mac = pytest.mark.skipif(platform.system() != "Darwin", reason="requires macOS + Metal")


@needs_mac
def test_golden_vector_add_passes():
    out = subprocess.run(
        [sys.executable, "scripts/run_problem.py", "p001_vector_add",
         "tests/golden_kernels/vector_add.metal"],
        capture_output=True, text=True, cwd=ROOT,
    )
    res = json.loads(out.stdout)
    assert res["compiled"], res.get("diagnostics")
    assert res["correct"], res.get("verify_detail")
    assert res["speedup"] is not None


@needs_mac
def test_golden_relu_passes():
    out = subprocess.run(
        [sys.executable, "scripts/run_problem.py", "p002_relu",
         "tests/golden_kernels/relu.metal"],
        capture_output=True, text=True, cwd=ROOT,
    )
    res = json.loads(out.stdout)
    assert res["compiled"] and res["correct"]


@needs_mac
def test_wrong_kernel_fails(tmp_path):
    bad = tmp_path / "bad.metal"
    bad.write_text("""#include <metal_stdlib>
using namespace metal;
// MKB_GRID: 1048576 1 1
// MKB_TG: 256 1 1
kernel void vector_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    out[i] = a[i] - b[i];  // wrong on purpose
}""")
    out = subprocess.run(
        [sys.executable, "scripts/run_problem.py", "p001_vector_add", str(bad)],
        capture_output=True, text=True, cwd=ROOT,
    )
    res = json.loads(out.stdout)
    assert res["compiled"] and not res["correct"]


@needs_mac
def test_noncompiling_kernel_reports_diagnostics(tmp_path):
    bad = tmp_path / "broken.metal"
    bad.write_text("kernel void vector_add( this is not valid MSL")
    out = subprocess.run(
        [sys.executable, "scripts/run_problem.py", "p001_vector_add", str(bad)],
        capture_output=True, text=True, cwd=ROOT,
    )
    res = json.loads(out.stdout)
    assert not res["compiled"]
    assert len(res["diagnostics"]) > 0
