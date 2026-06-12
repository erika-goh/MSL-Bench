"""Pure-Python tests — run anywhere, no Mac or torch required."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkb.score import fast_p, tier_table
from mkb.timing import summarize
from mkb.verify import verify
from mkb.compile import parse_launch_config


def test_verify_pass():
    ref = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    cand = {"out": np.array([1.0, 2.0, 3.0 + 1e-7], dtype=np.float32)}
    v = verify(cand, ref, atol=1e-5, rtol=1e-5)
    assert v.correct


def test_verify_fail_wrong_values():
    ref = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    cand = {"out": np.array([1.0, 2.0, 99.0], dtype=np.float32)}
    v = verify(cand, ref, atol=1e-5, rtol=1e-5)
    assert not v.correct
    assert "out of tolerance" in v.detail


def test_verify_fail_nan():
    ref = {"out": np.array([1.0], dtype=np.float32)}
    cand = {"out": np.array([np.nan], dtype=np.float32)}
    assert not verify(cand, ref, atol=1e-5, rtol=1e-5).correct


def test_verify_fail_shape():
    ref = {"out": np.zeros(4, dtype=np.float32)}
    cand = {"out": np.zeros(5, dtype=np.float32)}
    assert not verify(cand, ref, atol=1e-5, rtol=1e-5).correct


def test_fast_p():
    recs = [
        {"tier": 1, "correct": True, "speedup": 2.5},
        {"tier": 1, "correct": True, "speedup": 0.8},
        {"tier": 1, "correct": False, "speedup": None},
        {"tier": 2, "correct": True, "speedup": 1.2},
    ]
    assert fast_p(recs, 0) == 0.75
    assert fast_p(recs, 1) == 0.5
    assert fast_p(recs, 2) == 0.25
    t = tier_table(recs)
    assert t[1]["n"] == 3 and t[2]["n"] == 1


def test_parse_launch_config():
    src = "// MKB_GRID: 1048576 1 1\n// MKB_TG: 256 1 1\nkernel void f(){}"
    grid, tg = parse_launch_config(src)
    assert grid == (1048576, 1, 1)
    assert tg == (256, 1, 1)


def test_summarize_flags_noisy():
    quiet = summarize([1.0, 1.01, 0.99, 1.0, 1.02, 0.98, 1.0, 1.01, 0.99, 1.0])
    assert not quiet.noisy
    noisy = summarize([1.0, 3.0, 0.5, 2.5, 1.0, 4.0, 0.7, 2.0, 1.5, 3.5])
    assert noisy.noisy
