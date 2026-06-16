"""Pure-Python tests — run anywhere, no Mac or torch required."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkb import timing as _timing
from mkb.problems import launch_config
from mkb.score import fast_p, tier_table
from mkb.timing import (
    CALIBRATION_SCHEMA_VERSION,
    STABILITY_THRESHOLD_FRAC,
    check_calibration,
    check_stability,
    record_calibration,
    summarize,
)
from mkb.verify import verify


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


# ---------- launch_config (spec-owned grid derivation) ----------

def test_launch_config_default_derives_grid_from_output_shape():
    p = {"id": "p_test", "outputs": [{"name": "out", "shape": (1 << 24,), "dtype": "float32"}]}
    grid, tg = launch_config(p)
    assert grid == (1 << 24, 1, 1)
    assert tg == (256, 1, 1)


def test_launch_config_2d_output_flattens_to_1d_dispatch():
    p = {"id": "p_test", "outputs": [{"name": "C", "shape": (1024, 1024), "dtype": "float32"}]}
    grid, _ = launch_config(p)
    assert grid == (1024 * 1024, 1, 1)


def test_launch_config_override_grid_only_keeps_default_tg():
    p = {"id": "p_test",
         "outputs": [{"name": "out", "shape": (4096,), "dtype": "float32"}],
         "launch": {"grid": (4096, 1, 1)}}
    grid, tg = launch_config(p)
    assert grid == (4096, 1, 1)
    assert tg == (256, 1, 1)


def test_launch_config_override_both():
    p = {"id": "p_test",
         "outputs": [{"name": "out", "shape": (4096,), "dtype": "float32"}],
         "launch": {"grid": (64, 1, 1), "threadgroup": (128, 1, 1)}}
    grid, tg = launch_config(p)
    assert grid == (64, 1, 1)
    assert tg == (128, 1, 1)


def test_make_inputs_constant_init():
    from mkb.problems import make_inputs
    p = {"inputs": [{"name": "alpha", "shape": (1,), "dtype": "float32",
                     "init": "constant", "value": 0.01}]}
    out = make_inputs(p)
    assert out["alpha"].shape == (1,)
    assert out["alpha"].dtype == np.float32
    assert out["alpha"][0] == np.float32(0.01)


def test_launch_config_rejects_non_3_tuple():
    p = {"id": "p_test",
         "outputs": [{"name": "out", "shape": (16,), "dtype": "float32"}],
         "launch": {"grid": (16, 1)}}  # 2-tuple, not 3
    with pytest.raises(ValueError, match="3-tuples"):
        launch_config(p)


def test_summarize_flags_noisy():
    quiet = summarize([1.0, 1.01, 0.99, 1.0, 1.02, 0.98, 1.0, 1.01, 0.99, 1.0])
    assert not quiet.noisy
    noisy = summarize([1.0, 3.0, 0.5, 2.5, 1.0, 4.0, 0.7, 2.0, 1.5, 3.5])
    assert noisy.noisy


# ---------- calibration ----------

_FAKE_ENV = {"macos": "26.5", "device": "Apple Test", "python": "3.12.0", "torch": "2.12.0"}


def _patch_env(monkeypatch, env):
    monkeypatch.setattr(_timing, "_current_env_metadata", lambda: dict(env))


def test_check_calibration_no_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(_timing, "CALIBRATION_FILE", tmp_path / "missing.json")
    _patch_env(monkeypatch, _FAKE_ENV)
    assert check_calibration("p001_vector_add", 0.05) is None


def test_check_calibration_old_schema_demands_rerecord(tmp_path, monkeypatch):
    f = tmp_path / "calib.json"
    f.write_text(json.dumps({"p001_vector_add": 0.05}))  # pre-Phase-1 schema
    monkeypatch.setattr(_timing, "CALIBRATION_FILE", f)
    _patch_env(monkeypatch, _FAKE_ENV)
    msg = check_calibration("p001_vector_add", 0.05)
    assert msg is not None and "schema" in msg and "re-record" in msg


def test_check_calibration_env_mismatch_marks_stale(tmp_path, monkeypatch):
    f = tmp_path / "calib.json"
    monkeypatch.setattr(_timing, "CALIBRATION_FILE", f)
    _patch_env(monkeypatch, _FAKE_ENV)
    record_calibration("p001_vector_add", 0.05)  # records under _FAKE_ENV
    # Now pretend torch was upgraded between recording and checking.
    _patch_env(monkeypatch, {**_FAKE_ENV, "torch": "2.13.0"})
    msg = check_calibration("p001_vector_add", 0.05)
    assert msg is not None
    assert "stale" in msg and "torch" in msg
    assert "'2.12.0'" in msg and "'2.13.0'" in msg  # names both sides per error-surfaces rule


def test_check_calibration_env_match_within_drift_passes(tmp_path, monkeypatch):
    f = tmp_path / "calib.json"
    monkeypatch.setattr(_timing, "CALIBRATION_FILE", f)
    _patch_env(monkeypatch, _FAKE_ENV)
    record_calibration("p001_vector_add", 0.05)
    assert check_calibration("p001_vector_add", 0.051) is None  # ~2% drift, under 10% threshold


def test_check_calibration_env_match_drift_warns(tmp_path, monkeypatch):
    f = tmp_path / "calib.json"
    monkeypatch.setattr(_timing, "CALIBRATION_FILE", f)
    _patch_env(monkeypatch, _FAKE_ENV)
    record_calibration("p001_vector_add", 0.05)
    msg = check_calibration("p001_vector_add", 0.07)  # 40% drift
    assert msg is not None and "drift" in msg and "stale" not in msg


def test_check_calibration_unknown_kernel_returns_none(tmp_path, monkeypatch):
    f = tmp_path / "calib.json"
    monkeypatch.setattr(_timing, "CALIBRATION_FILE", f)
    _patch_env(monkeypatch, _FAKE_ENV)
    record_calibration("p001_vector_add", 0.05)
    assert check_calibration("p999_unrecorded", 0.05) is None


def test_record_calibration_preserves_sibling_entries_on_same_schema(tmp_path, monkeypatch):
    f = tmp_path / "calib.json"
    monkeypatch.setattr(_timing, "CALIBRATION_FILE", f)
    _patch_env(monkeypatch, _FAKE_ENV)
    record_calibration("p001_vector_add", 0.05)
    record_calibration("p002_relu", 0.03)
    data = json.loads(f.read_text())
    assert data["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert data["p001_vector_add"]["median_ms"] == 0.05
    assert data["p002_relu"]["median_ms"] == 0.03


# ---------- A/B/A stability ----------

def _stats(median_ms: float):
    # Build a TimingStats-shaped object via summarize; samples shape doesn't matter
    # for check_stability since it only reads median_ms.
    return summarize([median_ms] * 10)


def test_check_stability_pass_when_blocks_agree():
    s = check_stability(_stats(0.100), _stats(0.103))  # 3% delta, under 7%
    assert s.stable
    assert s.delta_frac < STABILITY_THRESHOLD_FRAC
    assert s.message == "ok"


def test_check_stability_fail_names_both_medians_and_delta():
    s = check_stability(_stats(0.100), _stats(0.120))  # block 3 20% slower
    assert not s.stable
    # Error-surfaces rule: message must say which blocks disagreed and by how much.
    assert "0.1000" in s.message and "0.1200" in s.message
    assert "20.0%" in s.message and "7%" in s.message
    assert "untrustworthy" in s.message


def test_check_stability_message_reports_direction_and_likely_cause():
    # Block 3 slower → thermal-throttling hint.
    s_slow = check_stability(_stats(0.100), _stats(0.120))
    assert "slower" in s_slow.message
    assert "thermally throttling" in s_slow.message or "cool" in s_slow.message
    # Block 3 faster → cold-pipeline hint.
    s_fast = check_stability(_stats(0.120), _stats(0.100))
    assert "faster" in s_fast.message
    assert "warmup" in s_fast.message or "cold" in s_fast.message


def test_check_stability_zero_block1_is_failure():
    s = check_stability(_stats(0.0), _stats(0.05))
    assert not s.stable
    assert "block 1 median was zero" in s.message


def test_check_stability_respects_explicit_threshold():
    # Same medians, but a much tighter threshold flips the verdict.
    a, b = _stats(0.100), _stats(0.105)  # 5% delta
    assert check_stability(a, b, threshold=0.10).stable      # passes at 10%
    assert not check_stability(a, b, threshold=0.02).stable  # fails at 2%
