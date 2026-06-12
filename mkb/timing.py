"""Timing utilities — the trust layer for benchmark numbers.

Laptop thermals are the #1 validity threat. Policy:
- warmup runs before any measurement
- median-of-N with IQR reported; runs with IQR > 15% of median are flagged noisy
- a calibration check that re-times a golden kernel and warns on >10% drift
- speedup is a ratio of medians measured in the same session, so thermal drift
  hits candidate and reference roughly equally
"""
from __future__ import annotations

import datetime as _dt
import json
import platform as _platform
import statistics
import subprocess as _subprocess
import sys as _sys
import time
from dataclasses import dataclass
from pathlib import Path

CALIBRATION_FILE = Path(__file__).resolve().parents[1] / "results" / "calibration.json"
NOISY_IQR_FRAC = 0.15
DRIFT_WARN_FRAC = 0.10
CALIBRATION_SCHEMA_VERSION = 1
# Fields that must match between baseline and current env, else baseline is stale.
_CRITICAL_ENV_FIELDS = ("macos", "device", "python", "torch")
# Max allowed delta between candidate medians in blocks 1 and 3 of an A/B/A
# timing run before we declare the session thermally unstable. Picked blind
# at 7%; tune once we have a feel for actual stable-session variance.
STABILITY_THRESHOLD_FRAC = 0.07


@dataclass
class TimingStats:
    median_ms: float
    iqr_ms: float
    noisy: bool
    samples: list[float]


@dataclass
class StabilityResult:
    stable: bool
    delta_frac: float
    message: str


def check_stability(
    block1: TimingStats,
    block3: TimingStats,
    threshold: float = STABILITY_THRESHOLD_FRAC,
) -> StabilityResult:
    """A/B/A trust check: candidate medians from blocks 1 and 3 must agree.

    If they don't, the machine state shifted during the measurement window —
    so the reference block sandwiched between them was measured under
    inconsistent conditions and the speedup ratio is untrustworthy. The
    direction of the shift suggests the likely cause:
      - block 3 slower → thermal degradation; let machine cool, plug in power
      - block 3 faster → GPU was cold/idle in block 1; bump warmups, ensure
        consistent system load before measurement
    """
    if block1.median_ms <= 0:
        return StabilityResult(False, float("inf"),
                               "block 1 median was zero — candidate timing failed")
    delta = abs(block1.median_ms - block3.median_ms) / block1.median_ms
    if delta <= threshold:
        return StabilityResult(True, delta, "ok")
    direction = "faster" if block3.median_ms < block1.median_ms else "slower"
    if direction == "faster":
        hint = ("GPU likely cold/idle during block 1 and reached high-perf state "
                "by block 3; bump warmup count or ensure consistent system load "
                "before measurement")
    else:
        hint = ("machine likely thermally throttling, on battery, or under "
                "competing GPU load; let it cool, plug in power, retry")
    return StabilityResult(
        False, delta,
        f"timing instability: candidate ran {delta:.1%} {direction} in block 3 "
        f"({block3.median_ms:.4f}ms) vs block 1 ({block1.median_ms:.4f}ms), "
        f"threshold {threshold:.0%} — speedup untrustworthy; {hint}"
    )


def summarize(samples_ms: list[float]) -> TimingStats:
    med = statistics.median(samples_ms)
    if len(samples_ms) >= 4:
        q = statistics.quantiles(samples_ms, n=4)
        iqr = q[2] - q[0]
    else:
        iqr = max(samples_ms) - min(samples_ms)
    return TimingStats(med, iqr, med > 0 and (iqr / med) > NOISY_IQR_FRAC, samples_ms)


def time_reference_mps(reference_fn, inputs: dict, warmup: int = 3, runs: int = 10) -> TimingStats:
    """Time the PyTorch reference on the MPS backend (wall clock + synchronize).

    Imported lazily so the rest of mkb works without torch installed.
    """
    import torch

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    t_inputs = {k: torch.from_numpy(v).to(dev) for k, v in inputs.items()}

    def sync():
        if dev.type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        reference_fn(**t_inputs)
        sync()

    samples = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        reference_fn(**t_inputs)
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return summarize(samples)


def _current_env_metadata() -> dict[str, str]:
    """Snapshot of fields that must match between a recorded baseline and the
    machine state at check time. If any drift, the baseline is stale, not just
    thermally suspect — so we surface that distinction in error messages.
    """
    try:
        soc = _subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip() or "unknown"
    except Exception:
        soc = "unknown"
    try:
        import torch  # lazy: don't force torch onto pure-Python users
        torch_v = torch.__version__
    except ImportError:
        torch_v = "absent"
    return {
        "macos": _platform.mac_ver()[0] or "unknown",
        "device": soc,
        "python": f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
        "torch": torch_v,
    }


def record_calibration(kernel_id: str, median_ms: float) -> None:
    data: dict = {}
    if CALIBRATION_FILE.exists():
        existing = json.loads(CALIBRATION_FILE.read_text())
        # Preserve sibling entries only when their schema matches; otherwise the
        # whole file is from a prior format and we start fresh.
        if existing.get("schema_version") == CALIBRATION_SCHEMA_VERSION:
            data = existing
    data["schema_version"] = CALIBRATION_SCHEMA_VERSION
    entry = _current_env_metadata()
    entry["median_ms"] = float(median_ms)
    entry["recorded_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    data[kernel_id] = entry
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(data, indent=2))


def check_calibration(kernel_id: str, median_ms: float) -> str | None:
    """Returns a warning string if the recorded baseline is stale (env shifted)
    or the candidate has drifted from it. None means baseline is valid AND the
    candidate is within DRIFT_WARN_FRAC of it.
    """
    if not CALIBRATION_FILE.exists():
        return None
    data = json.loads(CALIBRATION_FILE.read_text())
    if data.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        return (f"calibration baseline schema v{data.get('schema_version')!r} does not "
                f"match expected v{CALIBRATION_SCHEMA_VERSION} — please re-record "
                f"with --calibrate.")
    entry = data.get(kernel_id)
    if not isinstance(entry, dict):
        return None  # no baseline for this kernel yet — not an error
    base = entry.get("median_ms")
    if not isinstance(base, (int, float)) or base <= 0:
        return None
    current = _current_env_metadata()
    mismatches = [
        f"{f}: {entry.get(f)!r} -> {current[f]!r}"
        for f in _CRITICAL_ENV_FIELDS
        if entry.get(f) != current[f]
    ]
    if mismatches:
        return ("calibration baseline is stale (env changed): "
                + "; ".join(mismatches) + " — please re-record with --calibrate.")
    drift = abs(median_ms - base) / base
    if drift > DRIFT_WARN_FRAC:
        return (f"calibration drift {drift:.0%} on '{kernel_id}' "
                f"(baseline {base:.3f}ms, now {median_ms:.3f}ms) — machine may be hot, "
                f"on battery, or under load. Results from this session are suspect.")
    return None
