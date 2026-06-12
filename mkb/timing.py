"""Timing utilities — the trust layer for benchmark numbers.

Laptop thermals are the #1 validity threat. Policy:
- warmup runs before any measurement
- median-of-N with IQR reported; runs with IQR > 15% of median are flagged noisy
- a calibration check that re-times a golden kernel and warns on >10% drift
- speedup is a ratio of medians measured in the same session, so thermal drift
  hits candidate and reference roughly equally
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

CALIBRATION_FILE = Path(__file__).resolve().parents[1] / "results" / "calibration.json"
NOISY_IQR_FRAC = 0.15
DRIFT_WARN_FRAC = 0.10


@dataclass
class TimingStats:
    median_ms: float
    iqr_ms: float
    noisy: bool
    samples: list[float]


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


def record_calibration(kernel_id: str, median_ms: float) -> None:
    data = {}
    if CALIBRATION_FILE.exists():
        data = json.loads(CALIBRATION_FILE.read_text())
    data[kernel_id] = median_ms
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(data, indent=2))


def check_calibration(kernel_id: str, median_ms: float) -> str | None:
    """Returns a warning string if this run drifted from the recorded baseline."""
    if not CALIBRATION_FILE.exists():
        return None
    data = json.loads(CALIBRATION_FILE.read_text())
    base = data.get(kernel_id)
    if base is None or base <= 0:
        return None
    drift = abs(median_ms - base) / base
    if drift > DRIFT_WARN_FRAC:
        return (f"calibration drift {drift:.0%} on '{kernel_id}' "
                f"(baseline {base:.3f}ms, now {median_ms:.3f}ms) — machine may be hot, "
                f"on battery, or under load. Results from this session are suspect.")
    return None
