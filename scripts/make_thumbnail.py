#!/usr/bin/env python3
"""Generate a 16:10 portfolio thumbnail for MSL-Bench.

Outputs demo/thumbnail.svg (vector master) and demo/thumbnail.png (rasterized
via rsvg-convert). Data-driven — regenerating pulls current metrics from
results/raw/ so the thumbnail can't drift stale.

Layout: brand header on top; hero title in one line; three data panels below
(fast_p curves full-width; tier bars + failure-mode stack side by side).
Each LLM run gets a distinct color from a curated editorial palette.

Run:
    make thumbnail
    # or:
    .venv/bin/python scripts/make_thumbnail.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkb import problems as P
from mkb.score import fast_p

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
DEMO = ROOT / "demo"

W, H = 1600, 1000

# Portfolio palette (verbatim from erika-goh.github.io/portfolio/styles.css)
BG = "#050505"
TEXT = "#ededed"
TEXT_2 = "#a1a1a1"
TEXT_3 = "#6b6b6b"
ACCENT = "#407914"
LINE = "rgba(255,255,255,0.07)"
GRID_LINE = "rgba(255,255,255,0.045)"
FONT_BODY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, monospace"

# Curated editorial dark palette — one distinct color per run, all mid-luminance
# so nothing screams against the near-black background. Portfolio green
# (#407914) reserved for the leader.
CURVE_PALETTE = [
    ACCENT,      # #407914  forest green (leader)
    "#6b8fa5",   # steel blue
    "#c8a367",   # warm sand
    "#906188",   # plum
    "#b3735c",   # terracotta
    "#5c9c8a",   # sea foam
    "#c2b280",   # khaki
    "#8a6d7d",   # mauve
    "#90a373",   # sage green
]

STAGE_ORDER = ["correct", "verify", "compile", "runtime", "error", "none"]
STAGE_COLORS = {
    "correct": ACCENT,
    "verify":  "rgba(237,237,237,0.72)",
    "compile": "rgba(237,237,237,0.44)",
    "runtime": "rgba(237,237,237,0.32)",
    "error":   "rgba(237,237,237,0.22)",
    "none":    "rgba(237,237,237,0.12)",
}


# ────────────────────────── data ──────────────────────────

def _run_label(run_id: str) -> str:
    rest = run_id
    mt = ""
    for _mt in ("_mt5200", "_mt16384"):
        if rest.endswith(_mt):
            mt = f" · mt={_mt[3:]}"
            rest = rest[: -len(_mt)]
            break
    if rest.endswith("_one_shot"):
        mode = "one_shot"
        rest = rest[: -len("_one_shot")]
    elif rest.endswith("_repair"):
        mode = "repair"
        rest = rest[: -len("_repair")]
    else:
        mode = ""
    provider, _, model = rest.partition("_")
    model = model.replace("-versatile", "").replace("-instant", "")
    return f"{provider} · {model}" + (f" · {mode}" if mode else "") + mt


def collect_runs() -> tuple[dict[str, list[dict]], int]:
    runs: dict[str, list[dict]] = {}
    for f in sorted(RAW.glob("*.json")):
        rec = json.loads(f.read_text())
        runs.setdefault(rec["run"], []).append(rec)
    total_problems = len(list(P.discover()))
    return runs, total_problems


def build_curves(runs: dict[str, list[dict]], total_problems: int) -> list[dict]:
    curves = []
    for run_id, recs in runs.items():
        speedups = [r.get("speedup") or 0 for r in recs if r.get("correct")]
        n = len(recs)
        pts = [(0.0, sum(1 for s in speedups if s > 0) / max(n, 1))]
        p = 0.05
        while p <= 3.001:
            passing = sum(1 for s in speedups if s >= p)
            pts.append((p, passing / max(n, 1)))
            p += 0.05
        curves.append({
            "id": run_id,
            "label": _run_label(run_id),
            "recs": recs,
            "n": n,
            "partial": n < total_problems,
            "fast0": fast_p(recs, 0),
            "fast1": fast_p(recs, 1),
            "fast2": fast_p(recs, 2),
            "pts": pts,
        })
    # Sort leader-first by fast_1 (tiebreaker: n desc so full-suite beats partial)
    curves.sort(key=lambda c: (-c["fast1"], -c["n"]))
    return curves


def full_suite_leader(curves: list[dict]) -> dict | None:
    """Best fast_1 among full-suite runs. Falls back to overall leader."""
    full = [c for c in curves if not c["partial"]]
    if not full:
        return curves[0] if curves else None
    return max(full, key=lambda c: c["fast1"])


# ────────────────────────── chart panels ──────────────────────────

def curve_color(i: int) -> str:
    return CURVE_PALETTE[i % len(CURVE_PALETTE)]


def _chart_frame(x, y, w, h, title):
    """Panel background + title label. Returns the SVG string."""
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="rgba(15,15,15,0.30)" stroke="{LINE}" stroke-width="1" rx="6"/>',
        f'<text x="{x+22}" y="{y+30}" font-family="{FONT_MONO}" font-size="11" '
        f'fill="{TEXT_3}" letter-spacing="0.16em">{title}</text>',
    ]
    return "\n".join(parts)


def render_fastp_chart(x0, y0, w, h, curves):
    padL, padR, padT, padB = 56, 28, 62, 46
    inner_w = w - padL - padR
    inner_h = h - padT - padB
    xToPx = lambda x: x0 + padL + (x / 3.0) * inner_w
    yToPx = lambda y: y0 + padT + (1 - y) * inner_h

    parts = [_chart_frame(x0, y0, w, h, "FAST_P CURVES · % OF PROBLEMS CORRECT AND ≥ p× MPS")]

    # Grid + y labels
    for y_frac in (0, 0.25, 0.5, 0.75, 1.0):
        yp = yToPx(y_frac)
        parts.append(f'<line x1="{xToPx(0):.1f}" y1="{yp:.1f}" x2="{xToPx(3):.1f}" '
                     f'y2="{yp:.1f}" stroke="{GRID_LINE}" stroke-width="1"/>')
        parts.append(f'<text x="{xToPx(0)-10:.1f}" y="{yp+4:.1f}" text-anchor="end" '
                     f'font-family="{FONT_MONO}" font-size="10" fill="{TEXT_3}">'
                     f'{int(y_frac*100)}%</text>')
    # X labels
    for xv in (0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        xp = xToPx(xv)
        parts.append(f'<text x="{xp:.1f}" y="{y0+h-padB+18:.1f}" text-anchor="middle" '
                     f'font-family="{FONT_MONO}" font-size="10" fill="{TEXT_3}">{xv:.1f}×</text>')

    # p=1 reference dashed line
    xp1 = xToPx(1)
    parts.append(f'<line x1="{xp1:.1f}" y1="{y0+padT}" x2="{xp1:.1f}" y2="{y0+h-padB}" '
                 f'stroke="rgba(255,255,255,0.18)" stroke-width="1" stroke-dasharray="2 4"/>')
    parts.append(f'<text x="{xp1+6:.1f}" y="{y0+padT+12}" font-family="{FONT_MONO}" '
                 f'font-size="10" fill="{TEXT_3}">= MPS</text>')

    # X axis title
    parts.append(f'<text x="{x0+w/2:.1f}" y="{y0+h-10}" text-anchor="middle" '
                 f'font-family="{FONT_MONO}" font-size="10" fill="{TEXT_3}" '
                 f'letter-spacing="0.06em">SPEEDUP THRESHOLD p (× MPS)</text>')

    # Draw curves back to front so the leader lands on top
    for i in reversed(range(len(curves))):
        c = curves[i]
        color = curve_color(i)
        width = 1.75 if i == 0 else 1.25
        d = " ".join(
            f'{"M" if j == 0 else "L"}{xToPx(x):.1f},{yToPx(y):.1f}'
            for j, (x, y) in enumerate(c["pts"])
        )
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        x0v, y0v = c["pts"][0]
        parts.append(f'<circle cx="{xToPx(x0v):.1f}" cy="{yToPx(y0v):.1f}" '
                     f'r="{"3" if i == 0 else "2"}" fill="{color}"/>')
    return "\n".join(parts)


def render_tier_chart(x0, y0, w, h, curves):
    padL, padR, padT, padB = 42, 18, 58, 40
    inner_w = w - padL - padR
    inner_h = h - padT - padB
    tiers = [1, 2, 3, 4]
    group_w = inner_w / len(tiers)
    n_runs = len(curves)
    inner_pad = 8
    bar_gap = 2
    bar_w = max(3.5, (group_w - inner_pad * 2 - (n_runs - 1) * bar_gap) / n_runs)
    yToPx = lambda y: y0 + padT + (1 - y) * inner_h

    parts = [_chart_frame(x0, y0, w, h, "BY TIER · fast_0")]

    # Horizontal grid
    for y_frac in (0, 0.25, 0.5, 0.75, 1.0):
        yp = yToPx(y_frac)
        parts.append(f'<line x1="{x0+padL:.1f}" y1="{yp:.1f}" x2="{x0+w-padR:.1f}" '
                     f'y2="{yp:.1f}" stroke="{GRID_LINE}" stroke-width="1"/>')
        parts.append(f'<text x="{x0+padL-8:.1f}" y="{yp+4:.1f}" text-anchor="end" '
                     f'font-family="{FONT_MONO}" font-size="9" fill="{TEXT_3}">'
                     f'{int(y_frac*100)}%</text>')

    for ti, t in enumerate(tiers):
        gx = x0 + padL + ti * group_w + inner_pad
        for ri in range(n_runs):
            recs = [r for r in curves[ri]["recs"] if r["tier"] == t]
            attempted = len(recs)
            correct = sum(1 for r in recs if r.get("correct"))
            pct = 0 if attempted == 0 else correct / attempted
            color = curve_color(ri)
            bx = gx + ri * (bar_w + bar_gap)
            if attempted == 0:
                parts.append(f'<line x1="{bx:.1f}" y1="{y0+h-padB-0.5:.1f}" '
                             f'x2="{bx+bar_w:.1f}" y2="{y0+h-padB-0.5:.1f}" '
                             f'stroke="{TEXT_3}" stroke-width="1" stroke-dasharray="2 3" '
                             f'opacity="0.5"/>')
            elif pct == 0:
                parts.append(f'<rect x="{bx:.1f}" y="{y0+h-padB-1.5:.1f}" '
                             f'width="{bar_w:.1f}" height="1.5" fill="{color}" opacity="0.6"/>')
            else:
                yp = yToPx(pct)
                bh = (y0 + h - padB) - yp
                parts.append(f'<rect x="{bx:.1f}" y="{yp:.1f}" width="{bar_w:.1f}" '
                             f'height="{max(bh,1):.1f}" fill="{color}"/>')
        # Tier label
        cx = gx + (n_runs * bar_w + (n_runs - 1) * bar_gap) / 2
        parts.append(f'<text x="{cx:.1f}" y="{y0+h-padB+20:.1f}" text-anchor="middle" '
                     f'font-family="{FONT_MONO}" font-size="10" fill="{TEXT_3}" '
                     f'letter-spacing="0.06em">T{t}</text>')
    return "\n".join(parts)


def render_stage_chart(x0, y0, w, h, curves):
    padT = 58
    label_w = 240   # room for run label at left
    bar_h = 16
    bar_gap = 10
    padR = 20

    inner_w = w - label_w - padR
    n_runs = len(curves)

    # Compute counts
    per_run = []
    for c in curves:
        counts = {k: 0 for k in STAGE_ORDER}
        for r in c["recs"]:
            if r.get("correct"):
                counts["correct"] += 1
            else:
                stage = r.get("fail_stage") or "error"
                mapped = {"compile": "compile", "verify": "verify", "runtime": "runtime",
                          "no_code": "none", "provider_error": "error"}.get(stage, "error")
                counts[mapped] += 1
        per_run.append({"label": c["label"], "counts": counts, "n": c["n"]})

    max_total = max(sum(r["counts"].values()) for r in per_run) or 1
    x_scale = inner_w / max_total

    parts = [_chart_frame(x0, y0, w, h, "FAILURE MODES · correct + failure stages")]

    for ri, r in enumerate(per_run):
        y = y0 + padT + ri * (bar_h + bar_gap)
        # Truncate long labels; only draw if row fits inside the panel
        if y + bar_h > y0 + h - 12:
            break
        label = r["label"]
        if len(label) > 30:
            label = label[:29] + "…"
        parts.append(f'<text x="{x0+label_w-10:.1f}" y="{y+bar_h/2+3.5:.1f}" '
                     f'text-anchor="end" font-family="{FONT_MONO}" font-size="9.5" '
                     f'fill="{TEXT_3}" letter-spacing="0.02em">{label}</text>')
        bx = x0 + label_w
        for stage in STAGE_ORDER:
            count = r["counts"][stage]
            if count == 0:
                continue
            bw = count * x_scale
            # Correct segment: this run's distinct color (matches the fast_p
            # curve and tier bars). Failure segments: grayscale.
            fill_color = curve_color(ri) if stage == "correct" else STAGE_COLORS[stage]
            parts.append(f'<rect x="{bx:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                         f'height="{bar_h}" fill="{fill_color}"/>')
            if bw >= 18:
                text_fill = "#050505" if stage == "correct" else "rgba(5,5,5,0.85)"
                parts.append(f'<text x="{bx+bw/2:.1f}" y="{y+bar_h/2+3:.1f}" '
                             f'text-anchor="middle" font-family="{FONT_MONO}" '
                             f'font-size="9" fill="{text_fill}" font-weight="500">'
                             f'{count}</text>')
            bx += bw
    return "\n".join(parts)


# ────────────────────────── assembly ──────────────────────────

def build_svg(curves: list[dict], total_problems: int) -> str:
    leader = full_suite_leader(curves) or curves[0]
    n_runs = len(curves)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
                 f'width="{W}" height="{H}" preserveAspectRatio="xMidYMid meet">')
    parts.append('<defs>')
    parts.append('  <radialGradient id="topglow" cx="50%" cy="0%" r="65%" fx="50%" fy="0%">')
    parts.append(f'    <stop offset="0%" stop-color="{ACCENT}" stop-opacity="0.12"/>')
    parts.append(f'    <stop offset="60%" stop-color="{ACCENT}" stop-opacity="0"/>')
    parts.append('  </radialGradient>')
    parts.append('</defs>')

    # Background + top-glow
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{BG}"/>')
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="url(#topglow)"/>')

    # ── Brand header (top) ──
    parts.append(f'<circle cx="86" cy="72" r="7" fill="{ACCENT}"/>')
    parts.append(f'<circle cx="86" cy="72" r="13" fill="none" stroke="{ACCENT}" '
                 f'stroke-opacity="0.35"/>')
    parts.append(f'<text x="112" y="79" font-family="{FONT_BODY}" font-size="19" '
                 f'font-weight="500" fill="{TEXT}" letter-spacing="-0.01em">MSL-Bench</text>')
    parts.append(f'<text x="{W-72}" y="79" text-anchor="end" font-family="{FONT_MONO}" '
                 f'font-size="12" fill="{TEXT_3}" letter-spacing="0.14em">'
                 f'{total_problems} PROBLEMS · 4 TIERS · {n_runs} RUNS</text>')

    # ── Section label with 24px accent lead line ──
    lb_y = 148
    parts.append(f'<line x1="72" y1="{lb_y}" x2="96" y2="{lb_y}" stroke="{ACCENT}" '
                 f'stroke-width="1"/>')
    parts.append(f'<text x="112" y="{lb_y+4}" font-family="{FONT_MONO}" font-size="12" '
                 f'fill="{ACCENT}" letter-spacing="0.24em">01 .KERNELBENCH()</text>')

    # ── Hero title ── two-line, sits above the chart cluster
    parts.append(f'<text x="72" y="210" font-family="{FONT_BODY}" font-size="46" '
                 f'font-weight="500" fill="{TEXT}" letter-spacing="-0.025em">'
                 f'Evaluating LLM-Generated Metal Compute Kernels</text>')
    parts.append(f'<text x="72" y="256" font-family="{FONT_BODY}" font-size="46" '
                 f'font-weight="500" fill="{TEXT_2}" letter-spacing="-0.025em">'
                 f'on Apple Silicon</text>')

    # ── Charts: fast_p full-width, then tier + stage side by side ──
    margin = 72
    chart_top = 300
    fastp_h = 360
    row2_h = 250

    parts.append(render_fastp_chart(margin, chart_top, W - 2 * margin, fastp_h, curves))

    gap = 24
    row2_y = chart_top + fastp_h + 24
    half_w = (W - 2 * margin - gap) / 2
    parts.append(render_tier_chart(margin, row2_y, half_w, row2_h, curves))
    parts.append(render_stage_chart(margin + half_w + gap, row2_y, half_w, row2_h, curves))

    # ── Footer strip ──
    foot_y = H - 34
    partial_flag = "†" if leader["partial"] else ""
    stat_text = (f'BEST FAST_1 · {leader["fast1"]*100:.1f}%  ·  {leader["label"]}  '
                 f'·  n={leader["n"]}{partial_flag}')
    parts.append(f'<text x="72" y="{foot_y}" font-family="{FONT_MONO}" font-size="12" '
                 f'fill="{TEXT_2}" letter-spacing="0.06em">{stat_text}</text>')
    parts.append(f'<text x="{W-72}" y="{foot_y}" text-anchor="end" font-family="{FONT_MONO}" '
                 f'font-size="12" fill="{TEXT_3}" letter-spacing="0.14em">MSL-BENCH v0.1</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def rasterize(svg_path: Path, png_path: Path, width: int = W) -> bool:
    try:
        subprocess.run(
            ["rsvg-convert", "-w", str(width), "-o", str(png_path), str(svg_path)],
            check=True, capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  (rsvg-convert failed: {e}); SVG only", file=sys.stderr)
        return False


def main() -> None:
    runs, total_problems = collect_runs()
    if not runs:
        raise SystemExit("no records in results/raw/; nothing to render")

    curves = build_curves(runs, total_problems)
    svg = build_svg(curves, total_problems)

    DEMO.mkdir(exist_ok=True)
    svg_path = DEMO / "thumbnail.svg"
    svg_path.write_text(svg)
    print(f"wrote {svg_path.relative_to(ROOT)}  ({len(svg)/1024:.1f} KB)")

    png_path = DEMO / "thumbnail.png"
    if rasterize(svg_path, png_path):
        print(f"wrote {png_path.relative_to(ROOT)}  ({png_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
