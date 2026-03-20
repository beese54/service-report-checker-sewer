"""
overview_analytics.py — Unified Pipeline Analytics Report
==========================================================
Produces a single dark-theme PNG dashboard covering all 4 pipeline stages:
  Stage 1  — OCR Obstruction Analysis
  Stage 2N — SIFT D1/D4 + U1/U4
  Stage 3N — Washing Confidence D2/D5 + U2/U5
  Stage 4N — Geometry-First D3/D6 + U3/U6

Usage:
    python overview_analytics.py

Output:
    adjusted_images/no_obstruction/difference_analysis/overview_analytics_<YYYYMMDD_HHMMSS>.png
"""

import datetime
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\adjusted_images"
NO_OBS_DIR   = os.path.join(BASE_DIR, "no_obstruction")
ANALYSIS_DIR = os.path.join(NO_OBS_DIR, "difference_analysis")

OCR_RESULTS_PATH  = os.path.join(BASE_DIR, "obstruction", "extracted_text", "results.json")
S2N_PATTERN       = os.path.join(ANALYSIS_DIR, "run_report_*.json")
S3N_PATH          = os.path.join(ANALYSIS_DIR, "D2D5U2U5_check", "level2_check.json")
S4N_PATTERN       = os.path.join(ANALYSIS_DIR, "gf_run_*.json")

# ── Stage 4N thresholds (single source of truth in pipeline_config / config.json)
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from pipeline_config import (
    BLUR_REJECT_THRESHOLD,
    SSIM_MIN_THRESHOLD,
    INLIER_RATIO_MIN,
    GREASE_FLAG_THRESHOLD,
    ENTROPY_CONFIRM_DELTA,
    WATER_DETECT_THRESHOLD,
    HOMOGRAPHY_DET_MIN,
    HOMOGRAPHY_DET_MAX,
)

# ── Colour palette ─────────────────────────────────────────────────────────
COL_PASS   = "#4CAF50"
COL_FAIL   = "#E53935"
COL_REVIEW = "#FF9800"
COL_INFO   = "#1E88E5"
COL_NA     = "#9E9E9E"
COL_BG     = "#1A1A2E"
COL_AX     = "#16213E"
COL_TEXT   = "#E0E0E0"
COL_GRID   = "#2A2A4A"
COL_THRESH = "#FF9800"
COL_HIGH   = "#4CAF50"
COL_MED    = "#FF9800"
COL_LOW    = "#E53935"

# ── Section colours ────────────────────────────────────────────────────────
SECTION_COLOURS = {
    "S1":  "#7B1FA2",
    "S2N": "#1565C0",
    "S3N": "#00695C",
    "S4N": "#E65100",
    "REF": "#37474F",
}


# =============================================================================
# Helpers
# =============================================================================

def _load_json(path):
    """Load a JSON file; return None on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_latest(pattern):
    """Glob pattern, pick latest by mtime, return parsed JSON or None."""
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not candidates:
        return None, None
    path = candidates[-1]
    data = _load_json(path)
    return data, path


def _style_ax(ax, title, xlabel="", ylabel="Count"):
    ax.set_facecolor(COL_AX)
    ax.set_title(title, color=COL_TEXT, fontsize=9, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=COL_TEXT, fontsize=7.5)
    ax.set_ylabel(ylabel, color=COL_TEXT, fontsize=7.5)
    ax.tick_params(colors=COL_TEXT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_GRID)
    ax.yaxis.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def _no_data(ax, title):
    ax.set_facecolor(COL_AX)
    ax.set_title(title, color=COL_TEXT, fontsize=9, fontweight="bold", pad=6)
    ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
            ha="center", va="center", color=COL_NA, fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_GRID)


def _section_divider(fig, gs_row, label, colour):
    """Return axis used as a coloured section header bar."""
    ax = fig.add_subplot(gs_row)
    ax.set_facecolor(colour)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.5, label, transform=ax.transAxes,
            ha="left", va="center", color="white",
            fontsize=11, fontweight="bold")
    return ax


def _get_gf(result, *keys, default=None):
    """Walk gf result: sift_stats -> diag -> direct -> gate."""
    for top in ("sift_stats", "diag"):
        d = result.get(top) or {}
        for k in keys:
            if k in d:
                return d[k]
    for k in keys:
        if k in result:
            return result[k]
    gate = result.get("gate") or {}
    for k in keys:
        if k in gate:
            return gate[k]
    return default


def _annotate_bar(ax, bars, values, fmt="{}", color=None):
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    fmt.format(val),
                    ha="center", va="bottom",
                    color=color or COL_TEXT, fontsize=7.5)


def _pct(n, total):
    return f"{n/total:.0%}" if total else "0%"


# =============================================================================
# Data loading
# =============================================================================

def load_all_data():
    ocr_data    = _load_json(OCR_RESULTS_PATH)
    s2n_data, s2n_path = _load_latest(S2N_PATTERN)
    s3n_data    = _load_json(S3N_PATH)
    s4n_data, s4n_path = _load_latest(S4N_PATTERN)
    return {
        "ocr":  (ocr_data,  OCR_RESULTS_PATH if ocr_data else None),
        "s2n":  (s2n_data,  s2n_path),
        "s3n":  (s3n_data,  S3N_PATH if s3n_data else None),
        "s4n":  (s4n_data,  s4n_path),
    }


def _ts_from_path(path):
    if not path:
        return "N/A"
    base = os.path.basename(path)
    # try to pull a timestamp substring like 20260304_214712
    for part in base.replace(".json", "").split("_"):
        if len(part) == 8 and part.isdigit():
            # next part might be time
            idx = base.index(part)
            rest = base[idx:idx + 15].replace("_", " ")
            return rest
    return os.path.basename(path)


# =============================================================================
# Panel builders
# =============================================================================

# ── Stage 1 ──────────────────────────────────────────────────────────────────

def panel_ocr_summary(ax, ocr):
    if not ocr:
        _no_data(ax, "S1 — Folders & Images Processed")
        return
    n_folders = len(ocr)
    total_images = 0
    total_failed = 0
    mh_found = 0
    for folder_data in ocr.values():
        for img_data in folder_data.values():
            if isinstance(img_data, dict):
                total_images += 1
                if img_data.get("manhole_number"):
                    mh_found += 1
                if not img_data.get("raw_lines"):
                    total_failed += 1
    labels  = ["Folders", "Images", "No lines", "MH found"]
    values  = [n_folders, total_images, total_failed, mh_found]
    colours = [COL_INFO, COL_PASS, COL_FAIL, COL_REVIEW]
    bars = ax.bar(labels, values, color=colours, alpha=0.85, zorder=3)
    _annotate_bar(ax, bars, values)
    _style_ax(ax, "S1 — Folders & Images Processed")


def panel_ocr_conf_hist(ax, ocr):
    if not ocr:
        _no_data(ax, "S1 — OCR Confidence Distribution")
        return
    confs = []
    for folder_data in ocr.values():
        for img_data in folder_data.values():
            if isinstance(img_data, dict):
                for line in img_data.get("raw_lines", []):
                    c = line.get("confidence")
                    if c is not None:
                        confs.append(float(c))
    if not confs:
        _no_data(ax, "S1 — OCR Confidence Distribution")
        return
    bins = np.linspace(0, 1, 41)
    high   = [c for c in confs if c >= 0.80]
    medium = [c for c in confs if 0.50 <= c < 0.80]
    low    = [c for c in confs if c < 0.50]
    ax.hist(high,   bins=bins, color=COL_PASS,   alpha=0.85, label="HIGH (>=0.80)",   zorder=3)
    ax.hist(medium, bins=bins, color=COL_REVIEW,  alpha=0.85, label="MED (0.50-0.79)", zorder=3)
    ax.hist(low,    bins=bins, color=COL_FAIL,    alpha=0.85, label="LOW (<0.50)",     zorder=3)
    ax.axvline(0.50, color=COL_THRESH, linewidth=1.5, linestyle="--", zorder=4)
    ax.axvline(0.80, color=COL_THRESH, linewidth=1.5, linestyle="-.", zorder=4)
    _style_ax(ax, f"S1 — OCR Confidence  (n={len(confs)})", "Confidence")
    ax.legend(fontsize=7, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)


def panel_ocr_mh_rate(ax, ocr):
    if not ocr:
        _no_data(ax, "S1 — MH Number Extraction Rate")
        return
    mh_found = 0
    mh_missing = 0
    for folder_data in ocr.values():
        for img_data in folder_data.values():
            if isinstance(img_data, dict):
                if img_data.get("manhole_number"):
                    mh_found += 1
                else:
                    mh_missing += 1
    total = mh_found + mh_missing
    labels  = ["MH found", "MH missing"]
    values  = [mh_found, mh_missing]
    colours = [COL_PASS, COL_FAIL]
    bars = ax.bar(labels, values, color=colours, alpha=0.85, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val} ({_pct(val, total)})",
                ha="center", va="bottom", color=COL_TEXT, fontsize=8)
    _style_ax(ax, f"S1 — MH Extraction Rate  (n={total})")


# ── Stage 2N ─────────────────────────────────────────────────────────────────

def panel_s2n_summary(ax, s2n):
    if not s2n:
        _no_data(ax, "S2N — Detection Summary")
        return
    summary = s2n.get("summary", {})
    labels  = ["Auto\nAccepted", "For\nReview", "Auto\nRejected", "Skipped\nErrors"]
    keys    = ["auto_accepted", "sent_for_review", "auto_rejected", "skipped_errors"]
    colours = [COL_PASS, COL_REVIEW, COL_FAIL, COL_NA]
    values  = [summary.get(k, 0) for k in keys]
    bars = ax.bar(labels, values, color=colours, alpha=0.85, zorder=3)
    _annotate_bar(ax, bars, values)
    total = sum(values)
    _style_ax(ax, f"S2N — SIFT Detection Summary  (n={total})")


def panel_s2n_gate_pie(ax, s2n):
    if not s2n:
        _no_data(ax, "S2N — Detection Gate")
        return
    summary = s2n.get("summary", {})
    accepted = summary.get("auto_accepted", 0)
    review   = summary.get("sent_for_review", 0)
    rejected = summary.get("auto_rejected", 0)
    values = [accepted, review, rejected]
    labels = ["ACCEPTED\n(>=0.70)", "REVIEW\n(0.30-0.69)", "REJECTED\n(<0.30)"]
    colours = [COL_PASS, COL_REVIEW, COL_FAIL]
    non_zero = [(v, l, c) for v, l, c in zip(values, labels, colours) if v > 0]
    if not non_zero:
        _no_data(ax, "S2N — Detection Gate")
        return
    v, l, c = zip(*non_zero)
    ax.pie(v, labels=l, colors=c, autopct="%1.0f%%", startangle=90,
           textprops={"color": COL_TEXT, "fontsize": 7},
           wedgeprops={"alpha": 0.85})
    gate_active = s2n.get("gate_active", False)
    gate_str = "Gate: ACTIVE" if gate_active else "Gate: INACTIVE (no model)"
    ax.set_title(f"S2N — Detection Gate\n{gate_str}",
                 color=COL_TEXT, fontsize=9, fontweight="bold", pad=6)
    ax.set_facecolor(COL_AX)
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_GRID)


def panel_s2n_thresholds(ax, s2n):
    ax.set_facecolor(COL_AX)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_GRID)
    conf_thresh = {}
    if s2n:
        conf_thresh = s2n.get("conf_thresholds", {})
    lines = [
        "S2N Threshold Reference",
        "",
        "RATIO_THRESHOLD  = 0.75",
        "RANSAC_THRESHOLD = 5.0 px",
        "MIN_INLIERS      = 10",
        "MIN_COVERAGE_PCT = 25%",
        "",
        f"CONF_ACCEPT  = {conf_thresh.get('accept', 0.70):.2f}",
        f"CONF_REVIEW  = {conf_thresh.get('review', 0.30):.2f}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            ha="left", va="top", color=COL_TEXT, fontsize=8,
            fontfamily="monospace",
            bbox=dict(facecolor=COL_BG, edgecolor=COL_GRID, alpha=0.8, boxstyle="round"))
    ax.set_title("S2N — Threshold Reference", color=COL_TEXT, fontsize=9, fontweight="bold", pad=6)


# ── Stage 3N ─────────────────────────────────────────────────────────────────

def panel_s3n_tier_bar(ax, s3n):
    if not s3n:
        _no_data(ax, "S3N — Folder Tier Breakdown")
        return
    summary = s3n.get("summary", {})
    labels  = ["HIGH", "MEDIUM", "LOW", "FAILED"]
    colours = [COL_PASS, COL_REVIEW, COL_FAIL, COL_NA]
    values  = [summary.get(k, 0) for k in labels]
    bars = ax.bar(labels, values, color=colours, alpha=0.85, zorder=3)
    total = sum(values)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val}\n({_pct(val, total)})",
                    ha="center", va="bottom", color=COL_TEXT, fontsize=7.5)
    pass_tier = s3n.get("criteria", {}).get("pass_tier", "HIGH")
    _style_ax(ax, f"S3N — Folder Tier  (PASS_TIER={pass_tier})")


def panel_s3n_conf_hist(ax, s3n):
    if not s3n:
        _no_data(ax, "S3N — Washing Confidence")
        return
    confs = []
    for folder_data in s3n.get("folders", {}).values():
        for pair_data in folder_data.get("pairs", {}).values():
            c = pair_data.get("washing_confidence")
            if c is not None:
                confs.append(float(c))
    if not confs:
        _no_data(ax, "S3N — Washing Confidence")
        return
    bins  = np.linspace(0, 1, 41)
    high  = [c for c in confs if c >= 0.50]
    med   = [c for c in confs if 0.30 <= c < 0.50]
    low   = [c for c in confs if c < 0.30]
    ax.hist(high, bins=bins, color=COL_PASS,   alpha=0.85, label="HIGH (>=0.50)", zorder=3)
    ax.hist(med,  bins=bins, color=COL_REVIEW,  alpha=0.85, label="MED (0.30-0.49)", zorder=3)
    ax.hist(low,  bins=bins, color=COL_FAIL,    alpha=0.85, label="LOW (<0.30)",  zorder=3)
    ax.axvspan(0.30, 0.50, color=COL_REVIEW, alpha=0.08, zorder=2)
    ax.axvline(0.30, color=COL_THRESH, linewidth=1.5, linestyle="--", zorder=4)
    ax.axvline(0.50, color=COL_THRESH, linewidth=1.5, linestyle="-.", zorder=4)
    _style_ax(ax, f"S3N — Washing Confidence  (n={len(confs)})", "Confidence")
    ax.legend(fontsize=7, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)


def panel_s3n_entropy_hist(ax, s3n):
    if not s3n:
        _no_data(ax, "S3N — Entropy Increase")
        return
    vals = []
    for folder_data in s3n.get("folders", {}).values():
        for pair_data in folder_data.get("pairs", {}).values():
            v = pair_data.get("entropy_increase")
            if v is not None:
                vals.append(float(v))
    if not vals:
        _no_data(ax, "S3N — Entropy Increase")
        return
    lo = min(min(vals) - 0.05, -0.1)
    hi = max(max(vals) + 0.05, 0.5)
    bins = np.linspace(lo, hi, 40)
    passed = [v for v in vals if v >= ENTROPY_CONFIRM_DELTA]
    failed = [v for v in vals if v < ENTROPY_CONFIRM_DELTA]
    ax.hist(passed, bins=bins, color=COL_PASS, alpha=0.85, label="PASS", zorder=3)
    ax.hist(failed, bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL", zorder=3)
    ax.axvline(ENTROPY_CONFIRM_DELTA, color=COL_THRESH, linewidth=1.5,
               linestyle="--", zorder=4, label=f"Threshold={ENTROPY_CONFIRM_DELTA}")
    _style_ax(ax, f"S3N — Entropy Increase  (n={len(vals)})", "nats")
    ax.legend(fontsize=7, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)


def panel_s3n_signal_rates(ax, s3n):
    if not s3n:
        _no_data(ax, "S3N — Signal Contribution Rates")
        return
    signals = {
        "kp_ratio":       0,
        "std_increase":   0,
        "entropy_increase": 0,
        "match_ratio":    0,
        "edge_increase":  0,
        "lap_increase":   0,
    }
    signal_map = {
        "kp_ratio":         lambda p: p.get("kp_ratio", 0) > 1.0,
        "std_increase":     lambda p: p.get("std_increase_pct", 0) > 0,
        "entropy_increase": lambda p: p.get("entropy_increase", 0) >= ENTROPY_CONFIRM_DELTA,
        "match_ratio":      lambda p: p.get("match_ratio", 0) > 0,
        "edge_increase":    lambda p: p.get("edge_increase_pct", 0) > 0,
        "lap_increase":     lambda p: p.get("lap_increase_pct", 0) > 0,
    }
    counts = {k: 0 for k in signals}
    total  = 0
    for folder_data in s3n.get("folders", {}).values():
        for pair_data in folder_data.get("pairs", {}).values():
            if pair_data.get("status") != "OK":
                continue
            total += 1
            for sig, fn in signal_map.items():
                if fn(pair_data):
                    counts[sig] += 1
    if total == 0:
        _no_data(ax, "S3N — Signal Contribution Rates")
        return
    pcts = [counts[k] / total * 100 for k in signals]
    colours = [COL_PASS if p > 50 else COL_REVIEW if p >= 30 else COL_FAIL for p in pcts]
    labels = ["kp_ratio", "std_incr", "entropy\nincr", "match\nratio", "edge\nincr", "lap\nincr"]
    bars = ax.barh(labels, pcts, color=colours, alpha=0.85, zorder=3)
    ax.axvline(50, color=COL_THRESH, linewidth=1.2, linestyle="--", zorder=4)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%", ha="left", va="center", color=COL_TEXT, fontsize=7)
    _style_ax(ax, f"S3N — Signal Contribution Rates  (n={total})", "% pairs signal positive", "")
    ax.set_xlim(0, 110)


# ── Stage 4N ─────────────────────────────────────────────────────────────────

def panel_s4n_status_bar(ax, s4n):
    if not s4n:
        _no_data(ax, "S4N — Pair Status Breakdown")
        return
    counts = s4n.get("counts", {})
    status_order  = ["PASS", "REVIEW", "FAIL", "GATE_REJECTED", "ALIGNMENT_FAILED", "MISSING_IMAGES"]
    status_colors = [COL_PASS, COL_REVIEW, COL_FAIL, "#9C27B0", COL_INFO, COL_NA]
    values = [counts.get(k, 0) for k in status_order]
    short_labels = ["PASS", "REVIEW", "FAIL", "GATE\nREJ", "ALIGN\nFAIL", "MISSING"]
    bars = ax.bar(short_labels, values, color=status_colors, alpha=0.85, zorder=3)
    _annotate_bar(ax, bars, values)
    total = sum(values)
    _style_ax(ax, f"S4N — Pair Status  (total={total})")
    ax.tick_params(axis="x", labelsize=7)


def panel_s4n_score_dist(ax, s4n):
    if not s4n:
        _no_data(ax, "S4N — Score Distribution (0-5)")
        return
    scores = [r.get("score") for r in s4n.get("results", []) if r.get("score") is not None]
    if not scores:
        _no_data(ax, "S4N — Score Distribution (0-5)")
        return
    bins = np.arange(-0.5, 6.5, 1)
    passed = [s for s in scores if s >= 3]
    review = [s for s in scores if s == 2]
    failed = [s for s in scores if s < 2]
    ax.hist(passed, bins=bins, color=COL_PASS,   alpha=0.85, label="PASS (>=3)", zorder=3)
    ax.hist(review, bins=bins, color=COL_REVIEW,  alpha=0.85, label="REVIEW (2)", zorder=3)
    ax.hist(failed, bins=bins, color=COL_FAIL,    alpha=0.85, label="FAIL (<2)",  zorder=3)
    ax.axvline(2, color=COL_THRESH, linewidth=1.5, linestyle="--", zorder=4)
    ax.axvline(3, color=COL_THRESH, linewidth=1.5, linestyle="-.", zorder=4)
    ax.set_xticks(range(6))
    _style_ax(ax, f"S4N — Score Distribution  (n={len(scores)})", "Score (0-5)")
    ax.legend(fontsize=7, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)


def panel_s4n_phase_c_signals(ax, s4n):
    if not s4n:
        _no_data(ax, "S4N — Phase C Signal Rates")
        return
    results = s4n.get("results", [])
    processable = [r for r in results
                   if r.get("status") not in ("MISSING_IMAGES", "GATE_REJECTED", "ALIGNMENT_FAILED")]
    n = len(processable)
    if n == 0:
        _no_data(ax, "S4N — Phase C Signal Rates")
        return

    def _rate(fn):
        return sum(1 for r in processable if fn(r)) / n * 100

    signals = {
        "S1: Inliers >= 10":      _rate(lambda r: (_get_gf(r, "ransac_inliers") or 0) >= 10),
        "S2: pct_changed >= 5%":  _rate(lambda r: (_get_gf(r, "pct_changed") or 0) >= 5),
        "S3: Entropy confirmed":  _rate(lambda r: bool((r.get("texture") or {}).get("confirmed"))),
        "S4: Water detected":     _rate(lambda r: bool((r.get("water") or {}).get("water_detected"))),
        "S5: Grease NOT flagged": _rate(lambda r: not bool((r.get("grease") or {}).get("flagged"))),
    }
    labels = list(signals.keys())
    pcts   = list(signals.values())
    colours = [COL_PASS if p >= 50 else COL_REVIEW if p >= 30 else COL_FAIL for p in pcts]
    bars = ax.barh(labels, pcts, color=colours, alpha=0.85, zorder=3)
    ax.axvline(50, color=COL_THRESH, linewidth=1.2, linestyle="--", zorder=4)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%", ha="left", va="center", color=COL_TEXT, fontsize=7)
    _style_ax(ax, f"S4N — Phase C Signal Rates  (n={n})", "% pairs signal fired", "")
    ax.set_xlim(0, 120)
    ax.tick_params(axis="y", labelsize=7)


def panel_s4n_water_conf(ax, s4n):
    if not s4n:
        _no_data(ax, "S4N — Water Confidence Distribution")
        return
    vals = []
    for r in s4n.get("results", []):
        w = r.get("water") or {}
        wc = w.get("water_confidence")
        if wc is not None:
            vals.append(float(wc))
    if not vals:
        _no_data(ax, "S4N — Water Confidence Distribution")
        return
    bins = np.linspace(0, 1, 41)
    detected = [v for v in vals if v >= 0.1]
    not_det  = [v for v in vals if v < 0.1]
    ax.hist(detected, bins=bins, color=COL_INFO, alpha=0.85, label="Detected",     zorder=3)
    ax.hist(not_det,  bins=bins, color=COL_NA,   alpha=0.85, label="Not detected", zorder=3)
    ax.axvline(0.1, color=COL_THRESH, linewidth=1.5, linestyle="--", zorder=4,
               label="~3% combined")
    _style_ax(ax, f"S4N — Water Confidence  (n={len(vals)})", "Confidence")
    ax.legend(fontsize=7, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)


def _hist_pass_fail(ax, vals, threshold, above_passes, title, xlabel, fmt_thresh=""):
    """Generic pass/fail histogram with threshold line."""
    if not vals:
        _no_data(ax, title)
        return
    lo = min(vals)
    hi = max(vals)
    bins = np.linspace(max(0, lo - abs(lo) * 0.05), hi + abs(hi) * 0.05, 40)
    if above_passes:
        passed = [v for v in vals if v >= threshold]
        failed = [v for v in vals if v < threshold]
    else:
        passed = [v for v in vals if v < threshold]
        failed = [v for v in vals if v >= threshold]
    ax.hist(passed, bins=bins, color=COL_PASS, alpha=0.85, label="PASS", zorder=3)
    ax.hist(failed, bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL", zorder=3)
    thresh_label = fmt_thresh or f"Threshold={threshold}"
    ax.axvline(threshold, color=COL_THRESH, linewidth=1.5, linestyle="--",
               zorder=4, label=thresh_label)
    _style_ax(ax, f"{title}  (n={len(vals)})", xlabel)
    ax.legend(fontsize=7, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    n = len(vals)
    ax.text(0.98, 0.95,
            f"PASS {len(passed)} ({_pct(len(passed), n)})\nFAIL {len(failed)} ({_pct(len(failed), n)})",
            transform=ax.transAxes, ha="right", va="top", color=COL_TEXT, fontsize=7,
            bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.8))


def panel_s4n_blur(ax, s4n):
    if not s4n:
        _no_data(ax, "S4N — Blur Score")
        return
    vals = [_get_gf(r, "blur_score") for r in s4n.get("results", [])
            if _get_gf(r, "blur_score") is not None]
    _hist_pass_fail(ax, vals, BLUR_REJECT_THRESHOLD, above_passes=True,
                    title="S4N — Blur Score", xlabel="Laplacian Variance",
                    fmt_thresh=f"Threshold={BLUR_REJECT_THRESHOLD}")


def panel_s4n_ssim(ax, s4n):
    if not s4n:
        _no_data(ax, "S4N — SSIM Score")
        return
    vals = [_get_gf(r, "ssim_score") for r in s4n.get("results", [])
            if _get_gf(r, "ssim_score") is not None]
    _hist_pass_fail(ax, vals, SSIM_MIN_THRESHOLD, above_passes=True,
                    title="S4N — SSIM Score", xlabel="SSIM",
                    fmt_thresh=f"Threshold={SSIM_MIN_THRESHOLD}")


def panel_s4n_inlier(ax, s4n):
    if not s4n:
        _no_data(ax, "S4N — Inlier Ratio")
        return
    vals = [_get_gf(r, "inlier_ratio") for r in s4n.get("results", [])
            if _get_gf(r, "inlier_ratio") is not None]
    vals_pct = [v * 100 for v in vals]
    _hist_pass_fail(ax, vals_pct, INLIER_RATIO_MIN * 100, above_passes=True,
                    title="S4N — Inlier Ratio", xlabel="Inlier Ratio (%)",
                    fmt_thresh=f"Threshold={INLIER_RATIO_MIN:.0%}")


def panel_s4n_grease(ax, s4n):
    if not s4n:
        _no_data(ax, "S4N — Grease %")
        return
    vals = []
    for r in s4n.get("results", []):
        g = r.get("grease") or {}
        gp = g.get("grease_pct")
        if gp is not None:
            vals.append(float(gp))
    _hist_pass_fail(ax, vals, GREASE_FLAG_THRESHOLD, above_passes=False,
                    title="S4N — Grease %", xlabel="Grease Percentage",
                    fmt_thresh=f"Flag threshold={GREASE_FLAG_THRESHOLD}%")


# ── Funnel ────────────────────────────────────────────────────────────────────

def panel_funnel(ax, data):
    ocr_data   = data["ocr"][0]
    s2n_data   = data["s2n"][0]
    s3n_data   = data["s3n"][0]
    s4n_data   = data["s4n"][0]

    # Derive counts
    total_no_obs = 0
    s2n_pass  = 0
    s3n_high  = 0
    s4n_acc   = 0

    if s4n_data:
        # Use folders_processed as total (no-obstruction)
        total_no_obs = s4n_data.get("folders_processed", 0)
        c = s4n_data.get("counts", {})
        s4n_acc = c.get("PASS", 0) + c.get("REVIEW", 0)

    if s2n_data:
        s2n_pass = s2n_data.get("summary", {}).get("auto_accepted", 0)

    if s3n_data:
        s3n_high = s3n_data.get("summary", {}).get("HIGH", 0)

    ocr_folders = len(ocr_data) if ocr_data else 0

    total_all = total_no_obs + ocr_folders

    stages = [
        ("Total Folders", total_all,    "#455A64"),
        ("No-obstruction", total_no_obs, "#1565C0"),
        ("S2N PASS",      s2n_pass,     "#1976D2"),
        ("S3N HIGH",      s3n_high,     "#00897B"),
        ("S4N Accepted",  s4n_acc,      "#43A047"),
    ]

    ax.set_facecolor(COL_AX)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_GRID)

    n = len(stages)
    box_w = 0.14
    box_h = 0.55
    gap = (1.0 - n * box_w) / (n + 1)

    for i, (label, count, colour) in enumerate(stages):
        x = gap + i * (box_w + gap)
        y = (1 - box_h) / 2

        rect = mpatches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.01",
            facecolor=colour, edgecolor="white", linewidth=1.2, alpha=0.9,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)

        ax.text(x + box_w / 2, y + box_h * 0.65, str(count),
                transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=14, fontweight="bold")

        if total_all > 0:
            pct = count / total_all * 100
            ax.text(x + box_w / 2, y + box_h * 0.40, f"{pct:.0f}%",
                    transform=ax.transAxes, ha="center", va="center",
                    color="white", fontsize=9)

        ax.text(x + box_w / 2, y - 0.06, label,
                transform=ax.transAxes, ha="center", va="top",
                color=COL_TEXT, fontsize=8, fontweight="bold")

        # arrow to next box
        if i < n - 1:
            arrow_x = x + box_w + 0.005
            arrow_xend = x + box_w + gap - 0.005
            mid_y = y + box_h / 2
            ax.annotate("",
                xy=(arrow_xend, mid_y), xytext=(arrow_x, mid_y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=COL_THRESH, lw=1.5))

    ax.set_title("Pipeline Cascade Funnel", color=COL_TEXT,
                 fontsize=11, fontweight="bold", pad=6)


# ── Threshold table ───────────────────────────────────────────────────────────

THRESHOLD_TABLE = [
    # Stage, Parameter, Value, Meaning
    ("Stage 1 OCR", "conf HIGH",          ">= 0.80",         "High-confidence OCR line"),
    ("Stage 1 OCR", "conf MEDIUM",        "0.50 - 0.79",     "Acceptable OCR confidence"),
    ("Stage 1 OCR", "conf LOW",           "< 0.50",          "Low OCR confidence"),
    ("Stage 2N",    "RATIO_THRESHOLD",    "0.75",            "Lowe's ratio test"),
    ("Stage 2N",    "RANSAC_THRESHOLD",   "5.0 px",          "RANSAC tolerance"),
    ("Stage 2N",    "MIN_INLIERS",        "10",              "Minimum inliers to pass"),
    ("Stage 2N",    "MIN_COVERAGE_PCT",   "25.0 %",          "4x4 grid zone coverage"),
    ("Stage 2N",    "CONF_ACCEPT",        "0.70",            "YOLOv8 auto-accept"),
    ("Stage 2N",    "CONF_REVIEW",        "0.30",            "YOLOv8 queue for review"),
    ("Stage 3N",    "PASS_TIER",          "HIGH",            "Only HIGH tier passes"),
    ("Stage 3N",    "HIGH confidence",    ">= 0.50",         "Strong washing evidence"),
    ("Stage 3N",    "MEDIUM confidence",  "0.30 - 0.49",     "Moderate washing evidence"),
    ("Stage 3N",    "LOW confidence",     "< 0.30",          "Minimal washing evidence"),
    ("Stage 4N",    "BLUR_REJECT_THRESHOLD", f"{BLUR_REJECT_THRESHOLD}", "Laplacian variance gate"),
    ("Stage 4N",    "SSIM_MIN_THRESHOLD", f"{SSIM_MIN_THRESHOLD}",  "Global scene similarity floor"),
    ("Stage 4N",    "INLIER_RATIO_MIN",   f"{INLIER_RATIO_MIN:.0%}", "RANSAC inlier ratio floor"),
    ("Stage 4N",    "GREASE_FLAG_THRESHOLD", f"{GREASE_FLAG_THRESHOLD} %", "Grease presence flag"),
    ("Stage 4N",    "ENTROPY_CONFIRM_DELTA", f"{ENTROPY_CONFIRM_DELTA} nats", "Texture change confirmation"),
    ("Stage 4N",    "WATER_DETECT_THRESHOLD", f"{WATER_DETECT_THRESHOLD} %", "Water signature detection"),
    ("Stage 4N",    "HOMOGRAPHY_DET",     f"{HOMOGRAPHY_DET_MIN} - {HOMOGRAPHY_DET_MAX}", "Informational only (no gate)"),
    ("Stage 4N",    "PASS score",         ">= 3 / 5",        "At least 3 of 5 signals"),
    ("Stage 4N",    "REVIEW score",       "2 / 5",           "Borderline"),
    ("Stage 4N",    "FAIL score",         "< 2 / 5",         "Insufficient evidence"),
]


def panel_threshold_table(ax):
    ax.set_facecolor(COL_AX)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_GRID)

    col_headers = ["Stage", "Parameter", "Value", "Meaning"]
    cell_data   = [[r[0], r[1], r[2], r[3]] for r in THRESHOLD_TABLE]

    # Row colours: alternate + highlight by stage
    stage_colours = {
        "Stage 1 OCR": "#2D1B4E",
        "Stage 2N":    "#0D2B5E",
        "Stage 3N":    "#0B3B35",
        "Stage 4N":    "#3B1F0A",
    }
    row_colours = [[stage_colours.get(r[0], COL_AX)] * 4 for r in THRESHOLD_TABLE]

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_headers,
        cellLoc="left",
        loc="center",
        cellColours=row_colours,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)

    # Style header
    for col in range(4):
        cell = tbl[0, col]
        cell.set_facecolor("#263238")
        cell.set_text_props(color="white", fontweight="bold")

    # Style data cells
    for row in range(1, len(THRESHOLD_TABLE) + 1):
        for col in range(4):
            cell = tbl[row, col]
            cell.set_text_props(color=COL_TEXT)
            cell.set_edgecolor(COL_GRID)

    ax.set_title("Threshold Reference — All Stages", color=COL_TEXT,
                 fontsize=10, fontweight="bold", pad=6)


# =============================================================================
# Main figure assembly
# =============================================================================

def build_figure(data):
    ocr_d  = data["ocr"][0]
    s2n_d  = data["s2n"][0]
    s3n_d  = data["s3n"][0]
    s4n_d  = data["s4n"][0]

    fig = plt.figure(figsize=(24, 34), facecolor=COL_BG)

    # Height ratios: [title, funnel, div, s1, div, s2n, div, s3n, div, s4n-row9, s4n-row10, div, table]
    height_ratios = [0.5, 2.5, 0.25, 2.5, 0.25, 2.5, 0.25, 2.5, 0.25, 2.5, 2.5, 0.25, 3.0]
    gs = gridspec.GridSpec(13, 4, figure=fig,
                           height_ratios=height_ratios,
                           hspace=0.55, wspace=0.35,
                           top=0.97, bottom=0.02,
                           left=0.04, right=0.97)

    # ── Row 0: Title bar ──────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor("#263238")
    ax_title.axis("off")
    for spine in ax_title.spines.values():
        spine.set_edgecolor(COL_GRID)

    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s2n_ts = _ts_from_path(data["s2n"][1])
    s3n_ts = _ts_from_path(data["s3n"][1])
    s4n_ts = _ts_from_path(data["s4n"][1])

    ax_title.text(0.01, 0.75, "WRN Service Report Checker — Pipeline Analytics Overview",
                  transform=ax_title.transAxes, ha="left", va="center",
                  color="white", fontsize=14, fontweight="bold")
    ax_title.text(0.01, 0.25, f"Generated: {run_ts}",
                  transform=ax_title.transAxes, ha="left", va="center",
                  color=COL_NA, fontsize=9)
    ax_title.text(0.99, 0.75,
                  f"S2N run: {s2n_ts}  |  S3N: {s3n_ts}  |  S4N: {s4n_ts}",
                  transform=ax_title.transAxes, ha="right", va="center",
                  color=COL_NA, fontsize=8)

    # ── Row 1: Funnel ─────────────────────────────────────────────────────
    ax_funnel = fig.add_subplot(gs[1, :])
    panel_funnel(ax_funnel, data)

    # ── Row 2: S1 section divider ─────────────────────────────────────────
    _section_divider(fig, gs[2, :], "  STAGE 1 — OCR Obstruction Analysis", SECTION_COLOURS["S1"])

    # ── Row 3: S1 panels ──────────────────────────────────────────────────
    panel_ocr_summary   (fig.add_subplot(gs[3, 0]), ocr_d)
    panel_ocr_conf_hist (fig.add_subplot(gs[3, 1]), ocr_d)
    panel_ocr_mh_rate   (fig.add_subplot(gs[3, 2]), ocr_d)
    ax_blank1 = fig.add_subplot(gs[3, 3])
    ax_blank1.set_facecolor(COL_AX)
    ax_blank1.axis("off")

    # ── Row 4: S2N section divider ────────────────────────────────────────
    _section_divider(fig, gs[4, :], "  STAGE 2N — SIFT D1/D4 + U1/U4", SECTION_COLOURS["S2N"])

    # ── Row 5: S2N panels ─────────────────────────────────────────────────
    panel_s2n_summary   (fig.add_subplot(gs[5, 0]), s2n_d)
    panel_s2n_gate_pie  (fig.add_subplot(gs[5, 1]), s2n_d)
    panel_s2n_thresholds(fig.add_subplot(gs[5, 2]), s2n_d)
    ax_blank2 = fig.add_subplot(gs[5, 3])
    ax_blank2.set_facecolor(COL_AX)
    ax_blank2.axis("off")

    # ── Row 6: S3N section divider ────────────────────────────────────────
    _section_divider(fig, gs[6, :], "  STAGE 3N — Washing Confidence D2/D5 + U2/U5", SECTION_COLOURS["S3N"])

    # ── Row 7: S3N panels ─────────────────────────────────────────────────
    panel_s3n_tier_bar    (fig.add_subplot(gs[7, 0]), s3n_d)
    panel_s3n_conf_hist   (fig.add_subplot(gs[7, 1]), s3n_d)
    panel_s3n_entropy_hist(fig.add_subplot(gs[7, 2]), s3n_d)
    panel_s3n_signal_rates(fig.add_subplot(gs[7, 3]), s3n_d)

    # ── Row 8: S4N section divider ────────────────────────────────────────
    _section_divider(fig, gs[8, :], "  STAGE 4N — Geometry-First D3/D6 + U3/U6", SECTION_COLOURS["S4N"])

    # ── Row 9: S4N status + scoring ───────────────────────────────────────
    panel_s4n_status_bar     (fig.add_subplot(gs[9, 0]), s4n_d)
    panel_s4n_score_dist     (fig.add_subplot(gs[9, 1]), s4n_d)
    panel_s4n_phase_c_signals(fig.add_subplot(gs[9, 2]), s4n_d)
    panel_s4n_water_conf     (fig.add_subplot(gs[9, 3]), s4n_d)

    # ── Row 10: S4N deep metrics ──────────────────────────────────────────
    panel_s4n_blur  (fig.add_subplot(gs[10, 0]), s4n_d)
    panel_s4n_ssim  (fig.add_subplot(gs[10, 1]), s4n_d)
    panel_s4n_inlier(fig.add_subplot(gs[10, 2]), s4n_d)
    panel_s4n_grease(fig.add_subplot(gs[10, 3]), s4n_d)

    # ── Row 11: Threshold section divider ─────────────────────────────────
    _section_divider(fig, gs[11, :], "  THRESHOLD REFERENCE — All Stages", SECTION_COLOURS["REF"])

    # ── Row 12: Threshold table ───────────────────────────────────────────
    panel_threshold_table(fig.add_subplot(gs[12, :]))

    return fig


# =============================================================================
# Entry point
# =============================================================================

def main():
    print("Loading data sources...")
    data = load_all_data()

    for stage, (d, path) in data.items():
        status = f"OK: {os.path.basename(path)}" if d else "MISSING"
        print(f"  {stage:5s}: {status}")

    print("Building figure...")
    fig = build_figure(data)

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(ANALYSIS_DIR, f"overview_analytics_{ts}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=COL_BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
