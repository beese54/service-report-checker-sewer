"""
GF Pipeline Histogram Generator
=================================
Reads the most recent gf_run_*.json from the difference_analysis folder
and produces an 8-panel histogram image showing the distribution of every
diagnostic metric with pass/fail thresholds annotated.

Usage:
    python plot_gf_histograms.py [path/to/gf_run_YYYYMMDD_HHMMSS.json]

If no path is given the most-recent gf_run_*.json in the default location
is used automatically.

Output:
    <analysis_dir>/gf_metric_histograms_<timestamp>.png
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on Windows)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\adjusted_images\no_obstruction"
ANALYSIS_DIR = os.path.join(BASE_DIR, "difference_analysis")

# ── Thresholds (mirror d3d6_u3u6.py) ─────────────────────────────────────
BLUR_REJECT_THRESHOLD = 80.0
SSIM_MIN_THRESHOLD    = 0.05
HOMOGRAPHY_DET_MIN    = 0.2
HOMOGRAPHY_DET_MAX    = 5.0
INLIER_RATIO_MIN      = 0.30

# ── Colours ────────────────────────────────────────────────────────────────
COL_PASS    = "#4CAF50"
COL_FAIL    = "#E53935"
COL_THRESH  = "#FF9800"
COL_INFO    = "#1E88E5"
COL_NA      = "#9E9E9E"
COL_BG      = "#1A1A2E"
COL_AX      = "#16213E"
COL_TEXT    = "#E0E0E0"
COL_GRID    = "#2A2A4A"


# =============================================================================
# Data extraction helpers
# =============================================================================

def _get(result, *keys, default=None):
    """Walk nested dicts, return first found value across multiple top-level keys."""
    # try sift_stats first (successful pairs), then diag (failed pairs)
    for top in ("sift_stats", "diag"):
        d = result.get(top) or {}
        for k in keys:
            if k in d:
                return d[k]
    # fallback: direct on result (gate lives here)
    for k in keys:
        if k in result:
            return result[k]
    gate = result.get("gate") or {}
    for k in keys:
        if k in gate:
            return gate[k]
    return default


def extract_metrics(results):
    """Return a dict of metric_name -> list of (value, passed_bool_or_None) tuples."""
    blur_scores   = []
    circle_data   = []   # list of "found+centred" / "found" / "not found"
    ssim_scores   = []
    masked_total  = []
    kp_ratio      = []   # ratio_matches count
    h_det         = []
    inlier_ratios = []

    for r in results:
        gate = r.get("gate") or {}

        # ── Phase A: blur
        bs = gate.get("blur_score")
        if bs is not None:
            blur_scores.append((bs, bs >= BLUR_REJECT_THRESHOLD))

        # ── Phase A: circle
        if gate:
            if gate.get("circle_found") and gate.get("centering_ok"):
                circle_data.append("found + centred")
            elif gate.get("circle_found"):
                circle_data.append("found (off-axis)")
            elif r.get("status") not in ("MISSING_IMAGES",):
                circle_data.append("not found")

        # ── Phase A.5: SSIM
        sv = _get(r, "ssim_score")
        if sv is not None:
            ssim_scores.append((sv, sv >= SSIM_MIN_THRESHOLD))

        # ── Phase B: timestamps masked (total)
        nb = _get(r, "text_regions_masked_before") or 0
        na = _get(r, "text_regions_masked_after")  or 0
        if gate:   # only record if we got past loading
            masked_total.append(nb + na)

        # ── Phase B: SIFT ratio_matches
        rm = _get(r, "ratio_matches")
        if rm is not None:
            kp_ratio.append(rm)

        # ── Phase B: homography determinant
        det = _get(r, "homography_det")
        if det is not None:
            h_det.append((det, HOMOGRAPHY_DET_MIN <= det <= HOMOGRAPHY_DET_MAX))

        # ── Phase B: inlier ratio
        ir = _get(r, "inlier_ratio")
        if ir is not None:
            inlier_ratios.append((ir, ir >= INLIER_RATIO_MIN))

    return {
        "blur_scores":   blur_scores,
        "circle_data":   circle_data,
        "ssim_scores":   ssim_scores,
        "masked_total":  masked_total,
        "kp_ratio":      kp_ratio,
        "h_det":         h_det,
        "inlier_ratios": inlier_ratios,
    }


# =============================================================================
# Individual panel helpers
# =============================================================================

def _style_ax(ax, title, xlabel, ylabel="Count"):
    ax.set_facecolor(COL_AX)
    ax.set_title(title, color=COL_TEXT, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=COL_TEXT, fontsize=9)
    ax.set_ylabel(ylabel, color=COL_TEXT, fontsize=9)
    ax.tick_params(colors=COL_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_GRID)
    ax.yaxis.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def _pass_fail_legend(ax, pass_label="PASS", fail_label="FAIL"):
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=COL_PASS, label=pass_label),
            mpatches.Patch(facecolor=COL_FAIL, label=fail_label),
        ],
        fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT,
    )


def plot_blur(ax, data):
    if not data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Phase A: Blur Score", "Laplacian Variance")
        return

    vals   = [v for v, _ in data]
    passed = [v for v, p in data if p]
    failed = [v for v, p in data if not p]
    bins   = np.linspace(min(vals) - 5, max(vals) + 5, 40)

    ax.hist(passed, bins=bins, color=COL_PASS, alpha=0.85, label="PASS", zorder=3)
    ax.hist(failed, bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL", zorder=3)
    ax.axvline(BLUR_REJECT_THRESHOLD, color=COL_THRESH, linewidth=1.8,
               linestyle="--", zorder=4, label=f"Threshold={BLUR_REJECT_THRESHOLD}")
    _style_ax(ax, f"Phase A: Blur Score  (n={len(data)})", "Laplacian Variance")
    _pass_fail_legend(ax)
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)

    n_pass = len(passed)
    n_fail = len(failed)
    ax.text(0.98, 0.95, f"PASS {n_pass} ({n_pass/len(data):.0%})\nFAIL {n_fail} ({n_fail/len(data):.0%})",
            transform=ax.transAxes, ha="right", va="top", color=COL_TEXT,
            fontsize=8, bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.8))


def plot_circle(ax, data):
    if not data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Phase A: Pipe Circle Detection", "Category", ylabel="Count")
        return

    cats   = ["found + centred", "found (off-axis)", "not found"]
    colors = [COL_PASS, COL_INFO, COL_FAIL]
    counts = [data.count(c) for c in cats]
    bars   = ax.bar(cats, counts, color=colors, alpha=0.85, zorder=3)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(count), ha="center", va="bottom", color=COL_TEXT, fontsize=9)

    _style_ax(ax, f"Phase A: Pipe Circle Detection  (n={len(data)})",
              "Detection Result")
    ax.tick_params(axis="x", labelsize=8)
    total = len(data)
    note = f"advisory only — not a hard reject\nn={total}"
    ax.text(0.98, 0.95, note, transform=ax.transAxes, ha="right", va="top",
            color=COL_TEXT, fontsize=8,
            bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.8))


def plot_ssim(ax, data):
    if not data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Phase A.5: SSIM Score", "SSIM")
        return

    vals   = [v for v, _ in data]
    passed = [v for v, p in data if p]
    failed = [v for v, p in data if not p]
    lo, hi = min(vals) - 0.01, max(vals) + 0.01
    bins   = np.linspace(max(0, lo), min(1, hi), 40)

    ax.hist(passed, bins=bins, color=COL_PASS, alpha=0.85, label="PASS", zorder=3)
    ax.hist(failed, bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL", zorder=3)
    ax.axvline(SSIM_MIN_THRESHOLD, color=COL_THRESH, linewidth=1.8,
               linestyle="--", zorder=4, label=f"Threshold={SSIM_MIN_THRESHOLD}")
    _style_ax(ax, f"Phase A.5: SSIM Score  (n={len(data)})",
              "Structural Similarity Index")

    n_pass = len(passed)
    n_fail = len(failed)
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    ax.text(0.98, 0.95, f"PASS {n_pass} ({n_pass/len(data):.0%})\nFAIL {n_fail} ({n_fail/len(data):.0%})",
            transform=ax.transAxes, ha="right", va="top", color=COL_TEXT,
            fontsize=8, bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.8))


def plot_masked(ax, data):
    if not data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Phase B: Timestamps Masked", "Total Masked Regions")
        return

    bins = np.arange(-0.5, max(data) + 1.5, 1)
    none_detected = [v for v in data if v == 0]
    some_detected = [v for v in data if v > 0]
    ax.hist(none_detected, bins=bins, color=COL_INFO, alpha=0.85,
            label="none detected", zorder=3)
    ax.hist(some_detected, bins=bins, color=COL_PASS, alpha=0.85,
            label="applied", zorder=3)
    _style_ax(ax, f"Phase B: Timestamps Masked  (n={len(data)})",
              "Total Masked Regions (before + after)")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)

    n_some = len(some_detected)
    n_none = len(none_detected)
    ax.text(0.98, 0.95,
            f"Masking applied: {n_some} ({n_some/len(data):.0%})\nNone detected: {n_none} ({n_none/len(data):.0%})",
            transform=ax.transAxes, ha="right", va="top", color=COL_TEXT,
            fontsize=8, bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.8))


def plot_sift_kp(ax, data):
    if not data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Phase B: SIFT Ratio Matches", "Count")
        return

    bins = np.linspace(0, max(data) + 10, 40)
    ax.hist(data, bins=bins, color=COL_INFO, alpha=0.85, zorder=3)
    _style_ax(ax, f"Phase B: SIFT Ratio Matches  (n={len(data)})",
              "Ratio-test filtered match count")
    median_val = float(np.median(data))
    ax.axvline(median_val, color=COL_THRESH, linewidth=1.5,
               linestyle="--", zorder=4, label=f"Median={median_val:.0f}")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    ax.text(0.98, 0.95,
            f"Median: {median_val:.0f}\nMin: {min(data)}  Max: {max(data)}",
            transform=ax.transAxes, ha="right", va="top", color=COL_TEXT,
            fontsize=8, bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.8))


def plot_hdet(ax, data):
    if not data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Phase B: Homography Determinant", "Determinant")
        return

    vals   = [v for v, _ in data]
    passed = [v for v, p in data if p]
    failed = [v for v, p in data if not p]

    # clip for display (extreme outliers compress the useful range)
    clip_lo, clip_hi = -2.0, 8.0
    vals_c   = [max(clip_lo, min(clip_hi, v)) for v in vals]
    passed_c = [max(clip_lo, min(clip_hi, v)) for v, p in data if p]
    failed_c = [max(clip_lo, min(clip_hi, v)) for v, p in data if not p]
    bins = np.linspace(clip_lo, clip_hi, 45)

    ax.hist(passed_c, bins=bins, color=COL_PASS, alpha=0.85, label="PASS", zorder=3)
    ax.hist(failed_c, bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL", zorder=3)

    ax.axvline(HOMOGRAPHY_DET_MIN, color=COL_THRESH, linewidth=1.8,
               linestyle="--", zorder=4, label=f"Min={HOMOGRAPHY_DET_MIN}")
    ax.axvline(HOMOGRAPHY_DET_MAX, color=COL_THRESH, linewidth=1.8,
               linestyle="-.", zorder=4, label=f"Max={HOMOGRAPHY_DET_MAX}")
    # shade valid range
    ax.axvspan(HOMOGRAPHY_DET_MIN, min(HOMOGRAPHY_DET_MAX, clip_hi),
               color=COL_PASS, alpha=0.07, zorder=2)

    _style_ax(ax, f"Phase B: Homography Determinant  (n={len(data)})",
              f"det(H[:2,:2])  [clipped to {clip_lo},{clip_hi}]")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)

    n_pass = len(passed)
    n_fail = len(failed)
    ax.text(0.98, 0.95,
            f"PASS {n_pass} ({n_pass/len(data):.0%})\nFAIL {n_fail} ({n_fail/len(data):.0%})",
            transform=ax.transAxes, ha="right", va="top", color=COL_TEXT,
            fontsize=8, bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.8))


def plot_inlier(ax, data):
    if not data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Phase B: Inlier Ratio", "Ratio")
        return

    vals   = [v for v, _ in data]
    passed = [v for v, p in data if p]
    failed = [v for v, p in data if not p]
    bins   = np.linspace(0, 1.0, 40)

    ax.hist(passed, bins=bins, color=COL_PASS, alpha=0.85, label="PASS", zorder=3)
    ax.hist(failed, bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL", zorder=3)
    ax.axvline(INLIER_RATIO_MIN, color=COL_THRESH, linewidth=1.8,
               linestyle="--", zorder=4, label=f"Threshold={INLIER_RATIO_MIN:.0%}")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    _style_ax(ax, f"Phase B: Inlier Ratio  (n={len(data)})",
              "Inliers / Ratio-test Matches")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)

    n_pass = len(passed)
    n_fail = len(failed)
    ax.text(0.98, 0.95,
            f"PASS {n_pass} ({n_pass/len(data):.0%})\nFAIL {n_fail} ({n_fail/len(data):.0%})",
            transform=ax.transAxes, ha="right", va="top", color=COL_TEXT,
            fontsize=8, bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.8))


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        pattern   = os.path.join(ANALYSIS_DIR, "gf_run_*.json")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            print(f"No gf_run_*.json found in {ANALYSIS_DIR}")
            sys.exit(1)
        json_path = candidates[-1]

    print(f"Reading: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        run_data = json.load(f)

    results = run_data.get("results", [])
    print(f"  {len(results)} pair results loaded.")

    counts = run_data.get("counts", {})
    total  = run_data.get("folders_processed", "?")
    ts     = run_data.get("timestamp", "")

    metrics = extract_metrics(results)

    # ── Figure layout: 4 rows x 2 cols ──────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.patch.set_facecolor(COL_BG)

    # Title
    status_summary = "  |  ".join(
        f"{k}: {v}" for k, v in counts.items() if v > 0
    )
    fig.suptitle(
        f"D3/D6 + U3/U6 Geometry-First Pipeline — Metric Distributions\n"
        f"Run: {ts}   Folders: {total}   Pairs: {len(results)}\n"
        f"{status_summary}",
        color=COL_TEXT, fontsize=13, fontweight="bold", y=0.995,
    )

    plot_blur   (axes[0][0], metrics["blur_scores"])
    plot_circle (axes[0][1], metrics["circle_data"])
    plot_ssim   (axes[1][0], metrics["ssim_scores"])
    plot_masked (axes[1][1], metrics["masked_total"])
    plot_sift_kp(axes[2][0], metrics["kp_ratio"])

    # Row 2, col 1 — pair status breakdown (bar chart)
    ax_status = axes[2][1]
    status_order  = ["PASS", "REVIEW", "FAIL", "GATE_REJECTED", "ALIGNMENT_FAILED", "MISSING_IMAGES"]
    status_colors = [COL_PASS, "#FFC107", COL_FAIL, "#9C27B0", COL_INFO, COL_NA]
    s_counts = [counts.get(k, 0) for k in status_order]
    bars = ax_status.bar(status_order, s_counts, color=status_colors, alpha=0.85, zorder=3)
    for bar, count in zip(bars, s_counts):
        if count > 0:
            ax_status.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                           str(count), ha="center", va="bottom", color=COL_TEXT, fontsize=9)
    _style_ax(ax_status, f"Overall Pair Status  (total={sum(s_counts)})", "Status")
    ax_status.tick_params(axis="x", labelsize=7, rotation=20)

    plot_hdet   (axes[3][0], metrics["h_det"])
    plot_inlier (axes[3][1], metrics["inlier_ratios"])

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    import datetime
    out_ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(ANALYSIS_DIR, f"gf_metric_histograms_{out_ts}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=COL_BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
