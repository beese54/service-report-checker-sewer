"""
plot_stage3n_histograms.py — Stage 3N Metric Histograms
========================================================
Produces an 8-panel dark-theme histogram image showing the distribution
of every diagnostic metric from the Stage 3N (D2/D5 + U2/U5) washing
confidence pipeline.

Usage:
    python analytics/plot_stage3n_histograms.py [path/to/pipeline_run_*.json]

Output:
    pipeline_output/full_run_2/stage3n_metric_histograms.png
"""

import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker"
DEFAULT_JSON = os.path.join(
    BASE_DIR, "pipeline_output", "full_run_2",
    "pipeline_run_20260306_080308.json"
)
OUT_PATH = os.path.join(
    BASE_DIR, "pipeline_output", "full_run_2",
    "stage3n_metric_histograms.png"
)

# ── Thresholds ───────────────────────────────────────────────────────────────
THRESH_CONF_HIGH   = 0.50
THRESH_CONF_MED    = 0.30
THRESH_KP_RATIO    = 1.0
THRESH_ENTROPY     = 0.3
THRESH_STD         = 0.0    # any increase
THRESH_EDGE        = 0.0    # any increase
THRESH_LAP         = 0.0    # any increase
THRESH_MATCH_RATIO = 0.0    # informational

# ── Colours ──────────────────────────────────────────────────────────────────
COL_BG     = "#1A1A2E"
COL_AX     = "#16213E"
COL_GRID   = "#2A2A4A"
COL_TEXT   = "#E0E0E0"
COL_PASS   = "#4CAF50"
COL_FAIL   = "#E53935"
COL_THRESH = "#FF9800"
COL_HIGH   = "#4CAF50"
COL_MED    = "#FFC107"
COL_LOW    = "#E53935"
COL_INFO   = "#1E88E5"
COL_NA     = "#9E9E9E"


# =============================================================================
# Data extraction
# =============================================================================

def extract_metrics(folders):
    conf   = []   # (value, tier)
    kp_ratio    = []
    entropy     = []
    std_incr    = []
    edge_incr   = []
    lap_incr    = []
    match_ratio = []
    ransac      = []
    tiers       = []   # for tier bar chart

    for folder in folders:
        detail = folder.get("detail") or {}
        s3n    = detail.get("stage3n") or {}
        for pair_key, pdata in (s3n.get("pair_results") or {}).items():
            if pdata.get("status") != "OK":
                continue
            m    = pdata.get("metrics") or {}
            sift = pdata.get("sift_stats") or {}
            tier = pdata.get("washing_tier")
            tiers.append(tier)

            c = m.get("washing_confidence")
            if c is not None:
                conf.append((float(c), tier))

            v = m.get("kp_ratio")
            if v is not None:
                kp_ratio.append(float(v))

            v = m.get("entropy_increase")
            if v is not None:
                entropy.append(float(v))

            v = m.get("std_increase_pct")
            if v is not None:
                std_incr.append(float(v))

            v = m.get("edge_increase_pct")
            if v is not None:
                edge_incr.append(float(v))

            v = m.get("lap_increase_pct")
            if v is not None:
                lap_incr.append(float(v))

            v = m.get("match_ratio")
            if v is not None:
                match_ratio.append(float(v))

            v = sift.get("ransac_inliers")
            if v is not None:
                ransac.append(int(v))

    return {
        "conf":        conf,
        "kp_ratio":    kp_ratio,
        "entropy":     entropy,
        "std_incr":    std_incr,
        "edge_incr":   edge_incr,
        "lap_incr":    lap_incr,
        "match_ratio": match_ratio,
        "ransac":      ransac,
        "tiers":       tiers,
    }


# =============================================================================
# Style helpers
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


def _stats_box(ax, total, n_pass, label="PASS >= thresh"):
    n_fail = total - n_pass
    txt = (f"n = {total}\n"
           f"PASS: {n_pass} ({n_pass/total:.0%})\n"
           f"FAIL: {n_fail} ({n_fail/total:.0%})")
    ax.text(0.98, 0.97, txt,
            transform=ax.transAxes, ha="right", va="top",
            color=COL_TEXT, fontsize=8,
            bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.85))


def _thresh_vline(ax, value, label):
    ax.axvline(value, color=COL_THRESH, linewidth=1.8, linestyle="--",
               zorder=4, label=f"Threshold = {label}")


def _pass_fail_legend(ax):
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=COL_PASS, label="PASS"),
            mpatches.Patch(facecolor=COL_FAIL, label="FAIL"),
        ],
        fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT,
    )


# =============================================================================
# Individual panels
# =============================================================================

def plot_washing_conf(ax, data):
    conf = data["conf"]
    if not conf:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Washing Confidence", "Confidence")
        return

    vals = [v for v, _ in conf]
    high = [v for v, t in conf if t == "HIGH"]
    med  = [v for v, t in conf if t == "MEDIUM"]
    low  = [v for v, t in conf if t == "LOW"]

    bins = np.linspace(0, 1, 41)
    ax.hist(high, bins=bins, color=COL_HIGH, alpha=0.85, label="HIGH (>=65%)", zorder=3)
    ax.hist(med,  bins=bins, color=COL_MED,  alpha=0.85, label="MED (35-64%)", zorder=3)
    ax.hist(low,  bins=bins, color=COL_LOW,  alpha=0.85, label="LOW (<35%)",   zorder=3)
    ax.axvline(THRESH_CONF_HIGH, color=COL_THRESH, linewidth=1.8, linestyle="-.",
               zorder=4, label=f"HIGH threshold = {THRESH_CONF_HIGH:.0%}")
    ax.axvline(THRESH_CONF_MED,  color=COL_THRESH, linewidth=1.8, linestyle="--",
               zorder=4, label=f"MED threshold = {THRESH_CONF_MED:.0%}")
    ax.axvspan(THRESH_CONF_MED, THRESH_CONF_HIGH, color=COL_MED, alpha=0.07, zorder=2)

    _style_ax(ax, f"Washing Confidence  (n={len(vals)})", "Confidence")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)

    n_high = len(high)
    n_med  = len(med)
    n_low  = len(low)
    total  = len(vals)
    ax.text(0.98, 0.97,
            f"n = {total}\n"
            f"HIGH: {n_high} ({n_high/total:.0%})\n"
            f"MED:  {n_med}  ({n_med/total:.0%})\n"
            f"LOW:  {n_low}  ({n_low/total:.0%})",
            transform=ax.transAxes, ha="right", va="top",
            color=COL_TEXT, fontsize=8,
            bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.85))


def plot_tier_bar(ax, data):
    tiers = data["tiers"]
    if not tiers:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Washing Tier Distribution", "Tier")
        return

    c = Counter(tiers)
    cats    = ["HIGH", "MEDIUM", "LOW"]
    colours = [COL_HIGH, COL_MED, COL_LOW]
    counts  = [c.get(k, 0) for k in cats]
    total   = sum(counts)

    bars = ax.bar(cats, counts, color=colours, alpha=0.85, zorder=3)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{count}\n({count/total:.0%})",
                    ha="center", va="bottom", color=COL_TEXT, fontsize=9)

    _style_ax(ax, f"Washing Tier Distribution  (n={total})", "Tier")
    ax.text(0.98, 0.97, f"PASS tier: HIGH only",
            transform=ax.transAxes, ha="right", va="top",
            color=COL_THRESH, fontsize=8,
            bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.85))


def plot_kp_ratio(ax, data):
    vals = data["kp_ratio"]
    if not vals:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Keypoint Ratio (after/before)", "Ratio")
        return

    passed = [v for v in vals if v > THRESH_KP_RATIO]
    failed = [v for v in vals if v <= THRESH_KP_RATIO]

    # Clip extreme outliers for display
    clip = np.percentile(vals, 98)
    p_c  = [min(v, clip) for v in passed]
    f_c  = [min(v, clip) for v in failed]
    bins = np.linspace(0, clip * 1.05, 45)

    ax.hist(p_c, bins=bins, color=COL_PASS, alpha=0.85, label="PASS (>1.0)", zorder=3)
    ax.hist(f_c, bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL (<=1.0)", zorder=3)
    _thresh_vline(ax, THRESH_KP_RATIO, "1.0")
    _style_ax(ax, f"Keypoint Ratio after/before  (n={len(vals)})",
              f"kp_ratio  [clipped at p98={clip:.1f}]")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    _stats_box(ax, len(vals), len(passed))


def plot_entropy(ax, data):
    vals = data["entropy"]
    if not vals:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Entropy Increase", "nats")
        return

    passed = [v for v in vals if v >= THRESH_ENTROPY]
    failed = [v for v in vals if v < THRESH_ENTROPY]

    lo   = min(min(vals) - 0.05, -0.1)
    hi   = max(max(vals) + 0.05,  0.5)
    bins = np.linspace(lo, hi, 45)

    ax.hist(passed, bins=bins, color=COL_PASS, alpha=0.85, label="PASS", zorder=3)
    ax.hist(failed, bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL", zorder=3)
    _thresh_vline(ax, THRESH_ENTROPY, f"{THRESH_ENTROPY} nats")
    _style_ax(ax, f"Entropy Increase  (n={len(vals)})", "entropy_increase (nats)")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    _stats_box(ax, len(vals), len(passed))


def _plot_increase_pct(ax, vals, thresh, title, xlabel):
    if not vals:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, title, xlabel)
        return

    passed = [v for v in vals if v > thresh]
    failed = [v for v in vals if v <= thresh]

    # Log-scale for these (can be very large)
    all_pos = [v for v in vals if v > 0]
    if all_pos:
        lo = max(0.1, min(all_pos) * 0.8)
        hi = max(all_pos) * 1.3
        bins = np.logspace(np.log10(lo), np.log10(hi), 45)
        # include zeros / negatives in a separate underflow bin
        neg  = [v for v in vals if v <= 0]
        ax.hist([max(lo, v) for v in passed if v > 0],
                bins=bins, color=COL_PASS, alpha=0.85, label="PASS (>0%)", zorder=3)
        ax.hist([max(lo, v) for v in failed if v > 0],
                bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL (<=0%)", zorder=3)
        if neg:
            ax.bar(lo * 0.5, len(neg), width=lo * 0.3,
                   color=COL_FAIL, alpha=0.85, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, _: f"{x:.0f}%"))
    else:
        bins = np.linspace(min(vals) - 5, 5, 30)
        ax.hist(failed, bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL", zorder=3)

    if thresh == 0:
        ax.axvline(max(0.1, thresh + 0.5), color=COL_THRESH, linewidth=1.8,
                   linestyle="--", zorder=4, label="Threshold > 0%")
    else:
        _thresh_vline(ax, thresh, f"{thresh}%")

    _style_ax(ax, f"{title}  (n={len(vals)})", xlabel)
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    _stats_box(ax, len(vals), len(passed))


def plot_std_incr(ax, data):
    _plot_increase_pct(ax, data["std_incr"], THRESH_STD,
                       "Std Dev Increase %", "std_increase_pct [log]")


def plot_edge_incr(ax, data):
    _plot_increase_pct(ax, data["edge_incr"], THRESH_EDGE,
                       "Edge Density Increase %", "edge_increase_pct [log]")


def plot_lap_incr(ax, data):
    _plot_increase_pct(ax, data["lap_incr"], THRESH_LAP,
                       "Laplacian Variance Increase %", "lap_increase_pct [log]")


def plot_ransac(ax, data):
    vals = data["ransac"]
    if not vals:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "RANSAC Inliers (S3N pairs)", "Inliers")
        return

    bins = np.linspace(0, min(max(vals) + 5, 200), 45)
    overflow = [v for v in vals if v > 200]
    display  = [min(v, 200) for v in vals]

    ax.hist(display, bins=bins, color=COL_INFO, alpha=0.85, zorder=3)
    if overflow:
        ax.text(0.98, 0.70, f">{200}: {len(overflow)} pairs",
                transform=ax.transAxes, ha="right", va="top",
                color=COL_THRESH, fontsize=8)

    median_v = float(np.median(vals))
    ax.axvline(median_v, color=COL_THRESH, linewidth=1.5, linestyle="--",
               zorder=4, label=f"Median = {median_v:.0f}")

    _style_ax(ax, f"RANSAC Inliers (S3N pairs)  (n={len(vals)})",
              "ransac_inliers  [clipped at 200]")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    ax.text(0.98, 0.97,
            f"n = {len(vals)}\nMedian: {median_v:.0f}\nMax: {max(vals)}",
            transform=ax.transAxes, ha="right", va="top",
            color=COL_TEXT, fontsize=8,
            bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.85))


# =============================================================================
# Main
# =============================================================================

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON
    print(f"Reading: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)

    folders = d.get("folders", [])
    run_date = d.get("run_date", "")
    metrics  = extract_metrics(folders)

    n_pairs = len(metrics["conf"])
    print(f"  S3N pairs (status=OK): {n_pairs}")

    # ── Figure: 4 rows x 2 cols ───────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.patch.set_facecolor(COL_BG)

    fig.suptitle(
        f"Stage 3N — Washing Confidence Metric Distributions  (D2/D5 + U2/U5)\n"
        f"Run: {run_date}   Pairs processed: {n_pairs}",
        color=COL_TEXT, fontsize=13, fontweight="bold", y=0.995,
    )

    plot_washing_conf(axes[0][0], metrics)
    plot_tier_bar    (axes[0][1], metrics)
    plot_kp_ratio    (axes[1][0], metrics)
    plot_entropy     (axes[1][1], metrics)
    plot_std_incr    (axes[2][0], metrics)
    plot_edge_incr   (axes[2][1], metrics)
    plot_lap_incr    (axes[3][0], metrics)
    plot_ransac      (axes[3][1], metrics)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight",
                facecolor=COL_BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
