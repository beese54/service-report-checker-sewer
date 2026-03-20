"""
plot_stage4n_histograms.py — Stage 4N Metric Histograms
========================================================
Produces a 10-panel dark-theme histogram image showing the distribution
of every diagnostic metric from the Stage 4N (D3/D6 + U3/U6) geometry-first
pipeline.

Usage:
    python analytics/plot_stage4n_histograms.py [path/to/pipeline_run_*.json]

Output:
    pipeline_output/full_run_2/stage4n_metric_histograms.png
"""

import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker"
DEFAULT_JSON = os.path.join(
    BASE_DIR, "pipeline_output", "full_run_2",
    "pipeline_run_20260306_080308.json"
)
OUT_PATH = os.path.join(
    BASE_DIR, "pipeline_output", "full_run_2",
    "stage4n_metric_histograms.png"
)

# ── Thresholds (mirror stage4n_geometry.py / pipeline_config.py) ─────────────
THRESH_BLUR       = 35.0
THRESH_SSIM       = 0.05
THRESH_INLIER_R   = 0.05
THRESH_RANSAC     = 10      # signal 1: inliers >= 10
THRESH_PCT_CHG    = 5.0     # signal 2: pct_changed >= 5%
THRESH_ENTROPY    = 0.3     # signal 4: entropy confirmed >= 0.3 nats
THRESH_GREASE     = 2.0     # flag if >= 2%
HDET_MIN          = 0.2
HDET_MAX          = 5.0

# ── Colours ──────────────────────────────────────────────────────────────────
COL_BG     = "#1A1A2E"
COL_AX     = "#16213E"
COL_GRID   = "#2A2A4A"
COL_TEXT   = "#E0E0E0"
COL_PASS   = "#4CAF50"
COL_FAIL   = "#E53935"
COL_THRESH = "#FF9800"
COL_INFO   = "#1E88E5"
COL_NA     = "#9E9E9E"
COL_REV    = "#FFC107"


# =============================================================================
# Data extraction
# =============================================================================

def extract_metrics(folders):
    score       = []
    blur        = []
    ssim        = []
    ransac      = []
    pct_chg     = []
    inlier_r    = []
    hdet        = []
    grease_pct  = []
    entropy_d   = []
    water_conf  = []
    pair_status = []

    for folder in folders:
        detail = folder.get("detail") or {}
        s4n    = detail.get("stage4n") or {}
        for pair_key, p in (s4n.get("pair_results") or {}).items():
            status = p.get("status")
            pair_status.append(status)

            gate = p.get("gate") or {}
            sift = p.get("sift_stats") or {}

            v = p.get("score")
            if v is not None:
                score.append(int(v))

            v = gate.get("blur_score")
            if v is not None:
                blur.append((float(v), float(v) >= THRESH_BLUR))

            v = sift.get("ssim_score")
            if v is not None:
                ssim.append((float(v), float(v) >= THRESH_SSIM))

            v = sift.get("ransac_inliers")
            if v is not None:
                ransac.append(int(v))

            v = sift.get("pct_changed")
            if v is not None:
                pct_chg.append((float(v), float(v) >= THRESH_PCT_CHG))

            v = sift.get("inlier_ratio")
            if v is not None:
                inlier_r.append((float(v), float(v) >= THRESH_INLIER_R))

            v = sift.get("homography_det")
            if v is not None:
                hdet.append((float(v), HDET_MIN <= float(v) <= HDET_MAX))

            g = (p.get("grease") or {}).get("grease_pct")
            if g is not None:
                grease_pct.append((float(g), float(g) < THRESH_GREASE))   # pass = below thresh

            v = (p.get("texture") or {}).get("entropy_delta")
            if v is not None:
                entropy_d.append((float(v), float(v) >= THRESH_ENTROPY))

            v = (p.get("water") or {}).get("water_confidence")
            if v is not None:
                water_conf.append(float(v))

    return {
        "score":      score,
        "blur":       blur,
        "ssim":       ssim,
        "ransac":     ransac,
        "pct_chg":    pct_chg,
        "inlier_r":   inlier_r,
        "hdet":       hdet,
        "grease_pct": grease_pct,
        "entropy_d":  entropy_d,
        "water_conf": water_conf,
        "pair_status":pair_status,
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


def _stats_box(ax, total, n_pass, extra=""):
    n_fail = total - n_pass
    txt = (f"n = {total}\n"
           f"PASS: {n_pass} ({n_pass/total:.0%})\n"
           f"FAIL: {n_fail} ({n_fail/total:.0%})")
    if extra:
        txt += f"\n{extra}"
    ax.text(0.98, 0.97, txt,
            transform=ax.transAxes, ha="right", va="top",
            color=COL_TEXT, fontsize=8,
            bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.85))


def _thresh_vline(ax, value, label):
    ax.axvline(value, color=COL_THRESH, linewidth=1.8, linestyle="--",
               zorder=4, label=f"Threshold = {label}")


def _hist_pass_fail(ax, pairs, bins, title, xlabel, thresh_label=""):
    passed = [v for v, p in pairs if p]
    failed = [v for v, p in pairs if not p]
    ax.hist(passed, bins=bins, color=COL_PASS, alpha=0.85, label="PASS", zorder=3)
    ax.hist(failed, bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL", zorder=3)
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    _style_ax(ax, f"{title}  (n={len(pairs)})", xlabel)
    _stats_box(ax, len(pairs), len(passed))


# =============================================================================
# Individual panels
# =============================================================================

def plot_pair_status(ax, data):
    counts = Counter(data["pair_status"])
    order  = ["PASS", "REVIEW", "FAIL", "GATE_REJECTED", "ALIGNMENT_FAILED", "MISSING_IMAGES"]
    cols   = [COL_PASS, COL_REV, COL_FAIL, "#9C27B0", COL_INFO, COL_NA]
    values = [counts.get(k, 0) for k in order]
    labels = ["PASS", "REVIEW", "FAIL", "GATE\nREJ", "ALIGN\nFAIL", "MISSING"]
    total  = sum(values)

    bars = ax.bar(labels, values, color=cols, alpha=0.85, zorder=3)
    for bar, count in zip(bars, values):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{count}\n({count/total:.0%})",
                    ha="center", va="bottom", color=COL_TEXT, fontsize=8)

    _style_ax(ax, f"Pair Status Breakdown  (total={total})", "Status")
    ax.tick_params(axis="x", labelsize=8)


def plot_score(ax, data):
    scores = data["score"]
    if not scores:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Pair Score (0–5)", "Score")
        return

    bins  = np.arange(-0.5, 6.5, 1)
    high  = [s for s in scores if s >= 3]
    rev   = [s for s in scores if s == 2]
    low   = [s for s in scores if s < 2]

    ax.hist(high, bins=bins, color=COL_PASS, alpha=0.85, label="PASS (>=3)", zorder=3)
    ax.hist(rev,  bins=bins, color=COL_REV,  alpha=0.85, label="REVIEW (2)", zorder=3)
    ax.hist(low,  bins=bins, color=COL_FAIL, alpha=0.85, label="FAIL (<2)",  zorder=3)
    ax.axvline(2.5, color=COL_THRESH, linewidth=1.8, linestyle="--", zorder=4,
               label="PASS threshold = 3")
    ax.set_xticks(range(6))
    _style_ax(ax, f"Pair Score Distribution  (n={len(scores)})", "Score (0–5)")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)

    total = len(scores)
    ax.text(0.98, 0.97,
            f"n = {total}\n"
            f"PASS (>=3): {len(high)} ({len(high)/total:.0%})\n"
            f"REVIEW (2): {len(rev)}  ({len(rev)/total:.0%})\n"
            f"FAIL (<2):  {len(low)}  ({len(low)/total:.0%})",
            transform=ax.transAxes, ha="right", va="top",
            color=COL_TEXT, fontsize=8,
            bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.85))


def plot_blur(ax, data):
    pairs = data["blur"]
    if not pairs:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Blur Score (Gate)", "Laplacian Variance")
        return

    vals  = [v for v, _ in pairs]
    bins  = np.linspace(max(0, min(vals) - 5), min(max(vals) + 5, 500), 45)
    _thresh_vline(ax, THRESH_BLUR, f"{THRESH_BLUR}")
    _hist_pass_fail(ax, pairs, bins,
                    "Blur Score — Gate Phase",
                    f"Laplacian Variance  [threshold={THRESH_BLUR}]")


def plot_ssim(ax, data):
    pairs = data["ssim"]
    if not pairs:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "SSIM Score", "SSIM")
        return

    vals = [v for v, _ in pairs]
    bins = np.linspace(max(0, min(vals) - 0.01), min(1.0, max(vals) + 0.01), 45)
    _thresh_vline(ax, THRESH_SSIM, f"{THRESH_SSIM}")
    _hist_pass_fail(ax, pairs, bins, "SSIM Score", "Structural Similarity Index")


def plot_ransac(ax, data):
    vals = data["ransac"]
    if not vals:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "RANSAC Inliers", "Inliers")
        return

    passed = [v for v in vals if v >= THRESH_RANSAC]
    failed = [v for v in vals if v < THRESH_RANSAC]

    log_bins = np.logspace(np.log10(max(1, min(vals) * 0.8)),
                           np.log10(max(vals) * 1.2), 45)
    ax.hist(passed, bins=log_bins, color=COL_PASS, alpha=0.85, label=f"PASS (>={THRESH_RANSAC})", zorder=3)
    ax.hist(failed, bins=log_bins, color=COL_FAIL, alpha=0.85, label=f"FAIL (<{THRESH_RANSAC})",  zorder=3)
    ax.set_xscale("log")
    _thresh_vline(ax, THRESH_RANSAC, str(THRESH_RANSAC))
    _style_ax(ax, f"RANSAC Inliers — Signal 1  (n={len(vals)})", "Inliers (log scale)")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    _stats_box(ax, len(vals), len(passed),
               f"Median: {int(np.median(vals))}  Max: {max(vals)}")


def plot_pct_changed(ax, data):
    pairs = data["pct_chg"]
    if not pairs:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Pct Changed — Signal 2", "% Pixels Changed")
        return

    vals  = [v for v, _ in pairs]
    bins  = np.linspace(0, min(max(vals) + 2, 105), 45)
    _thresh_vline(ax, THRESH_PCT_CHG, f"{THRESH_PCT_CHG}%")
    _hist_pass_fail(ax, pairs, bins,
                    f"% Pixels Changed — Signal 2",
                    f"pct_changed  [threshold={THRESH_PCT_CHG}%]")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}%"))


def plot_inlier_ratio(ax, data):
    pairs = data["inlier_r"]
    if not pairs:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Inlier Ratio", "Ratio")
        return

    vals  = [v for v, _ in pairs]
    bins  = np.linspace(0, min(1.0, max(vals) + 0.02), 45)
    _thresh_vline(ax, THRESH_INLIER_R, f"{THRESH_INLIER_R:.0%}")
    _hist_pass_fail(ax, pairs, bins,
                    "Inlier Ratio",
                    f"inlier_ratio  [threshold={THRESH_INLIER_R:.0%}]")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))


def plot_hdet(ax, data):
    pairs = data["hdet"]
    if not pairs:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Homography Determinant", "det(H[:2,:2])")
        return

    clip_lo, clip_hi = -2.0, 8.0
    clipped = [(max(clip_lo, min(clip_hi, v)), p) for v, p in pairs]
    bins = np.linspace(clip_lo, clip_hi, 45)

    ax.axvline(HDET_MIN, color=COL_THRESH, linewidth=1.8, linestyle="--", zorder=4,
               label=f"Min = {HDET_MIN}")
    ax.axvline(HDET_MAX, color=COL_THRESH, linewidth=1.8, linestyle="-.", zorder=4,
               label=f"Max = {HDET_MAX}")
    ax.axvspan(HDET_MIN, min(HDET_MAX, clip_hi), color=COL_PASS, alpha=0.07, zorder=2)
    _hist_pass_fail(ax, clipped, bins,
                    f"Homography Determinant  (informational)",
                    f"det(H[:2,:2])  [clipped to [{clip_lo}, {clip_hi}]]")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)


def plot_grease(ax, data):
    pairs = data["grease_pct"]
    if not pairs:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Grease % — Signal 5", "Grease %")
        return

    vals  = [v for v, _ in pairs]
    bins  = np.linspace(0, min(max(vals) + 0.5, 15), 45)
    _thresh_vline(ax, THRESH_GREASE, f"{THRESH_GREASE}%  (flag above)")
    _hist_pass_fail(ax, pairs, bins,
                    f"Grease % — Signal 5  (pass = NOT flagged)",
                    f"grease_pct  [flag threshold = {THRESH_GREASE}%]")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x:.1f}%"))


def plot_entropy(ax, data):
    pairs = data["entropy_d"]
    if not pairs:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Entropy Delta — Signal 3", "nats")
        return

    vals  = [v for v, _ in pairs]
    lo    = min(min(vals) - 0.05, -0.1)
    hi    = max(max(vals) + 0.05,  0.5)
    bins  = np.linspace(lo, hi, 45)
    _thresh_vline(ax, THRESH_ENTROPY, f"{THRESH_ENTROPY} nats")
    _hist_pass_fail(ax, pairs, bins,
                    f"Entropy Delta (texture) — Signal 3",
                    f"entropy_delta (nats)  [threshold={THRESH_ENTROPY}]")


def plot_water(ax, data):
    vals = data["water_conf"]
    if not vals:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=COL_NA)
        _style_ax(ax, "Water Confidence — Signal 4", "Confidence")
        return

    bins     = np.linspace(0, 1, 41)
    detected = [v for v in vals if v >= 0.1]
    not_det  = [v for v in vals if v < 0.1]

    ax.hist(detected, bins=bins, color=COL_INFO, alpha=0.85, label="Detected",     zorder=3)
    ax.hist(not_det,  bins=bins, color=COL_NA,   alpha=0.85, label="Not detected", zorder=3)
    ax.axvline(0.1, color=COL_THRESH, linewidth=1.8, linestyle="--", zorder=4,
               label="Detect threshold ~0.1")

    _style_ax(ax, f"Water Confidence — Signal 4  (n={len(vals)})",
              "water_confidence")
    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)

    n_det = len(detected)
    ax.text(0.98, 0.97,
            f"n = {len(vals)}\nDetected: {n_det} ({n_det/len(vals):.0%})\n"
            f"Not detected: {len(not_det)} ({len(not_det)/len(vals):.0%})",
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

    folders  = d.get("folders", [])
    run_date = d.get("run_date", "")
    metrics  = extract_metrics(folders)

    n_pairs = len(metrics["pair_status"])
    print(f"  S4N pairs total: {n_pairs}")

    # ── Figure: 5 rows x 2 cols ───────────────────────────────────────────
    fig, axes = plt.subplots(5, 2, figsize=(18, 27))
    fig.patch.set_facecolor(COL_BG)

    fig.suptitle(
        f"Stage 4N — Geometry-First Metric Distributions  (D3/D6 + U3/U6)\n"
        f"Run: {run_date}   Total pairs: {n_pairs}",
        color=COL_TEXT, fontsize=13, fontweight="bold", y=0.995,
    )

    plot_pair_status (axes[0][0], metrics)
    plot_score       (axes[0][1], metrics)
    plot_blur        (axes[1][0], metrics)
    plot_ssim        (axes[1][1], metrics)
    plot_ransac      (axes[2][0], metrics)
    plot_pct_changed (axes[2][1], metrics)
    plot_entropy     (axes[3][0], metrics)
    plot_water       (axes[3][1], metrics)
    plot_grease      (axes[4][0], metrics)
    plot_hdet        (axes[4][1], metrics)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight",
                facecolor=COL_BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
