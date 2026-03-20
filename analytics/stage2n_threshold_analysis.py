"""
Stage 2N Threshold Analysis — RANSAC Inliers & Inlier Coverage Histograms
==========================================================================
Loads the full pipeline run JSON and analyses the distribution of two Stage 2N
pass/fail metrics across all D and U pairs:
  - ransac_inliers    (current threshold: >= 10)
  - inlier_coverage_pct (current threshold: >= 25%)

Current pass rule: folder passes if at least one pair (D or U) satisfies BOTH.

Output:
    pipeline_output/full_run_2/stage2n_threshold_analysis.png

Usage:
    python analytics/stage2n_threshold_analysis.py [path/to/pipeline_run_*.json]
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker"
DEFAULT_JSON = os.path.join(
    BASE_DIR, "pipeline_output", "full_run_2",
    "pipeline_run_20260306_080308.json"
)
OUT_PATH = os.path.join(
    BASE_DIR, "pipeline_output", "full_run_2",
    "stage2n_threshold_analysis.png"
)

# ── Current thresholds ───────────────────────────────────────────────────────
THRESH_INLIERS  = 10
THRESH_COVERAGE = 25.0   # percent

# ── Colour palette ───────────────────────────────────────────────────────────
COL_BG     = "#1A1A2E"
COL_AX     = "#16213E"
COL_GRID   = "#2A2A4A"
COL_TEXT   = "#E0E0E0"
COL_PASS   = "#4CAF50"   # green
COL_FAIL   = "#E53935"   # red
COL_THRESH = "#FF9800"   # orange
COL_D      = "#1E88E5"   # blue  — D-only pass
COL_U      = "#AB47BC"   # purple — U-only pass


# =============================================================================
# Data loading
# =============================================================================

def load_pairs(json_path):
    """Return (d_pairs, u_pairs, paired_folders, single_folders).

    d_pairs, u_pairs: list of dicts with keys ransac_inliers, inlier_coverage_pct
    paired_folders:   list of dicts with keys d_inliers, d_cov, u_inliers, u_cov
    single_folders:   count of folders that had only one OK pair
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    d_pairs    = []
    u_pairs    = []
    paired     = []   # both D and U OK
    single_count = 0

    folders = data.get("folders", [])
    for folder in folders:
        detail = folder.get("detail") or {}
        s2n    = detail.get("stage2n") or {}
        ps     = s2n.get("pair_stats") or {}

        d_stat = ps.get("D") or {}
        u_stat = ps.get("U") or {}

        d_ok = d_stat.get("status") == "OK"
        u_ok = u_stat.get("status") == "OK"

        if d_ok:
            d_pairs.append({
                "ransac_inliers":      d_stat["ransac_inliers"],
                "inlier_coverage_pct": d_stat["inlier_coverage_pct"],
                "folder":              folder.get("name", ""),
            })
        if u_ok:
            u_pairs.append({
                "ransac_inliers":      u_stat["ransac_inliers"],
                "inlier_coverage_pct": u_stat["inlier_coverage_pct"],
                "folder":              folder.get("name", ""),
            })

        if d_ok and u_ok:
            paired.append({
                "d_inliers": d_stat["ransac_inliers"],
                "d_cov":     d_stat["inlier_coverage_pct"],
                "u_inliers": u_stat["ransac_inliers"],
                "u_cov":     u_stat["inlier_coverage_pct"],
                "folder":    folder.get("name", ""),
            })
        elif d_ok or u_ok:
            single_count += 1

    return d_pairs, u_pairs, paired, single_count


# =============================================================================
# Style helpers
# =============================================================================

def _style_ax(ax, title, xlabel="", ylabel="Count", fontsize=10):
    ax.set_facecolor(COL_AX)
    ax.set_title(title, color=COL_TEXT, fontsize=fontsize, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=COL_TEXT, fontsize=9)
    ax.set_ylabel(ylabel, color=COL_TEXT, fontsize=9)
    ax.tick_params(colors=COL_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_GRID)
    ax.yaxis.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def _stats_box(ax, total, n_pass, label_pass="PASS", label_fail="FAIL"):
    n_fail = total - n_pass
    pct_p  = n_pass / total * 100 if total else 0
    pct_f  = n_fail / total * 100 if total else 0
    txt = (f"n = {total}\n"
           f"{label_pass}: {n_pass} ({pct_p:.0f}%)\n"
           f"{label_fail}: {n_fail} ({pct_f:.0f}%)")
    ax.text(0.98, 0.97, txt,
            transform=ax.transAxes, ha="right", va="top",
            color=COL_TEXT, fontsize=8,
            bbox=dict(facecolor=COL_AX, edgecolor=COL_GRID, alpha=0.85))


def _thresh_line(ax, value, label, vertical=True):
    fn = ax.axvline if vertical else ax.axhline
    fn(value, color=COL_THRESH, linewidth=1.8, linestyle="--", zorder=4,
       label=f"Threshold = {label}")


# =============================================================================
# Row 1 — RANSAC Inliers histograms (log x-axis)
# =============================================================================

def plot_inliers(ax, pairs, prefix):
    vals    = [p["ransac_inliers"] for p in pairs]
    passed  = [v for v in vals if v >= THRESH_INLIERS]
    failed  = [v for v in vals if v < THRESH_INLIERS]

    log_bins = np.logspace(np.log10(max(1, min(vals) * 0.8)),
                           np.log10(max(vals) * 1.2), 45)

    ax.hist(passed, bins=log_bins, color=COL_PASS, alpha=0.85, label="PASS", zorder=3)
    ax.hist(failed, bins=log_bins, color=COL_FAIL, alpha=0.85, label="FAIL", zorder=3)
    ax.set_xscale("log")
    _thresh_line(ax, THRESH_INLIERS, str(THRESH_INLIERS))
    _style_ax(ax,
              f"RANSAC Inliers — {prefix} Pairs",
              "RANSAC Inliers (log scale)")

    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    _stats_box(ax, len(vals), len(passed))

    median_v = float(np.median(vals))
    ax.axvline(median_v, color="#90CAF9", linewidth=1.2, linestyle=":",
               zorder=3, label=f"Median = {median_v:.0f}")


# =============================================================================
# Row 2 — Inlier Coverage % histograms (linear, 6.25% bins)
# =============================================================================

def plot_coverage(ax, pairs, prefix):
    vals   = [p["inlier_coverage_pct"] for p in pairs]
    # Quantised at 6.25% steps (4x4 grid)
    bin_edges = np.arange(0, 106.25, 6.25)   # 0, 6.25, 12.5, … 100, (106.25 right edge)
    centres   = bin_edges[:-1] + 3.125

    counts_pass = np.zeros(len(centres))
    counts_fail = np.zeros(len(centres))

    for v in vals:
        idx = min(int(v / 6.25), len(centres) - 1)
        if v >= THRESH_COVERAGE:
            counts_pass[idx] += 1
        else:
            counts_fail[idx] += 1

    width = 5.5   # slightly narrower than 6.25 so gaps are visible
    ax.bar(centres, counts_fail, width=width, color=COL_FAIL, alpha=0.85,
           label="FAIL", zorder=3)
    ax.bar(centres, counts_pass, width=width, color=COL_PASS, alpha=0.85,
           label="PASS", bottom=counts_fail, zorder=3)

    _thresh_line(ax, THRESH_COVERAGE, f"{THRESH_COVERAGE:.0f}%")
    ax.set_xlim(-3, 106)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}%"))
    _style_ax(ax,
              f"Inlier Coverage % — {prefix} Pairs",
              "Coverage % (6.25% steps, 4x4 grid)")

    ax.legend(fontsize=8, facecolor=COL_AX, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    n_pass = sum(1 for v in vals if v >= THRESH_COVERAGE)
    _stats_box(ax, len(vals), n_pass)


# =============================================================================
# Row 3 — Scatter plots for paired folders (D vs U)
# =============================================================================

def _pair_colours(paired):
    """Return colour list based on pass/fail per D and U."""
    cols = []
    for p in paired:
        d_pass = p["d_inliers"] >= THRESH_INLIERS and p["d_cov"] >= THRESH_COVERAGE
        u_pass = p["u_inliers"] >= THRESH_INLIERS and p["u_cov"] >= THRESH_COVERAGE
        if d_pass and u_pass:
            cols.append(COL_PASS)     # both pass
        elif d_pass:
            cols.append(COL_D)        # D only
        elif u_pass:
            cols.append(COL_U)        # U only
        else:
            cols.append(COL_FAIL)     # neither
    return cols


def plot_scatter_inliers(ax, paired):
    if not paired:
        ax.text(0.5, 0.5, "No paired data", transform=ax.transAxes,
                ha="center", va="center", color=COL_TEXT)
        _style_ax(ax, "D vs U: RANSAC Inliers", "D Inliers (log)", "U Inliers (log)")
        return

    d_vals = [p["d_inliers"] for p in paired]
    u_vals = [p["u_inliers"] for p in paired]
    cols   = _pair_colours(paired)

    ax.scatter(d_vals, u_vals, c=cols, alpha=0.65, s=18, zorder=3, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _thresh_line(ax, THRESH_INLIERS, str(THRESH_INLIERS), vertical=True)
    _thresh_line(ax, THRESH_INLIERS, str(THRESH_INLIERS), vertical=False)

    _style_ax(ax,
              f"D vs U RANSAC Inliers  (n={len(paired)} folders)",
              "D Inliers (log scale)",
              "U Inliers (log scale)")
    _scatter_legend(ax)


def plot_scatter_coverage(ax, paired):
    if not paired:
        ax.text(0.5, 0.5, "No paired data", transform=ax.transAxes,
                ha="center", va="center", color=COL_TEXT)
        _style_ax(ax, "D vs U: Coverage %", "D Coverage %", "U Coverage %")
        return

    d_vals = [p["d_cov"] for p in paired]
    u_vals = [p["u_cov"] for p in paired]
    cols   = _pair_colours(paired)

    jitter = np.random.default_rng(0).uniform(-1.2, 1.2, size=(len(d_vals), 2))
    ax.scatter(np.array(d_vals) + jitter[:, 0],
               np.array(u_vals) + jitter[:, 1],
               c=cols, alpha=0.65, s=18, zorder=3, linewidths=0)

    _thresh_line(ax, THRESH_COVERAGE, f"{THRESH_COVERAGE:.0f}%", vertical=True)
    _thresh_line(ax, THRESH_COVERAGE, f"{THRESH_COVERAGE:.0f}%", vertical=False)

    ax.set_xlim(-5, 108)
    ax.set_ylim(-5, 108)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}%"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}%"))

    _style_ax(ax,
              f"D vs U Coverage %  (n={len(paired)} folders)\n"
              f"(jitter added; quantised 6.25% steps)",
              "D Coverage %", "U Coverage %")
    _scatter_legend(ax)


def _scatter_legend(ax):
    handles = [
        mpatches.Patch(facecolor=COL_PASS, label="Both D & U pass"),
        mpatches.Patch(facecolor=COL_D,    label="D only passes"),
        mpatches.Patch(facecolor=COL_U,    label="U only passes"),
        mpatches.Patch(facecolor=COL_FAIL, label="Neither passes"),
    ]
    ax.legend(handles=handles, fontsize=7.5, facecolor=COL_AX,
              edgecolor=COL_GRID, labelcolor=COL_TEXT, loc="upper left")


# =============================================================================
# Row 4 — Summary bar chart
# =============================================================================

def plot_summary_bar(ax, paired, single_count):
    both  = sum(1 for p in paired
                if (p["d_inliers"] >= THRESH_INLIERS and p["d_cov"] >= THRESH_COVERAGE)
                and (p["u_inliers"] >= THRESH_INLIERS and p["u_cov"] >= THRESH_COVERAGE))
    d_only = sum(1 for p in paired
                 if (p["d_inliers"] >= THRESH_INLIERS and p["d_cov"] >= THRESH_COVERAGE)
                 and not (p["u_inliers"] >= THRESH_INLIERS and p["u_cov"] >= THRESH_COVERAGE))
    u_only = sum(1 for p in paired
                 if not (p["d_inliers"] >= THRESH_INLIERS and p["d_cov"] >= THRESH_COVERAGE)
                 and (p["u_inliers"] >= THRESH_INLIERS and p["u_cov"] >= THRESH_COVERAGE))
    neither = sum(1 for p in paired
                  if not (p["d_inliers"] >= THRESH_INLIERS and p["d_cov"] >= THRESH_COVERAGE)
                  and not (p["u_inliers"] >= THRESH_INLIERS and p["u_cov"] >= THRESH_COVERAGE))

    labels = [
        "Both D & U pass",
        "D only passes",
        "U only passes",
        "Neither (both pairs OK)",
        "Single pair available",
    ]
    values = [both, d_only, u_only, neither, single_count]
    colours = [COL_PASS, COL_D, COL_U, COL_FAIL, "#9E9E9E"]
    total = sum(values)

    bars = ax.barh(labels[::-1], values[::-1], color=colours[::-1], alpha=0.85, zorder=3)
    for bar, val in zip(bars, values[::-1]):
        if val > 0:
            pct = val / total * 100 if total else 0
            ax.text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val}  ({pct:.0f}%)",
                    ha="left", va="center", color=COL_TEXT, fontsize=9)

    ax.set_xlim(0, max(values) * 1.3 if values else 10)
    ax.axvline(0, color=COL_GRID, linewidth=0.5)
    _style_ax(ax,
              f"Folder Outcome Summary at Current Thresholds"
              f"  (inliers >= {THRESH_INLIERS} AND coverage >= {THRESH_COVERAGE:.0f}%)",
              "Count", "", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)

    # Note about current rule
    ax.text(0.98, 0.03,
            "Current rule: folder PASSES if at least one pair (D or U) satisfies BOTH thresholds",
            transform=ax.transAxes, ha="right", va="bottom",
            color=COL_THRESH, fontsize=8, style="italic")

    return dict(both=both, d_only=d_only, u_only=u_only,
                neither=neither, single=single_count, total=total)


# =============================================================================
# Console summary
# =============================================================================

def print_summary(d_pairs, u_pairs, paired, single_count):
    print("\n--- Stage 2N Threshold Analysis ---")
    print(f"D pairs (status=OK): {len(d_pairs)}")
    print(f"U pairs (status=OK): {len(u_pairs)}")
    print(f"Folders with BOTH D & U OK: {len(paired)}")
    print(f"Folders with only ONE pair OK: {single_count}")

    for label, pairs in [("D", d_pairs), ("U", u_pairs)]:
        inliers = [p["ransac_inliers"] for p in pairs]
        covs    = [p["inlier_coverage_pct"] for p in pairs]
        n_i_pass = sum(1 for v in inliers if v >= THRESH_INLIERS)
        n_c_pass = sum(1 for v in covs if v >= THRESH_COVERAGE)
        n_both   = sum(1 for p in pairs
                       if p["ransac_inliers"] >= THRESH_INLIERS
                       and p["inlier_coverage_pct"] >= THRESH_COVERAGE)
        print(f"\n{label} pairs:")
        print(f"  Inliers  >= {THRESH_INLIERS}: {n_i_pass}/{len(inliers)} "
              f"({n_i_pass/len(inliers)*100:.1f}%)  "
              f"median={np.median(inliers):.0f}  max={max(inliers)}")
        print(f"  Coverage >= {THRESH_COVERAGE}%: {n_c_pass}/{len(covs)} "
              f"({n_c_pass/len(covs)*100:.1f}%)  "
              f"median={np.median(covs):.1f}%  max={max(covs):.1f}%")
        print(f"  BOTH thresholds: {n_both}/{len(pairs)} ({n_both/len(pairs)*100:.1f}%)")

    # Threshold sweep for paired folders
    print(f"\nThreshold sweep over {len(paired)} folders with both D & U OK:")
    print(f"  {'Inliers':>8}  {'Coverage':>10}  {'Both':>6}  {'D-only':>7}  "
          f"{'U-only':>7}  {'Neither':>8}  {'Any-pass':>9}")
    for t_i in [5, 10, 15, 20]:
        for t_c in [12.5, 18.75, 25.0, 31.25]:
            both   = sum(1 for p in paired
                         if p["d_inliers"] >= t_i and p["d_cov"] >= t_c
                         and p["u_inliers"] >= t_i and p["u_cov"] >= t_c)
            d_only = sum(1 for p in paired
                         if p["d_inliers"] >= t_i and p["d_cov"] >= t_c
                         and not (p["u_inliers"] >= t_i and p["u_cov"] >= t_c))
            u_only = sum(1 for p in paired
                         if not (p["d_inliers"] >= t_i and p["d_cov"] >= t_c)
                         and p["u_inliers"] >= t_i and p["u_cov"] >= t_c)
            neither = len(paired) - both - d_only - u_only
            any_pass = both + d_only + u_only
            marker = " <-- current" if t_i == THRESH_INLIERS and t_c == THRESH_COVERAGE else ""
            print(f"  {t_i:>8}  {t_c:>9.2f}%  {both:>6}  {d_only:>7}  "
                  f"{u_only:>7}  {neither:>8}  {any_pass:>9}{marker}")


# =============================================================================
# Main
# =============================================================================

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON
    print(f"Reading: {json_path}")

    d_pairs, u_pairs, paired, single_count = load_pairs(json_path)

    print_summary(d_pairs, u_pairs, paired, single_count)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 22), facecolor=COL_BG)
    gs  = gridspec.GridSpec(
        4, 2, figure=fig,
        height_ratios=[2.5, 2.5, 2.5, 1.8],
        hspace=0.55, wspace=0.35,
        top=0.955, bottom=0.04, left=0.07, right=0.97,
    )

    fig.text(
        0.5, 0.985,
        "Stage 2N Threshold Analysis — RANSAC Inliers & Inlier Coverage\n"
        f"Thresholds: inliers >= {THRESH_INLIERS}  |  coverage >= {THRESH_COVERAGE:.0f}%  "
        f"|  Rule: folder passes if at least one pair satisfies BOTH",
        ha="center", va="top", color=COL_TEXT, fontsize=13, fontweight="bold",
        transform=fig.transFigure,
    )

    # Row 0 — RANSAC Inliers
    plot_inliers(fig.add_subplot(gs[0, 0]), d_pairs, "D")
    plot_inliers(fig.add_subplot(gs[0, 1]), u_pairs, "U")

    # Row 1 — Coverage %
    plot_coverage(fig.add_subplot(gs[1, 0]), d_pairs, "D")
    plot_coverage(fig.add_subplot(gs[1, 1]), u_pairs, "U")

    # Row 2 — Scatter (paired folders)
    plot_scatter_inliers (fig.add_subplot(gs[2, 0]), paired)
    plot_scatter_coverage(fig.add_subplot(gs[2, 1]), paired)

    # Row 3 — Summary bar (full width)
    ax_sum = fig.add_subplot(gs[3, :])
    counts = plot_summary_bar(ax_sum, paired, single_count)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight",
                facecolor=COL_BG, edgecolor="none")
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
