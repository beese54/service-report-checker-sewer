"""
verification/optimise_thresholds.py
=====================================
Threshold optimisation for the WRN pipeline.

Reads ground_truth.json (human-verified labels) plus the raw stage JSON
outputs, sweeps each tuneable threshold, and finds the value that maximises
precision while keeping recall >= --min-recall (default 0.80).

Output
------
  verification/threshold_sweep_<timestamp>.png   — P/R/F1 subplot per threshold
  verification/recommended_thresholds.json       — {param: recommended_value}

Usage
-----
  python verification/optimise_thresholds.py
  python verification/optimise_thresholds.py --min-recall 0.75
"""

import argparse
import json
import os
import sys
from datetime import datetime
from glob import glob

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_HERE)
GT_PATH     = os.path.join(_HERE, "ground_truth.json")
ANALYSIS_DIR = os.path.join(_ROOT, "adjusted_images", "no_obstruction", "difference_analysis")
LEVEL2_JSON = os.path.join(ANALYSIS_DIR, "D2D5U2U5_check", "level2_check.json")

# ── Current (baseline) threshold values ───────────────────────────────────────
CURRENT = {
    # Stage 3N
    "HIGH_CONFIDENCE_THRESHOLD":   0.50,
    "MEDIUM_CONFIDENCE_THRESHOLD": 0.30,
    # Stage 4N gate
    "BLUR_REJECT_THRESHOLD":       35.0,
    "SSIM_MIN_THRESHOLD":          0.05,
    "INLIER_RATIO_MIN":            0.05,
    # Stage 4N scoring
    "ENTROPY_CONFIRM_DELTA":       0.30,
    "WATER_DETECT_THRESHOLD":      3.0,
    "GREASE_FLAG_THRESHOLD":       2.0,
    "PASS_SCORE_MIN":              3,
}

# ── Sweep ranges ──────────────────────────────────────────────────────────────
def _frange(start, stop, step):
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals

SWEEP_RANGES = {
    "HIGH_CONFIDENCE_THRESHOLD":   _frange(0.40, 0.85, 0.025),
    "MEDIUM_CONFIDENCE_THRESHOLD": _frange(0.20, 0.55, 0.025),
    "BLUR_REJECT_THRESHOLD":       _frange(15,   80,   5),
    "SSIM_MIN_THRESHOLD":          _frange(0.02, 0.20, 0.01),
    "INLIER_RATIO_MIN":            _frange(0.02, 0.15, 0.01),
    "ENTROPY_CONFIRM_DELTA":       _frange(0.10, 0.60, 0.05),
    "WATER_DETECT_THRESHOLD":      _frange(1.0,  8.0,  0.5),
    "GREASE_FLAG_THRESHOLD":       _frange(0.5,  5.0,  0.25),
    "PASS_SCORE_MIN":              [2, 3, 4],
}

MIN_LABELS_WARN = 30


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth() -> dict:
    if not os.path.isfile(GT_PATH):
        print(f"NOTE: ground_truth.json not found at {GT_PATH}")
        print("Use the /verify UI to label folders, then re-run this script.")
        return {"labelled": {}, "skipped": [], "meta": {}}
    with open(GT_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_latest_gf_run() -> dict:
    """Load most recent gf_run_*.json; returns {folder_id: {prefix: entry}}."""
    files = sorted(glob(os.path.join(ANALYSIS_DIR, "gf_run_*.json")))
    if not files:
        return {}
    with open(files[-1], encoding="utf-8") as f:
        raw = json.load(f)
    # Re-index by (folder, prefix)
    result = {}
    for entry in raw.get("results", []):
        fid    = entry.get("folder")
        prefix = entry.get("prefix")
        if fid and prefix:
            result.setdefault(fid, {})[prefix] = entry
    return result


def load_level2() -> dict:
    """Load level2_check.json; returns {folder_id: folder_data}."""
    if not os.path.isfile(LEVEL2_JSON):
        return {}
    with open(LEVEL2_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("folders", {})


# ─────────────────────────────────────────────────────────────────────────────
# Re-classification functions (parameterised by threshold values)
# ─────────────────────────────────────────────────────────────────────────────

def classify_stage3n(folder_id: str, l2: dict, thresholds: dict) -> str:
    """Return 'PASS' or 'FAIL' for Stage 3N given a threshold set."""
    high = thresholds.get("HIGH_CONFIDENCE_THRESHOLD", CURRENT["HIGH_CONFIDENCE_THRESHOLD"])
    folder = l2.get(folder_id, {})
    pairs  = folder.get("pairs", {})
    if not pairs:
        return "UNKNOWN"
    for p in pairs.values():
        if p.get("status") == "OK":
            if (p.get("washing_confidence") or 0.0) >= high:
                return "PASS"
    return "FAIL"


def classify_stage4n_pair(entry: dict, thresholds: dict) -> str:
    """Re-derive Stage 4N pair verdict given a threshold set.

    Returns 'PASS' | 'REVIEW' | 'FAIL' | 'GATE_REJECTED'.
    """
    gate  = entry.get("gate", {})
    sift  = entry.get("sift_stats", {})
    tex   = entry.get("texture", {})
    water = entry.get("water", {})
    grease = entry.get("grease", {})

    blur_thr      = thresholds.get("BLUR_REJECT_THRESHOLD",  CURRENT["BLUR_REJECT_THRESHOLD"])
    ssim_thr      = thresholds.get("SSIM_MIN_THRESHOLD",     CURRENT["SSIM_MIN_THRESHOLD"])
    inlier_thr    = thresholds.get("INLIER_RATIO_MIN",       CURRENT["INLIER_RATIO_MIN"])
    entropy_thr   = thresholds.get("ENTROPY_CONFIRM_DELTA",  CURRENT["ENTROPY_CONFIRM_DELTA"])
    water_thr     = thresholds.get("WATER_DETECT_THRESHOLD", CURRENT["WATER_DETECT_THRESHOLD"])
    grease_thr    = thresholds.get("GREASE_FLAG_THRESHOLD",  CURRENT["GREASE_FLAG_THRESHOLD"])
    pass_score    = thresholds.get("PASS_SCORE_MIN",         CURRENT["PASS_SCORE_MIN"])

    blur_score   = gate.get("blur_score", 0.0) or 0.0
    ssim_score   = sift.get("ssim_score", 1.0) or 1.0
    inlier_ratio = sift.get("inlier_ratio", 0.0) or 0.0
    ransac       = sift.get("ransac_inliers", 0) or 0
    pct_changed  = sift.get("pct_changed", 0.0) or 0.0
    entropy_d    = tex.get("entropy_delta", 0.0) or 0.0
    spec_pct     = water.get("specular_pct", 0.0) or 0.0
    blue_pct     = water.get("blue_water_pct", 0.0) or 0.0
    combined_pct = spec_pct + blue_pct
    grease_pct   = grease.get("grease_pct", 0.0) or 0.0

    # Gate checks (same logic as script)
    if blur_score < blur_thr:
        return "GATE_REJECTED"
    if ssim_score < ssim_thr:
        return "GATE_REJECTED"
    if inlier_ratio < inlier_thr:
        return "GATE_REJECTED"

    # Five scoring signals
    s1 = ransac      >= 10
    s2 = pct_changed >= 5.0
    s3 = entropy_d   >= entropy_thr
    s4 = combined_pct > water_thr
    s5 = grease_pct  <= grease_thr

    score = sum([s1, s2, s3, s4, s5])

    if score >= pass_score:
        verdict = "PASS"
    elif score >= 2:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"

    # Hard overrides (same as _derive_verdict)
    if grease_pct > 15.0 and not s4:
        verdict = "FAIL"
    if ransac < 4 and verdict == "PASS":
        verdict = "REVIEW"

    return verdict


def classify_stage4n_folder(folder_id: str, gf: dict, thresholds: dict) -> str:
    """Return folder-level Stage 4N verdict: 'PASS' | 'REVIEW' | 'FAIL' | 'UNKNOWN'."""
    pairs = gf.get(folder_id, {})
    if not pairs:
        return "UNKNOWN"
    pair_verdicts = [classify_stage4n_pair(entry, thresholds) for entry in pairs.values()]
    if any(v == "PASS" for v in pair_verdicts):
        return "PASS"
    if any(v == "REVIEW" for v in pair_verdicts):
        return "REVIEW"
    return "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(labels: dict, classifier_fn) -> dict:
    """
    labels        : {folder_id: {"true_verdict": "PASS"|"FAIL", ...}}
    classifier_fn : folder_id -> "PASS"|"REVIEW"|"FAIL"|"UNKNOWN"

    Returns {tp, fp, fn, tn, precision, recall, f1}.
    Pipeline PASS (or REVIEW if accepted) = predicted positive.
    True PASS = actually positive.
    """
    tp = fp = fn = tn = 0
    for fid, entry in labels.items():
        true_pos  = entry.get("true_verdict") == "PASS"
        pred      = classifier_fn(fid)
        pred_pos  = pred in ("PASS", "REVIEW")

        if true_pos  and pred_pos:  tp += 1
        elif not true_pos and pred_pos:  fp += 1
        elif true_pos  and not pred_pos: fn += 1
        else:                            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else float("nan"))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# Single-threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_threshold(param_name: str, labels: dict, gf: dict, l2: dict,
                    min_recall: float) -> dict:
    """
    Sweep *param_name* over its range while keeping all other thresholds
    at CURRENT values.

    Returns:
        {
            "values": [...],
            "precision": [...],
            "recall": [...],
            "f1": [...],
            "current_value": float,
            "recommended_value": float | None,
        }
    """
    values = SWEEP_RANGES[param_name]
    prec_list, rec_list, f1_list = [], [], []

    stage3n_params = {"HIGH_CONFIDENCE_THRESHOLD", "MEDIUM_CONFIDENCE_THRESHOLD"}

    for v in values:
        thresholds = dict(CURRENT)
        thresholds[param_name] = v

        if param_name in stage3n_params:
            def clf(fid, _t=thresholds):
                return classify_stage3n(fid, l2, _t)
        else:
            def clf(fid, _t=thresholds):
                return classify_stage4n_folder(fid, gf, _t)

        m = compute_metrics(labels, clf)
        prec_list.append(m["precision"])
        rec_list.append(m["recall"])
        f1_list.append(m["f1"])

    # Find recommended: highest precision where recall >= min_recall
    recommended = None
    best_prec   = -1.0
    for v, p, r in zip(values, prec_list, rec_list):
        if not (isinstance(r, float) and r == r):  # nan check
            continue
        if r >= min_recall and p > best_prec:
            best_prec   = p
            recommended = v

    return {
        "values":            values,
        "precision":         prec_list,
        "recall":            rec_list,
        "f1":                f1_list,
        "current_value":     CURRENT[param_name],
        "recommended_value": recommended,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WRN pipeline threshold optimisation (precision-first)"
    )
    parser.add_argument(
        "--min-recall", type=float, default=0.80,
        help="Minimum recall floor (default 0.80). "
             "Recommended threshold maximises precision at this recall."
    )
    args = parser.parse_args()
    min_recall = args.min_recall

    # ── Load data ─────────────────────────────────────────────────────────────
    gt = load_ground_truth()
    labelled = gt.get("labelled", {})

    # Filter to folders with a true_verdict
    labels = {
        fid: entry for fid, entry in labelled.items()
        if entry.get("true_verdict") in ("PASS", "FAIL")
    }

    n_labels = len(labels)
    print(f"Ground truth labels loaded: {n_labels}")

    if n_labels < MIN_LABELS_WARN:
        print(
            f"\nWARNING: Only {n_labels} labels available "
            f"(recommended >= {MIN_LABELS_WARN} for meaningful sweep)."
        )
        if n_labels == 0:
            print("No labels found. Label some folders via the /verify UI first.")
            sys.exit(0)

    gf = load_latest_gf_run()
    l2 = load_level2()

    # ── Sweep each threshold ──────────────────────────────────────────────────
    print(f"\nSweeping {len(SWEEP_RANGES)} thresholds (min_recall={min_recall})...\n")
    sweep_results = {}
    recommended   = {}

    for param in SWEEP_RANGES:
        result = sweep_threshold(param, labels, gf, l2, min_recall)
        sweep_results[param] = result
        rec_val = result["recommended_value"]
        recommended[param] = rec_val

        cur = result["current_value"]
        r_str = f"{rec_val}" if rec_val is not None else "n/a (recall always < floor)"
        change = ""
        if rec_val is not None and rec_val != cur:
            change = f"  <-- CHANGE from {cur}"
        print(f"  {param:<32} current={cur:<8} recommended={r_str}{change}")

    # ── Write recommended_thresholds.json ─────────────────────────────────────
    rec_path = os.path.join(_HERE, "recommended_thresholds.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(
            {"min_recall_floor": min_recall,
             "n_labels":         n_labels,
             "generated_at":     datetime.now().isoformat(timespec="seconds"),
             "thresholds":       recommended},
            f, indent=2,
        )
    print(f"\nRecommended thresholds written to: {rec_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping plot.")
        print("Install with: pip install matplotlib")
        return

    n_params = len(SWEEP_RANGES)
    cols     = 3
    rows     = (n_params + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    fig.patch.set_facecolor("#1a1a1a")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (param, result) in zip(axes_flat, sweep_results.items()):
        xs   = result["values"]
        prec = [p if p == p else 0 for p in result["precision"]]  # nan→0
        rec  = [r if r == r else 0 for r in result["recall"]]
        f1   = [f if f == f else 0 for f in result["f1"]]

        ax.set_facecolor("#121212")
        ax.plot(xs, prec, color="#60a5fa", linewidth=1.8, label="Precision")
        ax.plot(xs, rec,  color="#4ade80", linewidth=1.8, label="Recall")
        ax.plot(xs, f1,   color="#facc15", linewidth=1.8, linestyle="--", label="F1")

        cur = result["current_value"]
        rec_val = result["recommended_value"]

        ax.axvline(cur, color="#f97316", linewidth=1.2, linestyle=":", label=f"Current ({cur})")
        if rec_val is not None:
            ax.axvline(rec_val, color="#c084fc", linewidth=1.2, label=f"Rec. ({rec_val})")
        ax.axhline(min_recall, color="#9ca3af", linewidth=0.8, linestyle="--", alpha=0.6,
                   label=f"Recall floor ({min_recall})")

        ax.set_title(param, color="#e0e0e0", fontsize=9, pad=4)
        ax.set_xlabel("Threshold value", color="#9ca3af", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(colors="#9ca3af", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(fontsize=6, facecolor="#1a1a1a", edgecolor="#333",
                  labelcolor="#e0e0e0", loc="best")
        ax.grid(True, color="#2a2a2a", linewidth=0.5)

    # Hide empty subplots
    for ax in axes_flat[n_params:]:
        ax.set_visible(False)

    fig.suptitle(
        f"WRN Pipeline Threshold Sweep  |  n={n_labels} labels  |  "
        f"min_recall={min_recall}",
        color="#e0e0e0", fontsize=12,
    )
    plt.tight_layout()

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(_HERE, f"threshold_sweep_{ts}.png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()
    print(f"Sweep plot saved to: {png_path}")


if __name__ == "__main__":
    main()
