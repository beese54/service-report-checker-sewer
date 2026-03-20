"""
Batch D2/D5 + U2/U5 Report Runner
==================================
Runs the D2/D5 and U2/U5 SIFT pipeline (from check_d2d5_u2u5.py) across
every job folder in adjusted_images/no_obstruction/, then:

  - Saves each report to difference_analysis/<folder>/<folder>_report.jpg
  - Copies each report into a tier subfolder inside D2D5U2U5_check/:
      D2D5U2U5_check/HIGH/    -- folder has >= 1 pair with HIGH washing confidence
      D2D5U2U5_check/MEDIUM/  -- best pair is MEDIUM (no HIGH pair)
      D2D5U2U5_check/LOW/     -- all pairs are LOW
      D2D5U2U5_check/FAILED/  -- no pairs could be processed
  - Writes level2_check.json to D2D5U2U5_check/

Usage:
    python batch_d2d5_u2u5.py
"""

import json
import os
import shutil
import sys

import cv2

from pipeline.stage3n_washing import PAIRS, build_report_image, process_pair

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\adjusted_images\no_obstruction"
ANALYSIS_DIR     = os.path.join(BASE_DIR, "difference_analysis")
CONSOLIDATED_DIR = os.path.join(ANALYSIS_DIR, "D2D5U2U5_check")

# Tier order used to pick the best tier across pairs
_TIER_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


# ─────────────────────────────────────────────────────────────────────────────
# Per-folder runner
# ─────────────────────────────────────────────────────────────────────────────

def run_folder(folder_name):
    """Run SIFT + washing-evidence pipeline for all PAIRS in one folder.

    Returns
    -------
    tuple  ``(report_path_or_None, pair_metrics_dict)``
        pair_metrics_dict maps prefix -> result dict for JSON output.
    """
    folder_path     = os.path.join(BASE_DIR, folder_name)
    out_dir         = os.path.join(ANALYSIS_DIR, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    pair_results    = []   # for building the report image
    pair_metrics    = {}   # for JSON output

    for base_before, base_after, prefix in PAIRS:
        print(f"    [{prefix}] {base_before}/{base_after} ...", end=" ", flush=True)
        try:
            stats, metrics, match_img, composite, visuals = process_pair(
                folder_path, folder_name, base_before, base_after, prefix, out_dir
            )
            pair_results.append((stats, metrics, match_img, composite, visuals))
            pair_metrics[prefix] = {
                "status":             "OK",
                "washing_tier":       metrics["washing_tier"],
                "washing_confidence": metrics["washing_confidence"],
                "ransac_inliers":     stats["ransac_inliers"],
                "kp_ratio":           metrics["kp_ratio"],
                "std_increase_pct":   metrics["std_increase_pct"],
                "entropy_increase":   metrics["entropy_increase"],
                "match_ratio":        metrics["match_ratio"],
                "edge_increase_pct":  metrics["edge_increase_pct"],
                "lap_increase_pct":   metrics["lap_increase_pct"],
            }
        except (FileNotFoundError, RuntimeError, IOError) as exc:
            print(f"SKIPPED -- {exc}")
            pair_metrics[prefix] = {"status": "FAILED", "error": str(exc)}

    if not pair_results:
        return None, pair_metrics

    report      = build_report_image(folder_name, pair_results)
    report_path = os.path.join(out_dir, f"{folder_name}_report.jpg")
    cv2.imwrite(report_path, report, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"    Report saved: {os.path.basename(report_path)}")
    return report_path, pair_metrics


def _folder_tier(pair_metrics):
    """Return the best washing tier achieved by any pair in this folder."""
    ok_tiers = [
        v["washing_tier"]
        for v in pair_metrics.values()
        if v.get("status") == "OK"
    ]
    if not ok_tiers:
        return "FAILED"
    return max(ok_tiers, key=lambda t: _TIER_RANK.get(t, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    folders = sorted(
        e.name
        for e in os.scandir(BASE_DIR)
        if e.is_dir() and e.name != "difference_analysis"
    )
    n = len(folders)
    print(f"Found {n} folder(s) to process.")
    print()

    # Create tier subfolders up front
    for tier in ("HIGH", "MEDIUM", "LOW", "FAILED"):
        os.makedirs(os.path.join(CONSOLIDATED_DIR, tier), exist_ok=True)

    all_results = {}
    ok      = 0
    skipped = 0

    for i, folder_name in enumerate(folders, start=1):
        print(f"[{i}/{n}] {folder_name}")
        try:
            report_path, pair_metrics = run_folder(folder_name)
            tier       = _folder_tier(pair_metrics)
            level2_pass = tier == "HIGH"

            all_results[folder_name] = {
                "level2_pass":  level2_pass,
                "folder_tier":  tier,
                "pairs":        pair_metrics,
            }

            if report_path:
                tier_dir = os.path.join(CONSOLIDATED_DIR, tier)
                shutil.copy2(report_path, tier_dir)
                ok += 1
            else:
                tier_dir = os.path.join(CONSOLIDATED_DIR, "FAILED")
                skipped += 1

        except Exception as exc:
            print(f"    ERROR -- {exc}")
            all_results[folder_name] = {
                "level2_pass": False,
                "folder_tier": "FAILED",
                "pairs":       {},
                "error":       str(exc),
            }
            skipped += 1
        print()

    # ── Write level2_check.json ───────────────────────────────────────────────
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "FAILED": 0}
    for v in all_results.values():
        counts[v["folder_tier"]] = counts.get(v["folder_tier"], 0) + 1

    report_data = {
        "level": 2,
        "check": "D2/D5 + U2/U5  Washing Confidence",
        "criteria": {
            "pass_tier":           "HIGH",
            "high_threshold":      "confidence >= 0.50",
            "medium_threshold":    "0.30 <= confidence < 0.50",
            "low_threshold":       "confidence < 0.30",
            "confidence_formula":  "mean of 6 signal scores (kp_ratio, std_dev, glcm_entropy, match_ratio, edge_density, laplacian_variance)",
            "folder_logic":        "folder passes if AT LEAST ONE pair (D2 or U2) achieves HIGH tier",
        },
        "summary": {
            "total_folders": n,
            "pass":          counts["HIGH"],
            "fail":          n - counts["HIGH"],
            "HIGH":          counts["HIGH"],
            "MEDIUM":        counts["MEDIUM"],
            "LOW":           counts["LOW"],
            "FAILED":        counts["FAILED"],
        },
        "folders": all_results,
    }

    json_path = os.path.join(CONSOLIDATED_DIR, "level2_check.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("--")
    print(f"Done.  {ok}/{n} reports generated,  {skipped} skipped.")
    print()
    print(f"  HIGH   (PASS)  : {counts['HIGH']:>4}")
    print(f"  MEDIUM (FAIL)  : {counts['MEDIUM']:>4}")
    print(f"  LOW    (FAIL)  : {counts['LOW']:>4}")
    print(f"  FAILED         : {counts['FAILED']:>4}")
    print()
    print(f"  Reports grouped in : {CONSOLIDATED_DIR}")
    print(f"  JSON report        : {json_path}")


if __name__ == "__main__":
    main()
