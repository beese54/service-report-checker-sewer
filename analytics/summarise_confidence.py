"""
D2/D5 + U2/U5 Washing Confidence Summary
==========================================
Re-runs SIFT + metrics only (no image saving) across all job folders and
produces:
  - Console summary grouped by confidence tier (HIGH / MEDIUM / LOW)
  - difference_analysis/D2D5U2U5_confidence_summary.json

Folder-level tier = average washing_confidence across whichever pairs
succeeded.  Thresholds match check_d2d5_u2u5.py:
  HIGH   >= 0.50
  MEDIUM >= 0.30
  LOW    <  0.30

Usage:
    python summarise_confidence.py
"""

import json
import os
import sys

import cv2

from pipeline.stage3n_washing import (
    PAIRS,
    compute_homography,
    compute_washing_metrics,
    load_pair,
    sift_match,
    warp_and_diff,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\adjusted_images\no_obstruction"
ANALYSIS_DIR = os.path.join(BASE_DIR, "difference_analysis")
SUMMARY_JSON = os.path.join(ANALYSIS_DIR, "D2D5U2U5_confidence_summary.json")

TIER_THRESHOLDS = {"HIGH": 0.50, "MEDIUM": 0.30}


def folder_tier(confidence):
    if confidence >= TIER_THRESHOLDS["HIGH"]:
        return "HIGH"
    if confidence >= TIER_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Per-folder analysis (metrics only, no image I/O)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_folder(folder_name):
    """Return list of per-pair metric dicts. Empty list if all pairs skip."""
    folder_path = os.path.join(BASE_DIR, folder_name)
    pair_results = []

    for base_before, base_after, prefix in PAIRS:
        try:
            before_bgr, after_bgr = load_pair(folder_path, base_before, base_after)
            gray_b = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
            gray_a = cv2.cvtColor(after_bgr,  cv2.COLOR_BGR2GRAY)

            kp1, kp2, good_matches = sift_match(gray_b, gray_a)
            H, inlier_matches      = compute_homography(kp1, kp2, good_matches)
            _, _, mean_diff, pct_changed = warp_and_diff(before_bgr, after_bgr, H)

            metrics = compute_washing_metrics(gray_b, gray_a, kp1, kp2, good_matches)
            pair_results.append({
                "pair":           f"{base_before}/{base_after}",
                "prefix":         prefix,
                "kp_before":      len(kp1),
                "kp_after":       len(kp2),
                "ratio_matches":  len(good_matches),
                "ransac_inliers": len(inlier_matches),
                "mean_diff":      round(mean_diff, 1),
                "pct_changed":    round(pct_changed, 1),
                **metrics,
            })
        except (FileNotFoundError, RuntimeError, IOError):
            pass  # missing images or insufficient matches — silently skip

    return pair_results


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
    print(f"Analysing {n} folder(s) ...")
    print()

    records = {}   # folder_name -> {tier, confidence, pairs: [...]}

    for i, folder_name in enumerate(folders, start=1):
        print(f"  [{i:>3}/{n}] {folder_name}", end=" ", flush=True)
        try:
            pair_results = analyse_folder(folder_name)
        except Exception as exc:
            print(f"ERROR -- {exc}")
            records[folder_name] = {"tier": "ERROR", "confidence": None, "pairs": []}
            continue

        if not pair_results:
            print("no pairs processed")
            records[folder_name] = {"tier": "NO_DATA", "confidence": None, "pairs": []}
            continue

        avg_conf = round(sum(p["washing_confidence"] for p in pair_results) / len(pair_results), 3)
        tier     = folder_tier(avg_conf)
        records[folder_name] = {
            "tier":       tier,
            "confidence": avg_conf,
            "pairs":      pair_results,
        }
        print(f"conf={avg_conf:.3f}  [{tier}]")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    print()
    print(f"JSON saved: {SUMMARY_JSON}")
    print()

    # ── Group and print summary ────────────────────────────────────────────────
    high   = [(f, r["confidence"]) for f, r in records.items() if r["tier"] == "HIGH"]
    medium = [(f, r["confidence"]) for f, r in records.items() if r["tier"] == "MEDIUM"]
    low    = [(f, r["confidence"]) for f, r in records.items() if r["tier"] == "LOW"]
    other  = [(f, r["tier"])       for f, r in records.items() if r["tier"] not in ("HIGH","MEDIUM","LOW")]

    def print_group(label, items, show_conf=True):
        print(f"{label}  ({len(items)} folders)")
        print("-" * 52)
        if not items:
            print("  (none)")
        else:
            for folder, val in sorted(items, key=lambda x: x[0]):
                if show_conf:
                    print(f"  {folder:<20}  conf={val:.3f}")
                else:
                    print(f"  {folder:<20}  [{val}]")
        print()

    print("=" * 52)
    print("WASHING CONFIDENCE SUMMARY")
    print("=" * 52)
    print()
    print_group("HIGH   (>= 0.50)", high)
    print_group("MEDIUM (0.30 - 0.49)", medium)
    print_group("LOW    (< 0.30)", low)
    if other:
        print_group("OTHER (no data / error)", other, show_conf=False)

    print("=" * 52)
    print(f"  HIGH   : {len(high):>4}")
    print(f"  MEDIUM : {len(medium):>4}")
    print(f"  LOW    : {len(low):>4}")
    if other:
        print(f"  OTHER  : {len(other):>4}")
    print(f"  TOTAL  : {n:>4}")
    print("=" * 52)


if __name__ == "__main__":
    main()
