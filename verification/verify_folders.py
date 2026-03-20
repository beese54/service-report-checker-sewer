"""
verification/verify_folders.py
================================
Backend helpers for the human verification UI.

Functions:
    get_all_folders()           List all no-obstruction folder IDs in sorted order.
    get_ground_truth()          Load ground_truth.json (creates if absent).
    save_ground_truth(gt)       Atomically write ground_truth.json.
    get_next_folder(gt)         Return first folder not yet labelled or skipped.
    save_label(folder_id, data) Append one label entry and persist.
    get_progress()              Return {labelled, skipped, total, breakdown}.
    build_folder_context(fid)   Collect all available pipeline metrics for fid.
"""

import json
import os
from datetime import datetime
from glob import glob

# ── Base paths (same convention as pipeline scripts) ─────────────────────────
_ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NO_OBS_DIR        = os.environ.get("NO_OBSTRUCTION_DIR",  os.path.join(_ROOT, "adjusted_images", "no_obstruction"))
ANALYSIS_DIR      = os.path.join(NO_OBS_DIR, "difference_analysis")
LEVEL2_JSON       = os.path.join(ANALYSIS_DIR, "D2D5U2U5_check", "level2_check.json")
ORIGINAL_DIR      = os.environ.get("ORIGINAL_IMAGES_DIR", os.path.join(_ROOT, "original_images"))
GROUND_TRUTH_PATH = os.environ.get("GROUND_TRUTH_PATH",   os.path.join(_ROOT, "verification", "ground_truth.json"))

_SKIP_NAMES = {"difference_analysis"}


# ─────────────────────────────────────────────────────────────────────────────
# Folder discovery
# ─────────────────────────────────────────────────────────────────────────────

def get_all_folders() -> list[str]:
    """Return sorted list of no-obstruction job folder IDs."""
    if not os.path.isdir(NO_OBS_DIR):
        return []
    return sorted(
        e.name for e in os.scandir(NO_OBS_DIR)
        if e.is_dir() and e.name not in _SKIP_NAMES
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth persistence
# ─────────────────────────────────────────────────────────────────────────────

def _empty_gt(total: int) -> dict:
    return {
        "labelled": {},
        "skipped": [],
        "meta": {"total_folders": total, "labelled": 0, "skipped": 0},
    }


def get_ground_truth() -> dict:
    """Load ground_truth.json; create with correct total if absent."""
    total = len(get_all_folders())
    if not os.path.isfile(GROUND_TRUTH_PATH):
        return _empty_gt(total)
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    gt.setdefault("labelled", {})
    gt.setdefault("skipped", [])
    gt.setdefault("meta", {})
    gt["meta"]["total_folders"] = total
    return gt


def save_ground_truth(gt: dict) -> None:
    os.makedirs(os.path.dirname(GROUND_TRUTH_PATH), exist_ok=True)
    tmp = GROUND_TRUTH_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    os.replace(tmp, GROUND_TRUTH_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────────────────────

def get_next_folder(gt: dict) -> str | None:
    """Return the first folder not yet labelled or skipped, or None if done."""
    done = set(gt.get("labelled", {}).keys()) | set(gt.get("skipped", []))
    for fid in get_all_folders():
        if fid not in done:
            return fid
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Label submission
# ─────────────────────────────────────────────────────────────────────────────

def save_label(folder_id: str, agreed: bool, true_verdict: str | None,
               failed_stage: str | None, notes: str) -> None:
    """Append one labelling decision to ground_truth.json."""
    gt = get_ground_truth()
    ctx = build_folder_context(folder_id)

    gt["labelled"][folder_id] = {
        "pipeline_verdict":  ctx.get("pipeline_verdict", "UNKNOWN"),
        "pipeline_score":    ctx.get("pipeline_score"),
        "agreed":            agreed,
        "true_verdict":      true_verdict,
        "failed_stage":      failed_stage,
        "notes":             notes,
        "labelled_at":       datetime.now().isoformat(timespec="seconds"),
    }
    gt["meta"]["labelled"] = len(gt["labelled"])
    gt["meta"]["skipped"]  = len(gt["skipped"])
    save_ground_truth(gt)


def save_skip(folder_id: str) -> None:
    gt = get_ground_truth()
    if folder_id not in gt["skipped"]:
        gt["skipped"].append(folder_id)
    gt["meta"]["skipped"] = len(gt["skipped"])
    save_ground_truth(gt)


# ─────────────────────────────────────────────────────────────────────────────
# Progress
# ─────────────────────────────────────────────────────────────────────────────

def get_progress() -> dict:
    gt    = get_ground_truth()
    total = len(get_all_folders())
    labelled_entries = gt.get("labelled", {})

    agreed    = sum(1 for v in labelled_entries.values() if v.get("agreed"))
    disagreed = sum(1 for v in labelled_entries.values() if not v.get("agreed"))

    return {
        "total":     total,
        "labelled":  len(labelled_entries),
        "skipped":   len(gt.get("skipped", [])),
        "remaining": total - len(labelled_entries) - len(gt.get("skipped", [])),
        "agreed":    agreed,
        "disagreed": disagreed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Folder context builder
# ─────────────────────────────────────────────────────────────────────────────

def _load_latest_gf_run() -> dict:
    """Load the most recent gf_run_*.json from ANALYSIS_DIR."""
    pattern = os.path.join(ANALYSIS_DIR, "gf_run_*.json")
    files   = sorted(glob(pattern))
    if not files:
        return {}
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


def _load_level2() -> dict:
    """Load level2_check.json (Stage 3N washing confidence per folder)."""
    if not os.path.isfile(LEVEL2_JSON):
        return {}
    with open(LEVEL2_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("folders", {})


def build_folder_context(folder_id: str) -> dict:
    """
    Collect all available pipeline metrics for *folder_id*.

    Returns a dict with the following structure (keys absent when data
    unavailable):

        {
            "folder_id":        str,
            "pipeline_verdict": "PASS"|"REVIEW"|"FAIL"|"UNKNOWN",
            "pipeline_score":   int | None,  # sum of pair scores from Stage 4N

            "stage4n": {
                "D3": { status, score, blur_score, ssim_score, grease_pct,
                        entropy_delta, water_detected, water_confidence,
                        ransac_inliers, inlier_ratio, reject_reasons },
                "U3": { ... },
            },

            "stage3n": {
                "D2": { washing_tier, washing_confidence, ransac_inliers, ... },
                "U2": { ... },
            },

            "images": {
                "originals": { "D3": url, "D6": url, "U3": url, "U6": url, ... },
                "pipeline": {
                    "D3_report": url, "U3_report": url,
                    "D_composite": url, "D2_composite": url, ... }
            },

            "existing_label": dict | None,
        }
    """
    gf    = _load_latest_gf_run()
    l2    = _load_level2()
    gt    = get_ground_truth()

    # ── Stage 4N: pair entries for this folder ────────────────────────────────
    gf_results = gf.get("results", [])
    s4n = {}
    score_total = 0
    for entry in gf_results:
        if entry.get("folder") != folder_id:
            continue
        prefix = entry.get("prefix", "?")
        score_total += entry.get("score", 0)
        gate   = entry.get("gate", {})
        sift   = entry.get("sift_stats", {})
        grease = entry.get("grease", {})
        texture = entry.get("texture", {})
        water  = entry.get("water", {})
        s4n[prefix] = {
            "status":           entry.get("status"),
            "score":            entry.get("score"),
            "blur_score":       gate.get("blur_score"),
            "circle_found":     gate.get("circle_found"),
            "reject_reasons":   gate.get("reject_reasons", []),
            "ransac_inliers":   sift.get("ransac_inliers"),
            "ssim_score":       sift.get("ssim_score"),
            "inlier_ratio":     sift.get("inlier_ratio"),
            "grease_pct":       grease.get("grease_pct"),
            "grease_flagged":   grease.get("flagged"),
            "entropy_delta":    texture.get("entropy_delta"),
            "texture_confirmed": texture.get("confirmed"),
            "water_detected":   water.get("water_detected"),
            "water_confidence": water.get("water_confidence"),
        }

    # Aggregate pipeline verdict from Stage 4N statuses
    statuses = [v.get("status") for v in s4n.values() if v.get("status")]
    if any(s == "PASS" for s in statuses):
        pipeline_verdict = "PASS"
    elif any(s == "REVIEW" for s in statuses):
        pipeline_verdict = "REVIEW"
    elif statuses:
        pipeline_verdict = "FAIL"
    else:
        pipeline_verdict = "UNKNOWN"

    # ── Stage 3N: per-pair data ───────────────────────────────────────────────
    folder_l2 = l2.get(folder_id, {})
    s3n = {}
    for prefix, pair_data in folder_l2.get("pairs", {}).items():
        s3n[prefix] = {
            "status":             pair_data.get("status"),
            "washing_tier":       pair_data.get("washing_tier"),
            "washing_confidence": pair_data.get("washing_confidence"),
            "ransac_inliers":     pair_data.get("ransac_inliers"),
            "kp_ratio":           pair_data.get("kp_ratio"),
            "entropy_increase":   pair_data.get("entropy_increase"),
        }

    # ── Image URL builder ─────────────────────────────────────────────────────
    # Routes /originals and /analysis are mounted in app/main.py
    def orig_url(base):
        # Check original_images first, then adjusted_images
        for root in [ORIGINAL_DIR, NO_OBS_DIR]:
            candidate = os.path.join(root, folder_id, f"{base}.jpg")
            if os.path.isfile(candidate):
                return f"/originals/{folder_id}/{base}.jpg"
        return None

    def analysis_url(filename):
        path = os.path.join(ANALYSIS_DIR, folder_id, filename)
        if os.path.isfile(path):
            return f"/analysis/{folder_id}/{filename}"
        return None

    originals = {}
    for base in ["D1", "D2", "D3", "D4", "D5", "D6", "U1", "U2", "U3", "U4", "U5", "U6"]:
        u = orig_url(base)
        if u:
            originals[base] = u

    pipeline_imgs = {}
    for fname in [
        f"{folder_id}_gf_report.jpg",
        f"{folder_id}_report.jpg",
        "D3_gf_report.jpg", "U3_gf_report.jpg",
        "D3_keypoint_matches.jpg", "U3_keypoint_matches.jpg",
        "D_composite.jpg", "U_composite.jpg",
        "D2_composite.jpg", "U2_composite.jpg",
    ]:
        u = analysis_url(fname)
        if u:
            pipeline_imgs[fname] = u

    return {
        "folder_id":        folder_id,
        "pipeline_verdict": pipeline_verdict,
        "pipeline_score":   score_total if s4n else None,
        "stage4n":          s4n,
        "stage3n":          s3n,
        "images": {
            "originals": originals,
            "pipeline":  pipeline_imgs,
        },
        "existing_label": gt.get("labelled", {}).get(folder_id),
    }
