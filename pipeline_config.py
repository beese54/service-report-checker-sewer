"""
Pipeline configuration — centralised thresholds and env-var loading.

All pipeline scripts import from here so thresholds are changed in one place.
Loading order: config.json defaults → env var overrides → module-level constants.

To change a threshold without touching code, edit config.json at the project root
(or /app/config.json in the Docker container).  Env vars still take precedence over
config.json so env-var overrides continue to work as before.
"""

import json
import os


# ── config.json loader ────────────────────────────────────────────────────────
_CONFIG_JSON_PATH = os.environ.get(
    "CONFIG_JSON",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
)


def _load_config_json() -> dict:
    try:
        with open(_CONFIG_JSON_PATH, encoding="utf-8") as _f:
            return json.load(_f)
    except Exception:
        return {}


_CFG = _load_config_json()


def _cfg(section: str, key: str, default):
    """Read a value from config.json, falling back to default."""
    return _CFG.get(section, {}).get(key, default)


# ── Stage 2N — D1/D4 + U1/U4 SIFT thresholds ────────────────────────────────
# A pair PASSES (Level 1) if BOTH conditions are met:
#   1. RANSAC inliers >= STAGE2N_MIN_INLIERS  (confirms same physical location)
#   2. Inlier spatial coverage >= STAGE2N_MIN_COVERAGE_PCT
#      (confirms inliers are spread across the pipe surface, not one spot)
# A folder passes Level 1 if at least one pair (D or U) passes both.
STAGE2N_MIN_INLIERS:      int   = int(os.environ.get(
    "STAGE2N_MIN_INLIERS",     _cfg("stage2n", "min_inliers",     10)))
STAGE2N_MIN_COVERAGE_PCT: float = float(os.environ.get(
    "STAGE2N_MIN_COVERAGE_PCT", _cfg("stage2n", "min_coverage_pct", 25.0)))

# ── Stage 3N — D2/D5 + U2/U5 washing confidence ──────────────────────────────
# Only "HIGH" tier (washing_confidence >= STAGE3N_HIGH_CONFIDENCE) counts as PASS.
STAGE3N_PASS_TIER: str = os.environ.get("STAGE3N_PASS_TIER", "HIGH")
# Tier thresholds (lowered from 0.65/0.35 — analysis of 63 rejects showed 6 folders
# clustered just below 0.58; lowering HIGH to 0.50 rescues them without opening floodgates)
STAGE3N_HIGH_CONFIDENCE:   float = float(os.environ.get(
    "STAGE3N_HIGH_CONFIDENCE",   _cfg("stage3n", "high_confidence",   0.50)))
STAGE3N_MEDIUM_CONFIDENCE: float = float(os.environ.get(
    "STAGE3N_MEDIUM_CONFIDENCE", _cfg("stage3n", "medium_confidence", 0.30)))

# ── Stage 2N — fallback Lowe's ratio (more permissive than primary 0.75) ──────
RATIO_THRESHOLD_FALLBACK: float = float(os.environ.get(
    "RATIO_THRESHOLD_FALLBACK", _cfg("stage2n", "ratio_threshold_fallback", 0.80)))

# ── Stage 4N — D3/D6 + U3/U6 geometry-first verdict ─────────────────────────
# REVIEW goes to NEEDS_REVIEW bucket (not ACCEPTED, not REJECTED).
STAGE4N_REVIEW_IS_ACCEPTED: bool = os.environ.get(
    "STAGE4N_REVIEW_IS_ACCEPTED", "false"
).lower() in ("1", "true", "yes")

# ── Stage 4N — tunable thresholds ────────────────────────────────────────────
BLUR_REJECT_THRESHOLD:  float = float(os.environ.get(
    "BLUR_REJECT_THRESHOLD",  _cfg("stage4n", "blur_reject_threshold",  35.0)))
GREASE_FLAG_THRESHOLD:  float = float(os.environ.get(
    "GREASE_FLAG_THRESHOLD",  _cfg("stage4n", "grease_flag_threshold",  2.0)))
ENTROPY_CONFIRM_DELTA:  float = float(os.environ.get(
    "ENTROPY_CONFIRM_DELTA",  _cfg("stage4n", "entropy_confirm_delta",  0.3)))
WATER_DETECT_THRESHOLD: float = float(os.environ.get(
    "WATER_DETECT_THRESHOLD", _cfg("stage4n", "water_detect_threshold", 3.0)))
SSIM_MIN_THRESHOLD:     float = float(os.environ.get(
    "SSIM_MIN_THRESHOLD",     _cfg("stage4n", "ssim_min_threshold",     0.05)))
HOMOGRAPHY_DET_MIN:     float = float(os.environ.get(
    "HOMOGRAPHY_DET_MIN",     _cfg("stage4n", "homography_det_min",     0.2)))
HOMOGRAPHY_DET_MAX:     float = float(os.environ.get(
    "HOMOGRAPHY_DET_MAX",     _cfg("stage4n", "homography_det_max",     5.0)))
INLIER_RATIO_MIN:       float = float(os.environ.get(
    "INLIER_RATIO_MIN",       _cfg("stage4n", "inlier_ratio_min",       0.05)))

# ── Obstruction / OCR ─────────────────────────────────────────────────────────
# Folder is ACCEPTED if at least one image is successfully OCR-processed.
OBSTRUCTION_ACCEPTED_IF_ANY_OCR: bool = os.environ.get(
    "OBSTRUCTION_ACCEPTED_IF_ANY_OCR", "true"
).lower() in ("1", "true", "yes")

# ── Detector (YOLOv8) ─────────────────────────────────────────────────────────
DETECTOR_CONF_ACCEPT: float = float(os.environ.get(
    "DETECTOR_CONF_ACCEPT", _cfg("detector", "conf_accept", 0.70)))
DETECTOR_CONF_REVIEW: float = float(os.environ.get(
    "DETECTOR_CONF_REVIEW", _cfg("detector", "conf_review", 0.30)))

# ── SIFT pipeline shared constants ───────────────────────────────────────────
FLANN_TREES:      int   = 5
FLANN_CHECKS:     int   = 50
RATIO_THRESHOLD:  float = _cfg("stage2n", "ratio_threshold", 0.75)
RANSAC_THRESHOLD: float = _cfg("stage2n", "ransac_threshold", 5.0)
DIFF_THRESHOLD:   int   = 20     # grey-level delta considered "significant" (0-255)

# ── Metadata masking for SIFT ─────────────────────────────────────────────────
# OCR-derived text regions are excluded from SIFT keypoint extraction.
# SIFT_MIN_MASKED_KP: minimum keypoints in physical regions after masking;
#   fewer than this raises "Insufficient Physical Landmarks".
# TEXT_MASK_PAD: pixel margin added around each OCR bounding box.
SIFT_MIN_MASKED_KP: int = int(os.environ.get(
    "SIFT_MIN_MASKED_KP", _cfg("stage2n", "sift_min_masked_kp", 10)))
TEXT_MASK_PAD:      int = int(os.environ.get("TEXT_MASK_PAD", 8))

# ── Output directory names ────────────────────────────────────────────────────
# Within output_dir passed to each stage runner.
NO_OBS_SUBDIR   = "no_obstruction"
OBS_SUBDIR      = "obstruction"
ANALYSIS_SUBDIR = "difference_analysis"
OCR_SUBDIR      = "extracted_text"
OBS_SIFT_SUBDIR = "difference_analysis"
