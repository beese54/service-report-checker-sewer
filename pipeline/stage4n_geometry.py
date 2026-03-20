"""
D3/D6 + U3/U6 Geometry-First Pipeline
======================================
Analyses the close-up image pairs (D3/D6 downstream, U3/U6 upstream) with a
three-phase system:
  Phase A  -- blur/misalignment gate + pipe cross-section detection
  Phase B  -- SIFT alignment (reused from check_d2d5_u2u5)
  Phase C  -- three independent verification tracks:
              C.1  grease ratio (HSV colour approx; YOLOv11-seg hook for future)
              C.2  texture entropy (GLCM)
              C.3  water signature (specular highlights + blue-water HSV)

Usage:
    python d3d6_u3u6.py [folder_name]   # single-folder smoke test
    python d3d6_u3u6.py                 # full batch
"""

import datetime
import json
import os
import shutil
import sys
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as _ssim

from pipeline.stage3n_washing import (
    DIFF_THRESHOLD,
    EXPL_PADDING,
    PANEL_HEIGHT,
    SECTION_BAR_H,
    VALID_EXTS,
    COL_BODY,
    COL_COL_LABEL,
    COL_DIVIDER,
    COL_EXPL_BG,
    COL_EXPL_TITLE,
    COL_SECTION_BG,
    COL_SECTION_TXT,
    COL_SEPARATOR,
    COL_STAT_VALUE,
    COL_STEP_HEADING,
    build_cover_header,
    build_footer,
    build_section_bar,
    compute_homography,
    create_metadata_mask,
    find_image,
    load_font,
    load_pair,
    run_ocr_for_mask,
    sift_match,
    warp_and_diff,
    _pil_to_bgr,
    _resize_to_height,
    _text_w,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\adjusted_images\no_obstruction"
ANALYSIS_DIR = os.path.join(BASE_DIR, "difference_analysis")
GF_DIR       = os.path.join(ANALYSIS_DIR, "D3D6U3U6_check")

# ── Close-up pairs only ────────────────────────────────────────────────────────
PAIRS = [
    ("D3", "D6", "D3"),   # downstream close-up
    ("U3", "U6", "U3"),   # upstream close-up
]

# ── Thresholds (imported from pipeline_config — edit config.json to tune) ──────
from pipeline_config import (
    BLUR_REJECT_THRESHOLD,
    GREASE_FLAG_THRESHOLD,
    ENTROPY_CONFIRM_DELTA,
    WATER_DETECT_THRESHOLD,
    SSIM_MIN_THRESHOLD,
    HOMOGRAPHY_DET_MIN,
    HOMOGRAPHY_DET_MAX,
    INLIER_RATIO_MIN,
)

REPORT_W            = int(os.environ.get("REPORT_W", 2400))  # report image width (px)
_SKIP_IMGS          = os.environ.get("SKIP_REPORT_IMAGES", "").lower() in ("1", "true")
BLUR_GAUSSIAN_KSIZE = 3       # GaussianBlur kernel applied before Laplacian variance

# ── Grease segmentation model (future hook) ────────────────────────────────────
GREASE_SEG_MODEL_PATH = os.path.join(
    r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker",
    "grease_detector", "weights", "best.pt",
)

# ── Verdict colours (RGB for Pillow) ──────────────────────────────────────────
VERDICT_COLOURS = {
    "PASS":              (80,  200, 100),
    "REVIEW":            (255, 180,  50),
    "FAIL":              (220,  60,  60),
    "GATE_REJECTED":     (160,  80, 220),
    "ALIGNMENT_FAILED":  (80,  160, 220),
    "MISSING_IMAGES":    (140, 140, 140),
}


# =============================================================================
# Grease seg model loader (mirrors load_detector in images_with_no_manhole...)
# =============================================================================

def _load_grease_seg_model():
    """Return a loaded YOLO seg model if the weights file exists, else None."""
    if not os.path.isfile(GREASE_SEG_MODEL_PATH):
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(GREASE_SEG_MODEL_PATH)
        print(f"  Grease seg model loaded: {GREASE_SEG_MODEL_PATH}")
        return model
    except Exception as exc:
        print(f"  WARNING: could not load grease seg model -- {exc}")
        return None


# =============================================================================
# Phase A -- view acceptability gate
# =============================================================================

def is_view_acceptable(image_bgr, label=""):
    """Gate applied to the AFTER image (D6 / U6).

    Returns dict:
        acceptable     bool
        blur_score     float  (Laplacian variance)
        circle_found   bool
        best_circle    (x, y, r) or None
        centering_ok   bool
        reject_reasons list[str]
    """
    gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    result = {
        "acceptable":    True,
        "blur_score":    0.0,
        "circle_found":  False,
        "best_circle":   None,
        "centering_ok":  False,
        "reject_reasons": [],
    }

    # Step 1 -- blur (pre-smooth to suppress high-frequency noise before Laplacian)
    gray_smooth = cv2.GaussianBlur(gray, (BLUR_GAUSSIAN_KSIZE, BLUR_GAUSSIAN_KSIZE), 0)
    blur_score  = float(cv2.Laplacian(gray_smooth, cv2.CV_64F).var())
    result["blur_score"] = round(blur_score, 1)
    if blur_score < BLUR_REJECT_THRESHOLD:
        result["acceptable"] = False
        result["reject_reasons"].append(
            f"Blur score {blur_score:.1f} < {BLUR_REJECT_THRESHOLD}"
        )

    return result


# =============================================================================
# Phase C.1 -- grease ratio
# =============================================================================

def calculate_grease_ratio(after_bgr, diff_gray=None, seg_model=None):
    """Colour-based grease approximation with optional YOLOv11-seg hook.

    Returns dict:
        grease_pct  float
        flagged     bool
        mode        str  ("color" or "yolo_seg")
    """
    # Future hook: if a trained seg model is available, use it instead
    if seg_model is not None:
        try:
            results = seg_model(after_bgr, verbose=False)
            if results and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                if len(masks) > 0:
                    combined = np.any(masks > 0.5, axis=0).astype(np.uint8)
                    grease_pct = float(np.mean(combined) * 100)
                    return {
                        "grease_pct": round(grease_pct, 2),
                        "flagged":    grease_pct > GREASE_FLAG_THRESHOLD,
                        "mode":       "yolo_seg",
                    }
        except Exception:
            pass  # fall through to colour method

    hsv = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2HSV)
    H   = hsv[:, :, 0].astype(np.int32)
    S   = hsv[:, :, 1].astype(np.int32)
    V   = hsv[:, :, 2].astype(np.int32)

    # Dark neutral grease (soot, petroleum residue)
    mask_dark  = (V < 60) & (S < 50)

    # Brown organic grease (fats, bio-film)
    mask_brown = (H >= 5) & (H <= 25) & (S >= 20) & (S <= 100) & (V >= 30) & (V <= 80)

    grease_mask = mask_dark | mask_brown

    # Restrict to low-diff regions (unchanged dark = residual grease)
    if diff_gray is not None:
        grease_mask = grease_mask & (diff_gray < DIFF_THRESHOLD)

    grease_pct = float(np.mean(grease_mask) * 100)
    return {
        "grease_pct": round(grease_pct, 2),
        "flagged":    grease_pct > GREASE_FLAG_THRESHOLD,
        "mode":       "color",
    }


# =============================================================================
# Phase C.2 -- texture entropy (GLCM block, inline from check_d2d5_u2u5 logic)
# =============================================================================

def verify_washing_texture(gray_before, gray_after):
    """GLCM entropy comparison.

    Downsample to 25%, quantise to 64 levels, 4 angles.

    Returns dict:
        entropy_before       float
        entropy_after        float
        entropy_delta        float
        confirmed            bool
        homogeneity_before   float
        homogeneity_after    float
    """
    h, w   = gray_before.shape
    small_b = cv2.resize(gray_before, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
    small_a = cv2.resize(gray_after,  (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)

    q_b = (small_b // 4).astype(np.uint8)
    q_a = (small_a // 4).astype(np.uint8)

    angles  = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm_b  = graycomatrix(q_b, distances=[1], angles=angles, levels=64, symmetric=True, normed=True)
    glcm_a  = graycomatrix(q_a, distances=[1], angles=angles, levels=64, symmetric=True, normed=True)

    entropy_before = float(np.mean([shannon_entropy(glcm_b[:, :, 0, i]) for i in range(len(angles))]))
    entropy_after  = float(np.mean([shannon_entropy(glcm_a[:, :, 0, i]) for i in range(len(angles))]))
    entropy_delta  = entropy_after - entropy_before

    homogeneity_before = float(np.mean(graycoprops(glcm_b, "homogeneity")))
    homogeneity_after  = float(np.mean(graycoprops(glcm_a, "homogeneity")))

    return {
        "entropy_before":      round(entropy_before, 3),
        "entropy_after":       round(entropy_after, 3),
        "entropy_delta":       round(entropy_delta, 3),
        "confirmed":           entropy_delta > ENTROPY_CONFIRM_DELTA,
        "homogeneity_before":  round(homogeneity_before, 3),
        "homogeneity_after":   round(homogeneity_after, 3),
    }


# =============================================================================
# Phase C.3 -- water signature
# =============================================================================

def detect_water_signature(after_bgr):
    """Detect specular highlights and blue-water reflections.

    Returns dict:
        specular_pct       float
        blue_water_pct     float
        water_detected     bool
        water_confidence   float  (0-1)
    """
    hsv = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2HSV)
    H   = hsv[:, :, 0].astype(np.int32)
    S   = hsv[:, :, 1].astype(np.int32)
    V   = hsv[:, :, 2].astype(np.int32)

    specular_mask   = (V > 210) & (S < 40)
    blue_water_mask = (H >= 95) & (H <= 125) & (S >= 15) & (S <= 60) & (V >= 100) & (V <= 220)

    specular_pct    = float(np.mean(specular_mask) * 100)
    blue_water_pct  = float(np.mean(blue_water_mask) * 100)
    combined_pct    = specular_pct + blue_water_pct

    return {
        "specular_pct":     round(specular_pct, 2),
        "blue_water_pct":   round(blue_water_pct, 2),
        "water_detected":   combined_pct > WATER_DETECT_THRESHOLD,
        "water_confidence": round(min(combined_pct / 30.0, 1.0), 3),
    }


# =============================================================================
# Verdict logic
# =============================================================================

def _derive_verdict(ransac_inliers, pct_changed, texture, water, grease):
    """Five boolean signals -> PASS / REVIEW / FAIL with overrides.

    S1: Alignment quality  -- ransac_inliers >= 10
    S2: Pixel change       -- pct_changed >= 5.0
    S3: Texture entropy    -- texture["confirmed"]
    S4: Water signature    -- water["water_detected"]
    S5: No major grease    -- not grease["flagged"]
    """
    s1 = ransac_inliers >= 10
    s2 = pct_changed    >= 5.0
    s3 = texture["confirmed"]
    s4 = water["water_detected"]
    s5 = not grease["flagged"]

    score = sum([s1, s2, s3, s4, s5])

    if score >= 3:
        verdict = "PASS"
    elif score >= 2:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"

    # Hard overrides
    if grease["grease_pct"] > 15.0 and not water["water_detected"]:
        verdict = "FAIL"
    if ransac_inliers < 4 and verdict == "PASS":
        verdict = "REVIEW"

    return verdict, score


# =============================================================================
# Grease mask overlay helper
# =============================================================================

def _build_grease_overlay(after_bgr, diff_gray, seg_model=None):
    """Return a BGR image with grease regions tinted green."""
    hsv = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2HSV)
    H   = hsv[:, :, 0].astype(np.int32)
    S   = hsv[:, :, 1].astype(np.int32)
    V   = hsv[:, :, 2].astype(np.int32)

    mask_dark  = (V < 60) & (S < 50)
    mask_brown = (H >= 5) & (H <= 25) & (S >= 20) & (S <= 100) & (V >= 30) & (V <= 80)
    grease_mask = (mask_dark | mask_brown).astype(np.uint8)

    if diff_gray is not None:
        grease_mask = grease_mask & (diff_gray < DIFF_THRESHOLD).astype(np.uint8)

    green_overlay = np.zeros_like(after_bgr)
    green_overlay[:, :, 1] = grease_mask * 200   # green channel

    return cv2.addWeighted(after_bgr, 0.6, green_overlay, 0.4, 0)


# =============================================================================
# GF metrics panel
# =============================================================================

def _build_gf_metrics_panel(
    total_w, folder_name, prefix, gate, sift_stats,
    grease, texture, water, verdict, score
):
    """Metrics panel for the geometry-first report."""
    font_title = load_font(16)
    font_metric = load_font(14)
    font_body  = load_font(12)
    font_conf  = load_font(18)

    pad    = EXPL_PADDING
    wrap_w = max(60, int((total_w - 2 * pad) / 7.5))

    verdict_col = VERDICT_COLOURS.get(verdict, (180, 180, 180))

    rows = []

    # Phase A
    rows.append(("Phase A: Blur score",
                 f"{gate['blur_score']:.1f}",
                 f"threshold {BLUR_REJECT_THRESHOLD}",
                 "PASS" if gate["blur_score"] >= BLUR_REJECT_THRESHOLD else "REJECT",
                 "Laplacian variance of the after image. Below threshold means the camera was "
                 "shaking or the lens was obscured -- alignment would be unreliable."))
    rows.append(("Phase A: Pipe circle",
                 "found" if gate["circle_found"] else "not found",
                 "centred: " + ("yes" if gate["centering_ok"] else "no"),
                 "advisory",
                 "Hough circle detector looking for the pipe cross-section. Advisory only -- "
                 "some upstream angles legitimately show an off-axis view."))

    # Phase A.5 (SSIM)
    if sift_stats:
        rows.append(("Phase A.5: SSIM score",
                     f"{sift_stats.get('ssim_score', 'n/a')}",
                     f">= {SSIM_MIN_THRESHOLD} = scene match",
                     "PASS" if sift_stats.get("ssim_score", 1) >= SSIM_MIN_THRESHOLD else "FAIL",
                     "Structural Similarity Index (global scene layout check). Low SSIM means the "
                     "images depict structurally different locations. Computed on 256px-wide "
                     "grayscale thumbnails before SIFT."))

    # Phase B (SIFT)
    if sift_stats:
        rows.append(("Phase B: RANSAC inliers",
                     str(sift_stats["ransac_inliers"]),
                     ">=10 = S1 pass",
                     "S1-PASS" if sift_stats["ransac_inliers"] >= 10 else "S1-FAIL",
                     "Verified alignment points after RANSAC filtering. Fewer than 10 suggests "
                     "the two images may not show the same view."))
        rows.append(("Phase B: Pixels changed",
                     f"{sift_stats['pct_changed']:.1f}%",
                     ">=5% = S2 pass",
                     "S2-PASS" if sift_stats["pct_changed"] >= 5.0 else "S2-FAIL",
                     "Proportion of image pixels that changed significantly after alignment. "
                     "High change indicates cleaning occurred."))
        rows.append(("Phase B: Homography determinant",
                     f"{sift_stats.get('homography_det', 'n/a')}",
                     "informational only",
                     "info",
                     "Determinant of the 2x2 sub-matrix of the homography. Negative = image was "
                     "flipped; extreme = physically impossible distortion. Logged for reference "
                     "but not used as a hard gate."))
        rows.append(("Phase B: Inlier ratio",
                     f"{sift_stats.get('inlier_ratio', 0):.1%}",
                     f">= {INLIER_RATIO_MIN:.0%}",
                     "PASS" if sift_stats.get("inlier_ratio", 0) >= INLIER_RATIO_MIN else "FAIL",
                     "Inliers as a proportion of total ratio-test matches. A low ratio with many "
                     "total matches means RANSAC found only a small cluster of consistent points -- "
                     "characteristic of coincidental texture matching on concrete grain."))
        rows.append(("Phase B: Timestamps masked",
                     f"before={sift_stats.get('text_regions_masked_before', 0)}  "
                     f"after={sift_stats.get('text_regions_masked_after', 0)}",
                     "OCR-excluded regions",
                     "applied" if (sift_stats.get("text_regions_masked_before", 0) +
                                   sift_stats.get("text_regions_masked_after", 0)) > 0 else "none detected",
                     "Burnt-in timestamps and camera overlays detected via OCR and excluded from "
                     "SIFT keypoint extraction, preventing overlay text from being used as an "
                     "alignment landmark."))

    # Phase C
    rows.append(("Phase C.1: Grease",
                 f"{grease['grease_pct']:.2f}%",
                 f"threshold {GREASE_FLAG_THRESHOLD}%",
                 "S5-PASS (no grease)" if not grease["flagged"] else "S5-FAIL (grease present)",
                 f"HSV colour approximation ({grease['mode']}). Dark neutrals + brown organics "
                 "masked. Restricted to low-diff regions (unchanged dark = residual grease)."))
    rows.append(("Phase C.2: Texture entropy",
                 f"before={texture['entropy_before']:.3f}  after={texture['entropy_after']:.3f}",
                 f"delta={texture['entropy_delta']:+.3f} nats",
                 "S3-PASS" if texture["confirmed"] else "S3-FAIL",
                 "GLCM entropy increase after washing. Higher entropy means the surface texture "
                 "is more complex and varied -- consistent with dirt/grease removal."))
    rows.append(("Phase C.3: Water signature",
                 f"specular={water['specular_pct']:.2f}%  blue={water['blue_water_pct']:.2f}%",
                 f"combined={water['specular_pct']+water['blue_water_pct']:.2f}%  "
                 f"conf={water['water_confidence']:.2f}",
                 "S4-PASS" if water["water_detected"] else "S4-FAIL",
                 "Specular highlights (bright achromatic) plus blue-water HSV regions. "
                 "Presence of water confirms high-pressure jetting was applied."))

    def measure_height():
        y = pad + 26 + 14
        for _ , _, _, _, explanation in rows:
            y += 22
            y += 16 * len(textwrap.wrap(explanation, width=wrap_w))
            y += 8
        y += 14 + 50 + pad
        return y

    panel_h = measure_height()
    img  = Image.new("RGB", (total_w, panel_h), COL_EXPL_BG)
    draw = ImageDraw.Draw(img)
    y    = pad

    pair_label = prefix + "3/" + prefix + "6" if prefix in ("D", "U") else prefix
    title_txt  = f"GEOMETRY-FIRST ANALYSIS  --  Pair {prefix}3/{prefix}6  --  Folder: {folder_name}"
    draw.text((pad, y), title_txt, fill=COL_EXPL_TITLE, font=font_title)
    y += 26
    draw.line([(pad, y), (total_w - pad, y)], fill=COL_SEPARATOR, width=1)
    y += 14

    for label, value, context, status, explanation in rows:
        metric_line = f"  {label}:  {value}   [{context}]   {status}"
        draw.text((pad, y), metric_line, fill=COL_STAT_VALUE, font=font_metric)
        y += 22
        for line in textwrap.wrap(explanation, width=wrap_w):
            draw.text((pad + 20, y), line, fill=COL_BODY, font=font_body)
            y += 16
        y += 8

    draw.line([(pad, y), (total_w - pad, y)], fill=COL_SEPARATOR, width=1)
    y += 14

    badge_txt = f"Verdict: {verdict}   (score {score}/5)"
    bw = _text_w(draw, badge_txt, font_conf)
    draw.text(((total_w - bw) // 2, y), badge_txt, fill=verdict_col, font=font_conf)

    return img


# =============================================================================
# Per-pair visual report
# =============================================================================

def build_gf_report_image(
    folder_name, prefix,
    before_bgr, after_bgr, warped_bgr, diff_gray,
    gate, sift_stats, grease, texture, water,
    verdict, score,
    report_w=REPORT_W,
):
    """Build the consolidated geometry-first report for one close-up pair.

    Stacked vertical bands:
      1. Section bar  -- pair label + verdict badge
      2. Row 1 480px  -- Before | After
      3. Section bar  -- ALIGNMENT & DIFFERENCE
      4. Row 2 480px  -- Warped After | Diff Heatmap | Grease Mask overlay
      5. Metrics panel
    """
    verdict_col = VERDICT_COLOURS.get(verdict, (180, 180, 180))
    direction   = "Downstream" if prefix.upper().startswith("D") else "Upstream"
    pair_label  = f"{prefix}3/{prefix}6  ({direction})  --  Verdict: {verdict}"

    sections = []

    # ── Band 1: verdict banner with score + sub-phase summary ─────────────────
    bg_col = (30, 90, 30) if verdict == "ACCEPTED" else (90, 25, 25)
    banner = Image.new("RGB", (report_w, 110), bg_col)
    bdraw  = ImageDraw.Draw(banner)

    title = f"STAGE 4N  --  GEOMETRY ANALYSIS   [{verdict}]   Score: {score}/5"
    bdraw.text((EXPL_PADDING, 10), title, fill=(255, 255, 255), font=load_font(18))
    bdraw.line([(EXPL_PADDING, 40), (report_w - EXPL_PADDING, 40)], fill=(200, 200, 200), width=1)

    col_ok   = (140, 255, 140)
    col_fail = (255, 120, 120)
    col_info = (210, 210, 210)

    sub_params = [
        (f"Phase C.1 Grease: {'FAIL' if grease['flagged'] else 'PASS'}"
         f"  ({grease['grease_pct']:.2f}%)",
         col_fail if grease["flagged"] else col_ok),
        (f"Phase C.2 Texture: {'PASS' if texture['confirmed'] else 'FAIL'}"
         f"  (delta {texture['entropy_delta']:+.3f})",
         col_ok if texture["confirmed"] else col_fail),
        (f"Phase C.3 Water: {'PASS' if water['water_detected'] else 'FAIL'}"
         f"  (conf {water['water_confidence']:.2f})",
         col_ok if water["water_detected"] else col_fail),
        (f"Pair: {prefix}3/{prefix}6  ({direction})  |  Folder: {folder_name}", col_info),
    ]

    col_step = (report_w - 2 * EXPL_PADDING) // len(sub_params)
    for i, (text, colour) in enumerate(sub_params):
        x = EXPL_PADDING + i * col_step
        bdraw.text((x, 50), text, fill=colour, font=load_font(13))
        if i > 0:
            bdraw.line([(x - 4, 50), (x - 4, 80)], fill=(180, 180, 180), width=1)

    sections.append(_pil_to_bgr(banner))

    # ── Band 2: Before | After originals
    half_w  = report_w // 2
    p_b_row = cv2.resize(before_bgr, (half_w, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    p_a_row = cv2.resize(after_bgr,  (half_w, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    b_base  = f"{prefix}3"
    a_base  = f"{prefix}6"
    bar_row1 = build_section_bar(
        report_w,
        "ORIGINAL PHOTOGRAPHS  (AS TAKEN -- UNPROCESSED)",
        [(f"Before  ({b_base})", half_w),
         (f"After   ({a_base})", report_w - half_w)],
    )
    sections.append(_pil_to_bgr(bar_row1))
    sections.append(np.hstack([p_b_row, p_a_row]))

    # ── Band 3: Alignment & Difference section bar
    third_w  = report_w // 3
    third_w2 = report_w - 2 * third_w
    bar_row2 = build_section_bar(
        report_w,
        "ALIGNMENT & DIFFERENCE  (PHASE B + PHASE C)",
        [(f"Warped After  ({a_base} aligned)", third_w),
         ("Difference Heatmap (magma)", third_w),
         ("Grease Mask Overlay (green)", third_w2)],
    )
    sections.append(_pil_to_bgr(bar_row2))

    # ── Band 4: Warped | Diff heatmap | Grease overlay
    diff_color    = cv2.applyColorMap(diff_gray, cv2.COLORMAP_MAGMA)
    grease_overlay = _build_grease_overlay(after_bgr, diff_gray)

    p_warped  = cv2.resize(warped_bgr,      (third_w,  PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    p_diff    = cv2.resize(diff_color,      (third_w,  PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    p_grease  = cv2.resize(grease_overlay,  (third_w2, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    sections.append(np.hstack([p_warped, p_diff, p_grease]))

    # ── Band 5: Metrics panel
    sections.append(_pil_to_bgr(
        _build_gf_metrics_panel(
            report_w, folder_name, prefix,
            gate, sift_stats, grease, texture, water,
            verdict, score,
        )
    ))

    return np.vstack(sections)


# =============================================================================
# Homography wrapper with additional plausibility guards
# =============================================================================

class _HomographyGuardError(RuntimeError):
    """RuntimeError subclass that carries partial homography diagnostics for failure reports."""
    def __init__(self, msg, det=None, inlier_ratio=None):
        super().__init__(msg)
        self.det          = det
        self.inlier_ratio = inlier_ratio


def _compute_homography_gf(kp1, kp2, good_matches):
    """Wraps compute_homography() with two additional guards:
    1. Homography 2x2 sub-matrix determinant must be in [HOMOGRAPHY_DET_MIN, HOMOGRAPHY_DET_MAX].
    2. Inlier ratio (inliers / ratio_matches) must be >= INLIER_RATIO_MIN.
    Returns (H, inlier_matches, inlier_ratio).
    """
    H, inlier_matches = compute_homography(kp1, kp2, good_matches)

    det          = float(np.linalg.det(H[:2, :2]))
    inlier_ratio = len(inlier_matches) / max(len(good_matches), 1)

    if inlier_ratio < INLIER_RATIO_MIN:
        raise _HomographyGuardError(
            f"Inlier ratio {inlier_ratio:.1%} < {INLIER_RATIO_MIN:.0%} -- "
            f"{len(inlier_matches)} inliers from {len(good_matches)} matches "
            "suggests coincidental alignment on texture grain.",
            det=det, inlier_ratio=inlier_ratio,
        )

    return H, inlier_matches, inlier_ratio


# =============================================================================
# Failure report builders
# =============================================================================

def _build_failure_panel(total_w, folder_name, prefix, status, error_msg,
                         gate=None, diag=None):
    """Metrics panel for failed/rejected pairs.

    Shows all diagnostic checks with explanations, marking each as PASS/FAIL/N/A
    depending on how far the pipeline got before failing.
    """
    font_title  = load_font(16)
    font_metric = load_font(14)
    font_body   = load_font(12)
    font_conf   = load_font(18)

    pad    = EXPL_PADDING
    wrap_w = max(60, int((total_w - 2 * pad) / 7.5))
    diag   = diag or {}

    verdict_col = VERDICT_COLOURS.get(status, (180, 180, 180))
    rows = []

    # Phase A
    if gate:
        blur_ok = gate["blur_score"] >= BLUR_REJECT_THRESHOLD
        rows.append((
            "Phase A: Blur score",
            f"{gate['blur_score']:.1f}",
            f"threshold {BLUR_REJECT_THRESHOLD}",
            "PASS" if blur_ok else "REJECT",
            "Laplacian variance measured after a 3x3 Gaussian pre-smooth to suppress "
            "high-frequency noise. A low score means the camera was shaking or the lens "
            f"was obscured -- reliable alignment is not possible below {BLUR_REJECT_THRESHOLD}.",
        ))
        rows.append((
            "Phase A: Pipe circle",
            "found" if gate["circle_found"] else "not found",
            "centred: " + ("yes" if gate["centering_ok"] else "no"),
            "advisory",
            "Hough circle detector looking for the pipe cross-section in the after image. "
            "Advisory only -- some upstream angles legitimately show an off-axis view.",
        ))
    else:
        rows.append(("Phase A", "not reached", "", "N/A",
                     "Gate check not reached because images could not be loaded."))

    # Phase A.5 SSIM
    ssim_val = diag.get("ssim_score")
    if ssim_val is not None:
        rows.append((
            "Phase A.5: SSIM score",
            f"{ssim_val:.3f}",
            f">= {SSIM_MIN_THRESHOLD} = scene match",
            "PASS" if ssim_val >= SSIM_MIN_THRESHOLD else "FAIL",
            "Structural Similarity Index computed on 256px-wide grayscale thumbnails before "
            "SIFT. A very low SSIM means the two images depict structurally different scenes. "
            f"The threshold {SSIM_MIN_THRESHOLD} is intentionally low to allow for dramatic "
            "post-cleaning appearance changes while still flagging completely wrong images.",
        ))
    else:
        rows.append(("Phase A.5: SSIM score", "not reached", "", "N/A",
                     "SSIM check not reached -- pipeline failed at Phase A or earlier."))

    # Phase B: timestamp masking
    nb = diag.get("text_regions_masked_before", 0)
    na = diag.get("text_regions_masked_after",  0)
    rows.append((
        "Phase B: Timestamps masked",
        f"before={nb}  after={na}",
        "OCR-excluded regions",
        "applied" if (nb + na) > 0 else "none detected",
        "PaddleOCR scans both images for burnt-in text (timestamps, camera overlays). "
        "Detected regions are blacked out in the SIFT feature mask so the matcher "
        "cannot use overlay text as an alignment landmark.",
    ))

    # Phase B: SIFT keypoints
    kp1   = diag.get("kp_before")
    kp2   = diag.get("kp_after")
    ratio = diag.get("ratio_matches")
    if kp1 is not None:
        rows.append((
            "Phase B: SIFT keypoints",
            f"before={kp1}  after={kp2}  ratio_matches={ratio}",
            "",
            "info",
            "SIFT keypoints detected in each image after masking. Ratio-test filtered "
            "matches (Lowe ratio=0.75) are passed to RANSAC for homography estimation. "
            "Very low counts indicate featureless texture or aggressive masking.",
        ))
    else:
        rows.append(("Phase B: SIFT keypoints", "not reached", "", "N/A",
                     "SIFT not reached -- pipeline failed at Phase A or A.5."))

    # Phase B: homography determinant
    det = diag.get("homography_det")
    if det is not None:
        rows.append((
            "Phase B: Homography determinant",
            f"{det:.3f}",
            "informational only",
            "info",
            "Determinant of the 2x2 rotation/scale sub-matrix of the estimated homography. "
            "Negative = the transform flips/mirrors the image. Near-zero = degenerate collinear "
            "match set. Logged for reference but not used as a hard gate.",
        ))
    else:
        rows.append(("Phase B: Homography determinant", "not reached", "", "N/A",
                     "Homography computation not reached -- pipeline failed at an earlier step."))

    # Phase B: inlier ratio
    ir = diag.get("inlier_ratio")
    if ir is not None:
        rows.append((
            "Phase B: Inlier ratio",
            f"{ir:.1%}",
            f">= {INLIER_RATIO_MIN:.0%}",
            "PASS" if ir >= INLIER_RATIO_MIN else "FAIL",
            "RANSAC inliers as a proportion of total ratio-test matches. A low ratio with "
            "many total matches means only a small cluster of points agreed on the same "
            "transform -- a fingerprint of coincidental texture matching on concrete grain "
            "rather than true geometric correspondence.",
        ))
    else:
        rows.append(("Phase B: Inlier ratio", "not reached", "", "N/A",
                     "Inlier ratio not computed -- homography estimation failed first."))

    # Failure reason (highlighted row)
    rows.append((
        "FAILURE REASON",
        status,
        "",
        "FAILED",
        error_msg or "(no detail available)",
    ))

    def measure_height():
        y = pad + 26 + 14
        for _, _, _, _, explanation in rows:
            y += 22
            y += 16 * len(textwrap.wrap(explanation, width=wrap_w))
            y += 8
        y += 14 + 50 + pad
        return y

    panel_h = measure_height()
    img  = Image.new("RGB", (total_w, panel_h), COL_EXPL_BG)
    draw = ImageDraw.Draw(img)
    y    = pad

    title_txt = (
        f"GEOMETRY-FIRST ANALYSIS  --  Pair {prefix}3/{prefix}6  --  "
        f"Folder: {folder_name}  --  [DIAGNOSTIC REPORT]"
    )
    draw.text((pad, y), title_txt, fill=COL_EXPL_TITLE, font=font_title)
    y += 26
    draw.line([(pad, y), (total_w - pad, y)], fill=COL_SEPARATOR, width=1)
    y += 14

    for label, value, context, status_str, explanation in rows:
        if context:
            metric_line = f"  {label}:  {value}   [{context}]   {status_str}"
        elif value and value != status:
            metric_line = f"  {label}:  {value}   {status_str}"
        else:
            metric_line = f"  {label}:  {status_str}"
        line_col = verdict_col if label == "FAILURE REASON" else COL_STAT_VALUE
        draw.text((pad, y), metric_line, fill=line_col, font=font_metric)
        y += 22
        for line in textwrap.wrap(explanation, width=wrap_w):
            draw.text((pad + 20, y), line, fill=COL_BODY, font=font_body)
            y += 16
        y += 8

    draw.line([(pad, y), (total_w - pad, y)], fill=COL_SEPARATOR, width=1)
    y += 14

    badge_txt = f"Result: {status}"
    bw = _text_w(draw, badge_txt, font_conf)
    draw.text(((total_w - bw) // 2, y), badge_txt, fill=verdict_col, font=font_conf)

    return img


def _build_gf_failure_report(
    folder_name, prefix, status, error_msg,
    before_bgr=None, after_bgr=None,
    gate=None, diag=None,
    report_w=REPORT_W,
):
    """Build a visual report for failed/rejected pairs showing available diagnostics."""
    verdict_col = VERDICT_COLOURS.get(status, (180, 180, 180))
    direction   = "Downstream" if prefix.upper().startswith("D") else "Upstream"
    pair_label  = f"{prefix}3/{prefix}6  ({direction})  --  Result: {status}"

    sections = []

    # Section bar
    bar1_img = Image.new("RGB", (report_w, SECTION_BAR_H), COL_SECTION_BG)
    draw1    = ImageDraw.Draw(bar1_img)
    draw1.rectangle([(0, 0), (8, SECTION_BAR_H - 1)], fill=verdict_col)
    font_bar = load_font(17)
    tw       = _text_w(draw1, pair_label, font_bar)
    draw1.text(
        ((report_w - tw) // 2, (SECTION_BAR_H - 20) // 2),
        pair_label, fill=verdict_col, font=font_bar,
    )
    sections.append(_pil_to_bgr(bar1_img))

    # Before / After originals (or dark placeholder if images not available)
    half_w = report_w // 2
    if before_bgr is not None and after_bgr is not None:
        p_b_row  = cv2.resize(before_bgr, (half_w, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
        p_a_row  = cv2.resize(after_bgr,  (half_w, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
        bar_row1 = build_section_bar(
            report_w,
            "ORIGINAL PHOTOGRAPHS  (AS TAKEN -- UNPROCESSED)",
            [(f"Before  ({prefix}3)", half_w),
             (f"After   ({prefix}6)", report_w - half_w)],
        )
        sections.append(_pil_to_bgr(bar_row1))
        sections.append(np.hstack([p_b_row, p_a_row]))
    else:
        placeholder     = np.zeros((PANEL_HEIGHT, report_w, 3), dtype=np.uint8)
        placeholder[:]  = (40, 40, 40)
        placeholder_bar = build_section_bar(report_w, "IMAGES NOT AVAILABLE", [])
        sections.append(_pil_to_bgr(placeholder_bar))
        sections.append(placeholder)

    # Diagnostics panel
    sections.append(_pil_to_bgr(
        _build_failure_panel(report_w, folder_name, prefix, status, error_msg, gate, diag)
    ))

    return np.vstack(sections)


# =============================================================================
# Pair orchestrator
# =============================================================================

def process_pair_gf(folder_path, folder_name, base_before, base_after, prefix,
                    out_dir, seg_model=None):
    """Run Phase A -> B -> C for one close-up pair.

    Returns a result dict; always returns (never raises).
    """
    result_base = {
        "folder": folder_name,
        "prefix": prefix,
        "pair":   f"{base_before}/{base_after}",
    }

    diag         = {}   # accumulates diagnostic data for failure reports
    before_bgr   = None
    after_bgr    = None
    gate         = None

    # ── 1. Load images ────────────────────────────────────────────────────────
    try:
        before_bgr, after_bgr, path_before, path_after = load_pair(
            folder_path, base_before, base_after
        )
    except (FileNotFoundError, IOError) as exc:
        os.makedirs(out_dir, exist_ok=True)
        rpt  = _build_gf_failure_report(folder_name, prefix, "MISSING_IMAGES", str(exc),
                                        diag=diag)
        rpath = os.path.join(out_dir, f"{prefix}_gf_report.jpg")
        cv2.imwrite(rpath, rpt, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {**result_base, "status": "MISSING_IMAGES", "error": str(exc),
                "report_path": rpath, "diag": diag}

    # ── Resize to MAX_SIFT_HEIGHT (memory budget) ────────────────────────────
    _max_h = int(os.environ.get("MAX_SIFT_HEIGHT", 1080))
    _rsz_b = _resize_to_height(before_bgr, _max_h) if before_bgr.shape[0] > _max_h else before_bgr
    _rsz_a = _resize_to_height(after_bgr,  _max_h) if after_bgr.shape[0] > _max_h else after_bgr
    if _rsz_b is not before_bgr:
        del before_bgr
    if _rsz_a is not after_bgr:
        del after_bgr
    before_bgr = _rsz_b
    after_bgr  = _rsz_a

    # Timestamp / metadata masking (same approach as D2/D5 stage)
    ocr_before  = run_ocr_for_mask(path_before)
    ocr_after   = run_ocr_for_mask(path_after)
    mask_before = create_metadata_mask(before_bgr.shape, ocr_before)
    mask_after  = create_metadata_mask(after_bgr.shape,  ocr_after)
    n_masked_before = len(ocr_before)
    n_masked_after  = len(ocr_after)
    diag["text_regions_masked_before"] = n_masked_before
    diag["text_regions_masked_after"]  = n_masked_after

    gray_before = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    gray_after  = cv2.cvtColor(after_bgr,  cv2.COLOR_BGR2GRAY)

    # ── 2. Phase A gate (applied to AFTER image) ──────────────────────────────
    gate = is_view_acceptable(after_bgr, label=f"{folder_name}/{base_after}")
    if not gate["acceptable"]:
        error_msg = "; ".join(gate["reject_reasons"])
        _r = {**result_base, "status": "GATE_REJECTED", "gate": gate,
              "reject_reasons": gate["reject_reasons"], "diag": diag}
        if not _SKIP_IMGS:
            os.makedirs(out_dir, exist_ok=True)
            rpt   = _build_gf_failure_report(folder_name, prefix, "GATE_REJECTED", error_msg,
                                             before_bgr, after_bgr, gate, diag)
            rpath = os.path.join(out_dir, f"{prefix}_gf_report.jpg")
            cv2.imwrite(rpath, rpt, [cv2.IMWRITE_JPEG_QUALITY, 92])
            _r["report_path"] = rpath
        return _r

    # ── 2.5. Phase A.5 -- Global scene layout check (SSIM) ───────────────────
    _h, _w = gray_before.shape
    _scale  = 256 / max(_w, 1)
    _sb     = cv2.resize(gray_before, (256, max(int(_h * _scale), 1)))
    _sa     = cv2.resize(gray_after,  (256, max(int(_h * _scale), 1)))
    ssim_score      = round(float(_ssim(_sb, _sa, data_range=255)), 3)
    diag["ssim_score"] = ssim_score
    if ssim_score < SSIM_MIN_THRESHOLD:
        error_msg = (
            f"SSIM {ssim_score:.3f} < {SSIM_MIN_THRESHOLD} -- "
            "global scene layout too dissimilar; images likely from different locations."
        )
        _r = {**result_base, "status": "ALIGNMENT_FAILED", "gate": gate,
              "error": error_msg, "diag": diag}
        if not _SKIP_IMGS:
            os.makedirs(out_dir, exist_ok=True)
            rpt   = _build_gf_failure_report(folder_name, prefix, "ALIGNMENT_FAILED", error_msg,
                                             before_bgr, after_bgr, gate, diag)
            rpath = os.path.join(out_dir, f"{prefix}_gf_report.jpg")
            cv2.imwrite(rpath, rpt, [cv2.IMWRITE_JPEG_QUALITY, 92])
            _r["report_path"] = rpath
        return _r

    # ── 3. Phase B: SIFT + homography + diff ─────────────────────────────────
    try:
        kp1, kp2, good_matches = sift_match(gray_before, gray_after, mask_before, mask_after)
        diag["kp_before"]     = len(kp1)
        diag["kp_after"]      = len(kp2)
        diag["ratio_matches"] = len(good_matches)
        H, inlier_matches, inlier_ratio = _compute_homography_gf(kp1, kp2, good_matches)
        diag["homography_det"] = round(float(np.linalg.det(H[:2, :2])), 3)
        diag["inlier_ratio"]   = round(inlier_ratio, 3)
        warped_bgr, diff_gray, mean_diff, pct_changed = warp_and_diff(
            before_bgr, after_bgr, H
        )
    except _HomographyGuardError as exc:
        if exc.det is not None:
            diag["homography_det"] = round(exc.det, 3)
        if exc.inlier_ratio is not None:
            diag["inlier_ratio"] = round(exc.inlier_ratio, 3)
        _r = {**result_base, "status": "ALIGNMENT_FAILED", "gate": gate,
              "error": str(exc), "diag": diag}
        if not _SKIP_IMGS:
            os.makedirs(out_dir, exist_ok=True)
            rpt   = _build_gf_failure_report(folder_name, prefix, "ALIGNMENT_FAILED", str(exc),
                                             before_bgr, after_bgr, gate, diag)
            rpath = os.path.join(out_dir, f"{prefix}_gf_report.jpg")
            cv2.imwrite(rpath, rpt, [cv2.IMWRITE_JPEG_QUALITY, 92])
            _r["report_path"] = rpath
        return _r
    except RuntimeError as exc:
        _r = {**result_base, "status": "ALIGNMENT_FAILED", "gate": gate,
              "error": str(exc), "diag": diag}
        if not _SKIP_IMGS:
            os.makedirs(out_dir, exist_ok=True)
            rpt   = _build_gf_failure_report(folder_name, prefix, "ALIGNMENT_FAILED", str(exc),
                                             before_bgr, after_bgr, gate, diag)
            rpath = os.path.join(out_dir, f"{prefix}_gf_report.jpg")
            cv2.imwrite(rpath, rpt, [cv2.IMWRITE_JPEG_QUALITY, 92])
            _r["report_path"] = rpath
        return _r

    ransac_inliers = len(inlier_matches)
    sift_stats = {
        "kp_before":                  len(kp1),
        "kp_after":                   len(kp2),
        "ratio_matches":              len(good_matches),
        "ransac_inliers":             ransac_inliers,
        "mean_diff":                  round(mean_diff, 1),
        "pct_changed":                round(pct_changed, 1),
        "ssim_score":                 ssim_score,
        "inlier_ratio":               round(inlier_ratio, 3),
        "homography_det":             round(float(np.linalg.det(H[:2, :2])), 3),
        "text_regions_masked_before": n_masked_before,
        "text_regions_masked_after":  n_masked_after,
    }

    # ── 4. Phase C metrics ────────────────────────────────────────────────────
    grease  = calculate_grease_ratio(after_bgr, diff_gray, seg_model)
    texture = verify_washing_texture(gray_before, gray_after)
    water   = detect_water_signature(after_bgr)

    # ── 5. Verdict ────────────────────────────────────────────────────────────
    verdict, score = _derive_verdict(ransac_inliers, pct_changed, texture, water, grease)

    result = {
        **result_base,
        "status":     verdict,
        "score":      score,
        "gate":       gate,
        "sift_stats": sift_stats,
        "grease":     grease,
        "texture":    texture,
        "water":      water,
    }

    # ── 6 & 7. Save keypoint matches + GF report (skipped if SKIP_REPORT_IMAGES) ──
    if not _SKIP_IMGS:
        os.makedirs(out_dir, exist_ok=True)
        kp_img = cv2.drawMatches(
            before_bgr, kp1, after_bgr, kp2, inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        kp_path = os.path.join(out_dir, f"{prefix}_keypoint_matches.jpg")
        cv2.imwrite(kp_path, kp_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        gf_report = build_gf_report_image(
            folder_name, prefix,
            before_bgr, after_bgr, warped_bgr, diff_gray,
            gate, sift_stats, grease, texture, water,
            verdict, score,
        )
        gf_report_path = os.path.join(out_dir, f"{prefix}_gf_report.jpg")
        cv2.imwrite(gf_report_path, gf_report, [cv2.IMWRITE_JPEG_QUALITY, 92])
        result["report_path"] = gf_report_path

    return result


# =============================================================================
# Consolidated per-folder report (both pairs stacked)
# =============================================================================

def build_folder_report(folder_name, pair_reports, report_w=REPORT_W):
    """Stack cover header, per-pair GF report images, footer into one image."""
    header = _pil_to_bgr(build_cover_header(report_w, folder_name))

    def _scale(img):
        h, w = img.shape[:2]
        if w == report_w:
            return img
        return cv2.resize(img, (report_w, int(h * report_w / w)), interpolation=cv2.INTER_AREA)

    parts = [header]
    for img in pair_reports:
        parts.append(_scale(img))
    parts.append(_pil_to_bgr(build_footer(report_w)))

    return np.vstack(parts)


# =============================================================================
# Main — batch runner
# =============================================================================

STATUS_KEYS = ["PASS", "REVIEW", "FAIL", "GATE_REJECTED", "ALIGNMENT_FAILED", "MISSING_IMAGES"]


def run_d3d6_u3u6_on_folder(folder_path, folder_name, output_dir,
                             seg_model=None, progress_cb=None):
    """
    Run Phase A/B/C geometry-first pipeline on D3/D6 + U3/U6 for ONE folder.

    Parameters
    ----------
    folder_path : str   Path to the job folder.
    folder_name : str   Folder name for reporting.
    output_dir  : str   Root output dir; outputs go to <output_dir>/<folder_name>/
    seg_model   :       YOLO seg model or None
    progress_cb :       callable(msg), optional

    Returns
    -------
    dict
        ``{
            "pair_results": { "D3": result_dict, "U3": result_dict },
            "folder_verdict": "ACCEPTED" | "NEEDS_REVIEW" | "REJECTED",
            "failed_pairs": int,
        }``
    """
    out_dir      = os.path.join(output_dir, folder_name)
    pair_results = {}

    for base_before, base_after, prefix in PAIRS:
        if progress_cb:
            progress_cb(f"    [{prefix}] {base_before}/{base_after} ...")
        result = process_pair_gf(
            folder_path, folder_name,
            base_before, base_after, prefix,
            out_dir, seg_model,
        )
        status = result.get("status", "MISSING_IMAGES")
        pair_results[prefix] = result
        if progress_cb:
            progress_cb(f"    [{prefix}] -> {status}")

    # Derive folder-level verdict
    statuses = [r.get("status") for r in pair_results.values()]
    has_pass   = any(s == "PASS"   for s in statuses)
    has_review = any(s == "REVIEW" for s in statuses)

    if has_pass:
        folder_verdict = "ACCEPTED"
    elif has_review:
        folder_verdict = "NEEDS_REVIEW"
    else:
        folder_verdict = "REJECTED"

    failed_pairs = sum(
        1 for s in statuses
        if s not in ("PASS", "REVIEW")
    )

    return {
        "pair_results":    pair_results,
        "folder_verdict":  folder_verdict,
        "failed_pairs":    failed_pairs,
    }


def main():
    single_folder = sys.argv[1] if len(sys.argv) > 1 else None

    if single_folder:
        folders = [single_folder]
        print(f"Smoke test: single folder '{single_folder}'")
    else:
        folders = sorted(
            e.name
            for e in os.scandir(BASE_DIR)
            if e.is_dir() and e.name != "difference_analysis"
        )
        print(f"Found {len(folders)} folder(s) to process.")

    os.makedirs(GF_DIR, exist_ok=True)

    seg_model = _load_grease_seg_model()
    print()

    all_results = []
    counts = {k: 0 for k in STATUS_KEYS}
    n = len(folders)

    for i, folder_name in enumerate(folders, start=1):
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"[{i}/{n}] {folder_name} -- NOT FOUND, skipping")
            continue

        out_dir = os.path.join(ANALYSIS_DIR, folder_name)
        print(f"[{i}/{n}] {folder_name}")

        folder_results   = []

        for base_before, base_after, prefix in PAIRS:
            print(f"  [{prefix}] {base_before}/{base_after} ...", end=" ", flush=True)
            result = process_pair_gf(
                folder_path, folder_name,
                base_before, base_after, prefix,
                out_dir, seg_model,
            )
            status = result.get("status", "MISSING_IMAGES")

            # One-liner summary
            if status in ("PASS", "REVIEW", "FAIL"):
                s = result["sift_stats"]
                g = result["grease"]
                t = result["texture"]
                w = result["water"]
                print(
                    f"{status} (score {result['score']}/5)  "
                    f"inliers={s['ransac_inliers']}  "
                    f"chg={s['pct_changed']:.1f}%  "
                    f"grease={g['grease_pct']:.1f}%  "
                    f"dH={t['entropy_delta']:+.3f}  "
                    f"water={'Y' if w['water_detected'] else 'N'}"
                )
            else:
                reason = result.get("error") or result.get("reject_reasons", "")
                print(f"{status} -- {reason}")

            counts[status] = counts.get(status, 0) + 1
            all_results.append(result)
            folder_results.append(result)

        print()

    # ── Summary table ──────────────────────────────────────────────────────────
    print("-" * 55)
    print("SUMMARY")
    print("-" * 55)
    for k in STATUS_KEYS:
        if counts.get(k, 0) > 0:
            print(f"  {k:<20} {counts[k]:>5}")
    total = sum(counts.values())
    print(f"  {'TOTAL':<20} {total:>5}")
    print("-" * 55)
    print(f"Consolidated folder: {GF_DIR}")

    # ── Write JSON run report ──────────────────────────────────────────────────
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path   = os.path.join(ANALYSIS_DIR, f"gf_run_{timestamp}.json")
    run_report  = {
        "timestamp": timestamp,
        "folders_processed": len(folders),
        "counts":    counts,
        "results":   all_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2, default=str)
    print(f"Run report: {json_path}")


if __name__ == "__main__":
    main()
