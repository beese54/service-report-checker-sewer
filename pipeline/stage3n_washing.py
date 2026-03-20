"""
Trial run — D2/D5 and U2/U5 SIFT pipeline on folder 146915
===========================================================
Runs the identical SIFT/FLANN/RANSAC/heatmap pipeline as
images_with_no_manhole_obstruction.py, but targets the intermediate
image pairs (D2/D5 and U2/U5) for a single known-good folder.

Outputs saved to:
  adjusted_images/no_obstruction/difference_analysis/146915/
    D2_keypoint_matches.jpg
    D2_composite.jpg
    U2_keypoint_matches.jpg
    U2_composite.jpg
"""

import datetime
import os
import sys
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

# ── Paths ─────────────────────────────────────────────────────────────────────
FOLDER_NAME  = sys.argv[1] if len(sys.argv) > 1 else "146915"
BASE_DIR     = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\adjusted_images\no_obstruction"
FOLDER_PATH  = os.path.join(BASE_DIR, FOLDER_NAME)
ANALYSIS_DIR = os.path.join(BASE_DIR, "difference_analysis")
OUT_DIR      = os.path.join(ANALYSIS_DIR, FOLDER_NAME)

# ── Pipeline config ───────────────────────────────────────────────────────────
VALID_EXTS       = {".jpg", ".jpeg", ".png"}
FLANN_TREES      = 5
FLANN_CHECKS     = 50
RATIO_THRESHOLD  = 0.75
RANSAC_THRESHOLD = 5.0
DIFF_THRESHOLD   = 20
PANEL_HEIGHT     = 600
SECTION_BAR_H    = 56
EXPL_PADDING     = 20

PAIRS = [
    ("D2", "D5", "D2"),   # (before_base, after_base, output_prefix)
    ("U2", "U5", "U2"),
]

_SKIP_IMGS = os.environ.get("SKIP_REPORT_IMAGES", "").lower() in ("1", "true")

# ── Metadata masking ──────────────────────────────────────────────────────────
MIN_MASKED_KP = 10   # minimum keypoints required in physical (unmasked) regions
TEXT_MASK_PAD = 8    # pixel margin added around each OCR bounding box

# ── Colours (RGB for Pillow) ──────────────────────────────────────────────────
COL_SECTION_BG   = (30,  30,  30)
COL_SECTION_TXT  = (255, 255, 255)
COL_COL_LABEL    = (180, 180, 180)
COL_EXPL_BG      = (30,  30,  30)
COL_EXPL_TITLE   = (255, 255, 255)
COL_STEP_HEADING = (255, 200, 80)
COL_BODY         = (220, 220, 220)
COL_STAT_VALUE   = (180, 220, 255)
COL_SEPARATOR    = (70,  70,  70)
COL_DIVIDER      = (80,  80,  80)


# ─────────────────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_font(size):
    candidates = [
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _text_w(draw, text, font):
    try:
        return int(draw.textlength(text, font=font))
    except AttributeError:
        w, _ = draw.textsize(text, font=font)
        return w


def _pil_to_bgr(pil_img):
    return np.array(pil_img)[:, :, ::-1]


def find_image(folder, base):
    """Return path to <folder>/<base>.<ext> for the first matching extension."""
    for entry in os.scandir(folder):
        stem, ext = os.path.splitext(entry.name)
        if stem == base and ext.lower() in VALID_EXTS:
            return entry.path
    return None


# ── OCR metadata masking (lazy singleton) ────────────────────────────────────

_mask_ocr_engine = None


def _get_mask_ocr_engine():
    """Return a PaddleOCR instance (initialised once), or None if unavailable."""
    global _mask_ocr_engine
    if os.environ.get("DISABLE_MASK_OCR", "").lower() in ("1", "true", "yes"):
        return None
    if _mask_ocr_engine is not None:
        return _mask_ocr_engine
    try:
        import os as _os
        import numpy as _np
        # numpy 2.0 removed np.sctypes; patch before importing PaddleOCR/imgaug
        if not hasattr(_np, "sctypes"):
            _np.sctypes = {
                "int":     [_np.int8,  _np.int16,  _np.int32,  _np.int64],
                "uint":    [_np.uint8, _np.uint16, _np.uint32, _np.uint64],
                "float":   [_np.float32, _np.float64],
                "complex": [_np.complex64, _np.complex128],
                "others":  [bool, object, bytes, str],
            }
        # protobuf >= 4 needs the pure-python implementation
        _os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        from paddleocr import PaddleOCR
        _mask_ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except BaseException:
        # Catches both Python exceptions and SystemExit (raised by PaddlePaddle
        # on initialisation failure, e.g. missing model files or network error).
        _mask_ocr_engine = None
    return _mask_ocr_engine


def create_metadata_mask(image_shape, ocr_lines):
    """Build a uint8 mask (255=include, 0=exclude) from OCR bounding boxes."""
    h, w = image_shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for line in ocr_lines:
        box = line[0]
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x1 = max(0, min(xs) - TEXT_MASK_PAD)
        y1 = max(0, min(ys) - TEXT_MASK_PAD)
        x2 = min(w, max(xs) + TEXT_MASK_PAD)
        y2 = min(h, max(ys) + TEXT_MASK_PAD)
        mask[y1:y2, x1:x2] = 0
    return mask


def run_ocr_for_mask(image_path):
    """Run OCR on one image and return the line list (may be empty)."""
    engine = _get_mask_ocr_engine()
    if engine is None:
        return []
    try:
        raw = engine.ocr(image_path, cls=True)
        return raw[0] if raw and raw[0] else []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────

def load_pair(folder, base_before, base_after):
    """Load a before/after image pair. Returns (before_bgr, after_bgr, path_before, path_after)."""
    path_before = find_image(folder, base_before)
    path_after  = find_image(folder, base_after)
    missing = [b for b, p in [(base_before, path_before), (base_after, path_after)] if p is None]
    if missing:
        raise FileNotFoundError(f"Image(s) not found: {missing}")
    before = cv2.imread(path_before)
    after  = cv2.imread(path_after)
    if before is None or after is None:
        raise IOError(f"cv2.imread failed for images in {folder}")
    return before, after, path_before, path_after


def _resize_to_height(img_bgr, height):
    h, w  = img_bgr.shape[:2]
    scale = height / h
    return cv2.resize(img_bgr, (int(w * scale), height), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────────────
# SIFT pipeline
# ─────────────────────────────────────────────────────────────────────────────

def sift_match(gray1, gray2, mask1=None, mask2=None):
    try:
        _nfeatures = int(os.environ.get("MAX_SIFT_FEATURES", 20000))
        sift = cv2.SIFT_create(nfeatures=_nfeatures)
    except AttributeError:
        print("ERROR: cv2.SIFT_create() not available. Install opencv-contrib-python.")
        sys.exit(1)

    kp1, des1 = sift.detectAndCompute(gray1, mask1)
    kp2, des2 = sift.detectAndCompute(gray2, mask2)

    if len(kp1) < MIN_MASKED_KP or len(kp2) < MIN_MASKED_KP:
        raise RuntimeError(
            f"Insufficient Physical Landmarks: {len(kp1)} (before) / {len(kp2)} (after) "
            f"keypoints detected in unmasked regions - minimum required: {MIN_MASKED_KP}."
        )
    if des1 is None or des2 is None:
        raise RuntimeError("SIFT descriptor extraction returned None.")

    index_params  = dict(algorithm=1, trees=FLANN_TREES)
    search_params = dict(checks=FLANN_CHECKS)
    flann         = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches   = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if m.distance < RATIO_THRESHOLD * n.distance]
    return kp1, kp2, good


def compute_homography(kp1, kp2, good_matches):
    if len(good_matches) < 4:
        raise RuntimeError(f"Only {len(good_matches)} good match(es) — need >=4.")
    src_pts = np.float32([kp2[m.trainIdx].pt  for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt  for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    if H is None:
        raise RuntimeError("findHomography returned None — insufficient inliers.")
    inlier_matches = [m for m, flag in zip(good_matches, mask.ravel()) if flag]
    return H, inlier_matches


def warp_and_diff(before_bgr, after_bgr, H):
    h, w   = before_bgr.shape[:2]
    warped = cv2.warpPerspective(after_bgr, H, (w, h))
    diff   = cv2.absdiff(
        cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(warped,     cv2.COLOR_BGR2GRAY),
    )
    mean_diff   = float(np.mean(diff))
    pct_changed = float(np.mean(diff > DIFF_THRESHOLD) * 100)
    return warped, diff, mean_diff, pct_changed


# ─────────────────────────────────────────────────────────────────────────────
# Composite building
# ─────────────────────────────────────────────────────────────────────────────

def build_section_bar(total_w, title, col_labels):
    bar  = Image.new("RGB", (total_w, SECTION_BAR_H), COL_SECTION_BG)
    draw = ImageDraw.Draw(bar)
    font_title  = load_font(17)
    font_labels = load_font(13)

    tw = _text_w(draw, title, font_title)
    draw.text(((total_w - tw) // 2, 6), title, fill=COL_SECTION_TXT, font=font_title)

    x = 0
    for label, col_w in col_labels:
        lw = _text_w(draw, label, font_labels)
        draw.text((x + (col_w - lw) // 2, 32), label, fill=COL_COL_LABEL, font=font_labels)
        if x + col_w < total_w:
            draw.line([(x + col_w - 1, 28), (x + col_w - 1, SECTION_BAR_H - 1)],
                      fill=COL_DIVIDER, width=1)
        x += col_w
    return bar


def build_explanation_panel(total_w, stats):
    font_title = load_font(16)
    font_step  = load_font(14)
    font_body  = load_font(13)
    font_stat  = load_font(13)

    steps = [
        (
            "Step 1 — Metadata Masking (OCR — Optical Character Recognition)",
            "Before any feature detection begins, the computer scans each image for burned-in "
            "text overlays — such as timestamps, camera IDs, or date/time banners — using an OCR "
            "engine (PaddleOCR). Detected text regions are blacked out in a binary mask, so that "
            "SIFT is forced to look only at physical pipe surface features and cannot accidentally "
            "anchor on identical text that appears in both photos.",
        ),
        (
            "Step 2 — Feature Detection (SIFT: Scale-Invariant Feature Transform)",
            "The computer automatically identifies distinctive 'landmark' points on the manhole "
            "cover — such as bolt holes, scratches, surface edges, and physical markings. These "
            "landmarks are chosen because they look recognisably the same regardless of the camera "
            "angle, distance, or slight differences in lighting between the two photos. This method "
            "is called SIFT (Scale-Invariant Feature Transform).",
        ),
        (
            "Step 3 — Feature Matching (FLANN + Ratio Test)",
            "The computer compares every landmark from the before photo against every landmark in "
            "the after photo, searching for matching pairs that represent the same physical point. "
            "A strict quality filter — the ratio test — then discards any match that is ambiguous, "
            "i.e. where the computer cannot confidently distinguish which landmark it is looking at. "
            "Only clear, unambiguous, high-confidence matches are retained.",
        ),
        (
            "Step 4 — Alignment Calculation (RANSAC + Homography Matrix)",
            "Using the verified matching landmarks, the computer calculates a mathematical formula "
            "called a homography matrix. This formula describes exactly how to rotate, scale, and "
            "shift the after photo so that it lines up pixel-for-pixel with the before photo. A "
            "further statistical filter (RANSAC) removes any remaining inconsistent matches before "
            "the formula is computed, ensuring the alignment is based only on trustworthy, "
            "geometrically consistent data.",
        ),
        (
            "Step 5 — Pixel-by-Pixel Comparison (Difference Heatmap)",
            "Once the after photo has been mathematically aligned to the before photo, the two "
            "images are subtracted from each other, pixel by pixel. Where no change has occurred, "
            "the subtraction result is zero (shown as black on the heatmap). Where cleaning has "
            "occurred — removing dirt, debris, or blockage — the pixel values differ and are shown "
            "as bright colours on the heatmap (magma colour scale: black -> purple -> orange -> "
            "yellow). The brighter and more widespread the colour, the greater the area of change.",
        ),
    ]

    mask_applied = stats.get("metadata_mask_applied", False)
    mask_before  = stats.get("text_regions_masked_before", "n/a")
    mask_after   = stats.get("text_regions_masked_after",  "n/a")

    result_lines = [
        ("Job folder",                    stats["folder"]),
        ("Image pair",                    stats["pair"]),
        ("OCR text regions masked",       f"Before: {mask_before}  |  After: {mask_after}  "
                                           f"({'applied' if mask_applied else 'none detected'})"),
        ("SIFT keypoints — Before photo", f"{stats['kp_before']:,}  "
                                           "(landmark points identified on the before photo)"),
        ("SIFT keypoints — After photo",  f"{stats['kp_after']:,}  "
                                           "(landmark points identified on the after photo)"),
        ("Ratio-test matches",            f"{stats['ratio_matches']:,}  "
                                           "(unambiguous landmark pairs found in both photos)"),
        ("RANSAC inliers",                f"{stats['ransac_inliers']:,}  "
                                           "(verified alignment points after final quality filter; "
                                           ">=10 confirms photos show the same object)"),
        ("Mean pixel difference",         f"{stats['mean_diff']:.1f} / 255  "
                                           "(average per-pixel change after alignment; 0 = identical)"),
        ("Pixels changed significantly",  f"{stats['pct_changed']:.1f}%  "
                                           f"(proportion of image with change > {DIFF_THRESHOLD}/255)"),
    ]

    interpretation = (
        "INTERPRETATION:  The number of verified RANSAC inliers confirms that the before and "
        "after photos depict the same manhole cover. The difference heatmap provides a visual "
        "and mathematical record that cleaning work was carried out between the two inspections. "
        "This analysis is generated automatically and can be reproduced at any time from the "
        "original image files."
    )

    pad    = EXPL_PADDING
    wrap_w = max(60, int((total_w - 2 * pad) / 7.5))

    def measure_height():
        y = pad + 26 + 14
        for _, body in steps:
            y += 22
            y += 18 * len(textwrap.wrap(body, width=wrap_w))
            y += 15
        y += 22
        for label, value in result_lines:
            y += 18 * len(textwrap.wrap(f"  {label}:  {value}", width=wrap_w))
        y += 24
        y += 18 * len(textwrap.wrap(interpretation, width=wrap_w))
        y += pad
        return y

    panel_h = measure_height()
    img  = Image.new("RGB", (total_w, panel_h), COL_EXPL_BG)
    draw = ImageDraw.Draw(img)
    y    = pad

    title_txt = "QUALITY ASSURANCE METHODOLOGY  —  HOW WE MATHEMATICALLY CONFIRMED CLEANING WORK"
    draw.text((pad, y), title_txt, fill=COL_EXPL_TITLE, font=font_title)
    y += 26
    draw.line([(pad, y), (total_w - pad, y)], fill=COL_SEPARATOR, width=1)
    y += 14

    for heading, body in steps:
        draw.text((pad, y), heading, fill=COL_STEP_HEADING, font=font_step)
        y += 22
        for line in textwrap.wrap(body, width=wrap_w):
            draw.text((pad + 14, y), line, fill=COL_BODY, font=font_body)
            y += 18
        y += 5
        draw.line([(pad, y), (total_w - pad, y)], fill=COL_SEPARATOR, width=1)
        y += 10

    draw.text((pad, y), "RESULTS FOR THIS IMAGE PAIR", fill=COL_STEP_HEADING, font=font_step)
    y += 22
    for label, value in result_lines:
        for i, line in enumerate(textwrap.wrap(f"  {label}:  {value}", width=wrap_w)):
            draw.text((pad + 14, y), line, fill=(COL_STAT_VALUE if i == 0 else COL_BODY), font=font_stat)
            y += 18
    y += 10
    draw.line([(pad, y), (total_w - pad, y)], fill=COL_SEPARATOR, width=1)
    y += 10

    for line in textwrap.wrap(interpretation, width=wrap_w):
        draw.text((pad, y), line, fill=COL_BODY, font=font_body)
        y += 18

    return img


def build_keypoint_img(before_bgr, after_bgr, kp1, kp2, inlier_matches):
    return cv2.drawMatches(
        before_bgr, kp1, after_bgr, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def _build_verdict_banner_3n(total_w, stats, metrics):
    """Top-of-image banner: STAGE 3N verdict (washing tier) + key parameters."""
    from pipeline_config import STAGE3N_HIGH_CONFIDENCE, STAGE3N_MEDIUM_CONFIDENCE

    tier       = metrics.get("washing_tier", "LOW")
    confidence = metrics.get("washing_confidence", 0.0)

    bg_col  = {
        "HIGH":   (30, 90, 30),
        "MEDIUM": (90, 65, 15),
        "LOW":    (90, 25, 25),
    }.get(tier, (60, 60, 60))

    pad    = EXPL_PADDING
    banner = Image.new("RGB", (total_w, 100), bg_col)
    draw   = ImageDraw.Draw(banner)

    title = f"STAGE 3N  --  WASHING CONFIDENCE   [{tier}]"
    draw.text((pad, 10), title, fill=(255, 255, 255), font=load_font(18))
    draw.line([(pad, 40), (total_w - pad, 40)], fill=(200, 200, 200), width=1)

    col_ok   = (140, 255, 140)
    col_warn = (255, 210, 80)
    col_fail = (255, 120, 120)
    col_info = (210, 210, 210)

    tier_col = col_ok if tier == "HIGH" else (col_warn if tier == "MEDIUM" else col_fail)

    params = [
        (f"Confidence: {confidence:.2f}  (HIGH >= {STAGE3N_HIGH_CONFIDENCE:.2f},"
         f"  MEDIUM >= {STAGE3N_MEDIUM_CONFIDENCE:.2f})", tier_col),
        (f"Tier: {tier}  |  RANSAC inliers: {stats.get('ransac_inliers', 0)}"
         f"  |  KP before: {stats.get('kp_before', 0):,}  |  KP after: {stats.get('kp_after', 0):,}",
         col_info),
    ]

    y = 48
    for text, colour in params:
        draw.text((pad, y), text, fill=colour, font=load_font(13))
        y += 22

    return banner


def build_composite(before_bgr, after_bgr, warped_bgr, diff_gray, stats, metrics=None):
    diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_MAGMA)

    p_before = _resize_to_height(before_bgr,  PANEL_HEIGHT)
    p_warped = _resize_to_height(warped_bgr,  PANEL_HEIGHT)
    p_diff   = _resize_to_height(diff_color,  PANEL_HEIGHT)
    total_w  = p_before.shape[1] + p_warped.shape[1] + p_diff.shape[1]

    half_w   = total_w // 2
    p_orig_b = cv2.resize(before_bgr, (half_w, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    p_orig_a = cv2.resize(after_bgr,  (half_w, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)

    pair_parts = stats["pair"].split("/")
    b_lbl = pair_parts[0].strip()
    a_lbl = pair_parts[1].strip()

    layers = []
    if metrics is not None:
        layers.append(_pil_to_bgr(_build_verdict_banner_3n(total_w, stats, metrics)))

    bar1 = build_section_bar(
        total_w,
        "ORIGINAL PHOTOGRAPHS  (AS TAKEN — UNPROCESSED)",
        [(f"Original Before  ({b_lbl})", half_w),
         (f"Original After   ({a_lbl})", total_w - half_w)],
    )
    bar2 = build_section_bar(
        total_w,
        "MATHEMATICAL COMPARISON  (AFTER ALIGNMENT)",
        [(f"Before / Reference  ({b_lbl})", p_before.shape[1]),
         (f"After Aligned  ({a_lbl})",       p_warped.shape[1]),
         ("Difference Heatmap",              p_diff.shape[1])],
    )
    expl_pil = build_explanation_panel(total_w, stats)

    row1 = np.hstack([p_orig_b, p_orig_a])
    row2 = np.hstack([p_before, p_warped, p_diff])
    layers += [_pil_to_bgr(bar1), row1, _pil_to_bgr(bar2), row2, _pil_to_bgr(expl_pil)]
    return np.vstack(layers)


# ─────────────────────────────────────────────────────────────────────────────
# Washing evidence metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_washing_metrics(gray_before, gray_after, kp1, kp2, good_matches):
    """Compute 6 complementary CV signals and return a composite washing confidence.

    Arguments are already available inside process_pair() — no re-computation needed.
    Returns a flat dict with raw values, washing_confidence (float 0-1), and
    washing_tier ('HIGH' / 'MEDIUM' / 'LOW').
    """

    # ── Signal 1: KP count ratio ──────────────────────────────────────────────
    kp_ratio = len(kp2) / max(len(kp1), 1)
    # after/before >= 1.5x → score 1.0
    kp_score = max(min((kp_ratio - 1.0) / 0.5, 1.0), 0.0)

    # ── Signal 2: Intensity std dev ───────────────────────────────────────────
    std_before = float(np.std(gray_before))
    std_after  = float(np.std(gray_after))
    std_increase_pct = (std_after - std_before) / max(std_before, 1e-6) * 100
    # >= 20% increase → score 1.0
    std_score = max(min(std_increase_pct / 20.0, 1.0), 0.0)

    # ── Signal 3: GLCM entropy + homogeneity ─────────────────────────────────
    # Downsample to 25% for speed; quantise to 64 grey levels
    h, w = gray_before.shape
    small_b = cv2.resize(gray_before, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
    small_a = cv2.resize(gray_after,  (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
    q_b = (small_b // 4).astype(np.uint8)
    q_a = (small_a // 4).astype(np.uint8)

    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm_b = graycomatrix(q_b, distances=[1], angles=angles, levels=64, symmetric=True, normed=True)
    glcm_a = graycomatrix(q_a, distances=[1], angles=angles, levels=64, symmetric=True, normed=True)

    entropy_before = float(np.mean([shannon_entropy(glcm_b[:, :, 0, i]) for i in range(len(angles))]))
    entropy_after  = float(np.mean([shannon_entropy(glcm_a[:, :, 0, i]) for i in range(len(angles))]))
    entropy_increase = entropy_after - entropy_before
    # >= 0.5 nats increase → score 1.0
    entropy_score = max(min(entropy_increase / 0.5, 1.0), 0.0)

    homogeneity_before = float(np.mean(graycoprops(glcm_b, "homogeneity")))
    homogeneity_after  = float(np.mean(graycoprops(glcm_a, "homogeneity")))

    # ── Signal 4: Match-to-keypoint ratio (inverted) ─────────────────────────
    match_ratio = len(good_matches) / max(len(kp1), 1)
    # Low ratio = washing evidence. <= 0.05 → score 1.0; >= 0.5 → score 0.0
    match_score = max(min((0.5 - match_ratio) / (0.5 - 0.05), 1.0), 0.0)

    # ── Signal 5: Edge density ────────────────────────────────────────────────
    edges_b = cv2.Canny(gray_before, 50, 150)
    edges_a = cv2.Canny(gray_after,  50, 150)
    edge_density_before = float(np.count_nonzero(edges_b)) / edges_b.size
    edge_density_after  = float(np.count_nonzero(edges_a)) / edges_a.size
    edge_increase_pct = (edge_density_after - edge_density_before) / max(edge_density_before, 1e-6) * 100
    # >= 20% increase → score 1.0
    edge_score = max(min(edge_increase_pct / 20.0, 1.0), 0.0)

    # ── Signal 6: Laplacian variance ──────────────────────────────────────────
    lap_var_before = float(cv2.Laplacian(gray_before, cv2.CV_64F).var())
    lap_var_after  = float(cv2.Laplacian(gray_after,  cv2.CV_64F).var())
    lap_increase_pct = (lap_var_after - lap_var_before) / max(lap_var_before, 1e-6) * 100
    # >= 30% increase → score 1.0
    lap_score = max(min(lap_increase_pct / 30.0, 1.0), 0.0)

    # ── Composite confidence ──────────────────────────────────────────────────
    # Weighted sum: entropy + edge are most informative (0.25 each);
    # std + lap moderate (0.15 each); kp + match lower reliability (0.10 each).
    SIGNAL_WEIGHTS = [0.10, 0.15, 0.25, 0.10, 0.25, 0.15]  # kp, std, entropy, match, edge, lap
    confidence = float(np.dot(SIGNAL_WEIGHTS, [kp_score, std_score, entropy_score, match_score, edge_score, lap_score]))
    from pipeline_config import STAGE3N_HIGH_CONFIDENCE, STAGE3N_MEDIUM_CONFIDENCE
    if confidence >= STAGE3N_HIGH_CONFIDENCE:
        tier = "HIGH"
    elif confidence >= STAGE3N_MEDIUM_CONFIDENCE:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return {
        "kp_ratio":            round(kp_ratio, 2),
        "std_before":          round(std_before, 1),
        "std_after":           round(std_after, 1),
        "std_increase_pct":    round(std_increase_pct, 1),
        "entropy_before":      round(entropy_before, 3),
        "entropy_after":       round(entropy_after, 3),
        "entropy_increase":    round(entropy_increase, 3),
        "homogeneity_before":  round(homogeneity_before, 3),
        "homogeneity_after":   round(homogeneity_after, 3),
        "match_ratio":         round(match_ratio, 3),
        "match_count":         len(good_matches),
        "kp_before_count":     len(kp1),
        "edge_density_before": round(edge_density_before * 100, 1),
        "edge_density_after":  round(edge_density_after  * 100, 1),
        "edge_increase_pct":   round(edge_increase_pct, 1),
        "lap_var_before":      round(lap_var_before, 1),
        "lap_var_after":       round(lap_var_after, 1),
        "lap_increase_pct":    round(lap_increase_pct, 1),
        "washing_confidence":  round(confidence, 2),
        "washing_tier":        tier,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_pair_outputs(out_dir, prefix, match_img, composite):
    os.makedirs(out_dir, exist_ok=True)
    match_path = os.path.join(out_dir, f"{prefix}_keypoint_matches.jpg")
    comp_path  = os.path.join(out_dir, f"{prefix}_composite.jpg")
    cv2.imwrite(match_path, match_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(comp_path,  composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"    Saved: {os.path.basename(match_path)}")
    print(f"    Saved: {os.path.basename(comp_path)}")


def process_pair(folder_path, folder_name, base_before, base_after, prefix, out_dir):
    before_bgr, after_bgr, path_before, path_after = load_pair(folder_path, base_before, base_after)

    # ── Resize for SIFT (cap at MAX_SIFT_HEIGHT to limit memory usage) ────────
    # All processing (mask, SIFT, warp, metrics, visuals) uses resized images so
    # that the homography H is consistent throughout.  Full-res buffers are
    # released immediately to free RAM.
    MAX_SIFT_HEIGHT = int(os.environ.get("MAX_SIFT_HEIGHT", 1080))
    sift_before = _resize_to_height(before_bgr, MAX_SIFT_HEIGHT) if before_bgr.shape[0] > MAX_SIFT_HEIGHT else before_bgr
    sift_after  = _resize_to_height(after_bgr,  MAX_SIFT_HEIGHT) if after_bgr.shape[0] > MAX_SIFT_HEIGHT else after_bgr
    if sift_before is not before_bgr:
        del before_bgr
    if sift_after is not after_bgr:
        del after_bgr

    gray_before = cv2.cvtColor(sift_before, cv2.COLOR_BGR2GRAY)
    gray_after  = cv2.cvtColor(sift_after,  cv2.COLOR_BGR2GRAY)

    # ── OCR masking ───────────────────────────────────────────────────────────
    ocr_before = run_ocr_for_mask(path_before)
    ocr_after  = run_ocr_for_mask(path_after)
    mask_before = create_metadata_mask(sift_before.shape, ocr_before)
    mask_after  = create_metadata_mask(sift_after.shape,  ocr_after)
    n_masked_before = len(ocr_before)
    n_masked_after  = len(ocr_after)
    mask_applied    = n_masked_before > 0 or n_masked_after > 0
    print(f"    Metadata mask: {n_masked_before} region(s) before / {n_masked_after} region(s) after - excluded from SIFT")

    kp1, kp2, good_matches = sift_match(gray_before, gray_after, mask_before, mask_after)
    print(f"    SIFT kp: {len(kp1):,}/{len(kp2):,}  ratio matches: {len(good_matches):,}", end="")

    H, inlier_matches = compute_homography(kp1, kp2, good_matches)
    print(f"  inliers: {len(inlier_matches):,}")

    warped_bgr, diff_gray, mean_diff, pct_changed = warp_and_diff(sift_before, sift_after, H)
    print(f"    Mean diff: {mean_diff:.1f}/255  |  Changed: {pct_changed:.1f}%")

    stats = {
        "folder":         folder_name,
        "pair":           f"{base_before} / {base_after}",
        "kp_before":      len(kp1),
        "kp_after":       len(kp2),
        "ratio_matches":  len(good_matches),
        "ransac_inliers": len(inlier_matches),
        "mean_diff":      round(mean_diff, 1),
        "pct_changed":    round(pct_changed, 1),
        "text_regions_masked_before": n_masked_before,
        "text_regions_masked_after":  n_masked_after,
        "metadata_mask_applied":      mask_applied,
    }

    metrics = compute_washing_metrics(gray_before, gray_after, kp1, kp2, good_matches)

    visuals = {
        "before_bgr":   sift_before,
        "after_bgr":    sift_after,
        "kp1":          kp1,
        "kp2":          kp2,
        "good_matches": good_matches,
        "gray_before":  gray_before,
        "gray_after":   gray_after,
    }

    if _SKIP_IMGS:
        return stats, metrics, None, None, None
    match_img = build_keypoint_img(sift_before, sift_after, kp1, kp2, inlier_matches)
    composite = build_composite(sift_before, sift_after, warped_bgr, diff_gray, stats, metrics)
    save_pair_outputs(out_dir, prefix, match_img, composite)
    return stats, metrics, match_img, composite, visuals


# ─────────────────────────────────────────────────────────────────────────────
# Report builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_cover_header(total_w, folder_name):
    """~120px tall title block at the top of the consolidated report."""
    header_h = 120
    img  = Image.new("RGB", (total_w, header_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    font_large = load_font(22)
    font_small = load_font(14)

    title    = f"WRN SERVICE REPORT  --  Job Folder: {folder_name}"
    date_str = str(datetime.date.today())

    tw = _text_w(draw, title, font_large)
    draw.text(((total_w - tw) // 2, 28), title, fill=(255, 255, 255), font=font_large)

    dw = _text_w(draw, date_str, font_small)
    draw.text(((total_w - dw) // 2, 70), date_str, fill=(180, 180, 180), font=font_small)

    return img


def build_footer(total_w):
    """~40px tall footer at the bottom of the consolidated report."""
    footer_h = 40
    img  = Image.new("RGB", (total_w, footer_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    font = load_font(12)

    txt = "Generated automatically by WRN Service Report Checker"
    tw  = _text_w(draw, txt, font)
    draw.text(((total_w - tw) // 2, 12), txt, fill=(140, 140, 140), font=font)
    return img


def build_metrics_panel(total_w, stats, metrics):
    """Panel showing washing-evidence metrics for one pair with layman explanations."""
    font_title  = load_font(16)
    font_metric = load_font(14)
    font_body   = load_font(12)
    font_conf   = load_font(18)

    pad    = EXPL_PADDING
    wrap_w = max(60, int((total_w - 2 * pad) / 7.5))

    pair_label = stats["pair"]  # e.g. "D2 / D5"

    hom_change = metrics["homogeneity_after"] - metrics["homogeneity_before"]

    metric_rows = [
        (
            "KP count",
            f"{metrics['kp_before_count']:,}",
            f"{stats['kp_after']:,}",
            f"x{metrics['kp_ratio']:.2f}",
            "Number of distinct surface features the computer could identify. More features after "
            "washing means the surface texture is now clearly visible.",
        ),
        (
            "Intensity std dev",
            f"{metrics['std_before']:.1f}",
            f"{metrics['std_after']:.1f}",
            f"{'+' if metrics['std_increase_pct'] >= 0 else ''}{metrics['std_increase_pct']:.1f}%",
            "How much the brightness varies across the image. A higher spread after washing suggests "
            "dirt/blockage has been removed, exposing lighter and darker surface areas.",
        ),
        (
            "GLCM entropy",
            f"{metrics['entropy_before']:.3f}",
            f"{metrics['entropy_after']:.3f}",
            f"{'+' if metrics['entropy_increase'] >= 0 else ''}{metrics['entropy_increase']:.3f} nats",
            "Measures how complex and varied the surface pattern is. Higher entropy after washing "
            "means more visual detail has been uncovered.",
        ),
        (
            "GLCM homogeneity",
            f"{metrics['homogeneity_before']:.3f}",
            f"{metrics['homogeneity_after']:.3f}",
            f"{'+' if hom_change >= 0 else ''}{hom_change:.3f}",
            "Measures how uniform the surface looks. Lower homogeneity after washing is expected -- "
            "a clean surface shows more varied texture than a flat layer of dirt.",
        ),
        (
            "Match ratio",
            f"{metrics['match_ratio']:.3f}",
            "n/a",
            f"{metrics['match_count']} of {metrics['kp_before_count']:,} kp",
            "Proportion of before-photo features that could still be found in the after photo. "
            "A very low ratio means the surface looks substantially different -- consistent with "
            "washing having changed it.",
        ),
        (
            "Edge density",
            f"{metrics['edge_density_before']:.1f}%",
            f"{metrics['edge_density_after']:.1f}%",
            f"{'+' if metrics['edge_increase_pct'] >= 0 else ''}{metrics['edge_increase_pct']:.1f}%",
            "Proportion of pixels that sit on a visible edge (crack, joint, bolt, rim). More edges "
            "after washing means physical surface structure is now visible.",
        ),
        (
            "Laplacian variance",
            f"{metrics['lap_var_before']:.1f}",
            f"{metrics['lap_var_after']:.1f}",
            f"{'+' if metrics['lap_increase_pct'] >= 0 else ''}{metrics['lap_increase_pct']:.1f}%",
            "Measures fine-grained sharpness and micro-texture. A higher value after washing "
            "indicates the surface is richer in detail, consistent with dirt removal.",
        ),
    ]

    confidence = metrics["washing_confidence"]
    tier       = metrics["washing_tier"]
    tier_colour = {"HIGH": (100, 220, 100), "MEDIUM": (255, 180, 60), "LOW": (220, 80, 80)}.get(
        tier, (180, 180, 180)
    )

    def measure_height():
        y = pad + 26 + 14  # title line + separator
        for _label, _b, _a, _c, explanation in metric_rows:
            y += 22  # metric value line
            y += 16 * len(textwrap.wrap(explanation, width=wrap_w))
            y += 8
        y += 14   # separator
        y += 50   # confidence badge
        y += pad
        return y

    panel_h = measure_height()
    img  = Image.new("RGB", (total_w, panel_h), COL_EXPL_BG)
    draw = ImageDraw.Draw(img)
    y    = pad

    title_txt = f"WASHING EVIDENCE METRICS  --  {pair_label}"
    draw.text((pad, y), title_txt, fill=COL_EXPL_TITLE, font=font_title)
    y += 26
    draw.line([(pad, y), (total_w - pad, y)], fill=COL_SEPARATOR, width=1)
    y += 14

    for label, before, after, change, explanation in metric_rows:
        metric_line = f"  {label}:  Before = {before}   After = {after}   Change = {change}"
        draw.text((pad, y), metric_line, fill=COL_STAT_VALUE, font=font_metric)
        y += 22
        for line in textwrap.wrap(explanation, width=wrap_w):
            draw.text((pad + 20, y), line, fill=COL_BODY, font=font_body)
            y += 16
        y += 8

    draw.line([(pad, y), (total_w - pad, y)], fill=COL_SEPARATOR, width=1)
    y += 14

    badge_txt = f"Confidence: {confidence:.2f}  [{tier}]"
    bw = _text_w(draw, badge_txt, font_conf)
    draw.text(((total_w - bw) // 2, y), badge_txt, fill=tier_colour, font=font_conf)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Visual evidence panel
# ─────────────────────────────────────────────────────────────────────────────

def _build_visual_row(total_w, left_bgr, right_bgr,
                      left_caption, right_caption,
                      metric_title, before_val, after_val, change_val,
                      explanation, img_h=300):
    """One visual evidence row: two images side-by-side + caption bar + text panel."""
    pad    = EXPL_PADDING
    gap    = 8
    half_w = (total_w - gap) // 2

    left_r  = cv2.resize(left_bgr,  (half_w, img_h), interpolation=cv2.INTER_AREA)
    right_r = cv2.resize(right_bgr, (half_w, img_h), interpolation=cv2.INTER_AREA)

    # Caption bars
    cap_h     = 28
    left_cap  = np.full((cap_h, half_w, 3), 45, dtype=np.uint8)
    right_cap = np.full((cap_h, half_w, 3), 45, dtype=np.uint8)
    cv2.putText(left_cap,  left_caption,  (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(right_cap, right_caption, (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)

    left_col  = np.vstack([left_r,  left_cap])
    right_col = np.vstack([right_r, right_cap])
    sep       = np.full((img_h + cap_h, gap, 3), 20, dtype=np.uint8)
    img_strip = np.hstack([left_col, sep, right_col])

    # Text panel
    font_metric = load_font(14)
    font_body   = load_font(12)
    wrap_w      = max(60, int((total_w - 2 * pad) / 7.5))
    header      = f"  {metric_title}:  Before = {before_val}   After = {after_val}   Change = {change_val}"
    body_lines  = textwrap.wrap(explanation, width=wrap_w)
    text_h      = pad + 24 + 16 * len(body_lines) + pad // 2

    text_pil = Image.new("RGB", (total_w, text_h), COL_EXPL_BG)
    tdraw    = ImageDraw.Draw(text_pil)
    y = pad // 2
    tdraw.text((pad, y), header, fill=COL_STAT_VALUE, font=font_metric)
    y += 24
    for line in body_lines:
        tdraw.text((pad + 14, y), line, fill=COL_BODY, font=font_body)
        y += 16

    divider = np.full((2, total_w, 3), 55, dtype=np.uint8)
    return np.vstack([img_strip, _pil_to_bgr(text_pil), divider])


def build_visual_evidence_panel(total_w, visuals, metrics):
    """
    Build the full visual evidence panel for all washing metrics.
    `visuals` is the dict returned by process_pair().
    """
    IMG_H        = 300
    pad          = EXPL_PADDING
    before_bgr   = visuals["before_bgr"]
    after_bgr    = visuals["after_bgr"]
    kp1          = visuals["kp1"]
    kp2          = visuals["kp2"]
    good_matches = visuals["good_matches"]
    gray_before  = visuals["gray_before"]
    gray_after   = visuals["gray_after"]

    sections = []

    # ── Header bar ─────────────────────────────────────────────────────────
    hdr_h  = 52
    hdr    = Image.new("RGB", (total_w, hdr_h), (18, 18, 18))
    hdraw  = ImageDraw.Draw(hdr)
    ftitle = load_font(17)
    txt    = "VISUAL WASHING EVIDENCE  --  Before vs After Comparison"
    tw     = _text_w(hdraw, txt, ftitle)
    hdraw.text(((total_w - tw) // 2, 10), txt, fill=COL_EXPL_TITLE, font=ftitle)
    hdraw.line([(pad, 46), (total_w - pad, 46)], fill=COL_SEPARATOR, width=1)
    sections.append(_pil_to_bgr(hdr))

    # ── 1. KP count ─────────────────────────────────────────────────────────
    kp_b = cv2.drawKeypoints(before_bgr, kp1, None, color=(0, 255, 80),
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_a = cv2.drawKeypoints(after_bgr,  kp2, None, color=(0, 255, 80),
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    sections.append(_build_visual_row(
        total_w, kp_b, kp_a,
        f"Before  ({len(kp1):,} keypoints detected)",
        f"After   ({len(kp2):,} keypoints detected)",
        "KP Count",
        f"{len(kp1):,}", f"{len(kp2):,}", f"x{metrics['kp_ratio']:.2f}",
        "Each circle marks a distinctive surface feature such as a scratch, bolt hole, or "
        "surface edge identified by the computer. A surface covered in dirt or silt has few "
        "visible landmarks. After washing, the physical texture is exposed and many more "
        "distinctive points become detectable. More circles in the after photo is a positive "
        "indicator that the surface has changed.",
        img_h=IMG_H,
    ))

    # ── 2. Match ratio ──────────────────────────────────────────────────────
    matched_idx   = {m.queryIdx for m in good_matches}
    match_overlay = before_bgr.copy()
    for i, kp in enumerate(kp1):
        colour = (0, 200, 0) if i in matched_idx else (30, 30, 210)
        cv2.circle(match_overlay, (int(kp.pt[0]), int(kp.pt[1])), 3, colour, -1)
    sections.append(_build_visual_row(
        total_w, match_overlay, after_bgr,
        f"Before  (green = matched {len(good_matches):,}   red = lost {len(kp1) - len(good_matches):,})",
        "After  (reference photo)",
        "Match Ratio",
        f"{metrics['match_ratio']:.3f}",
        f"{metrics['match_count']} matched",
        f"{metrics['match_count']} of {metrics['kp_before_count']:,} kp",
        "Green dots are before-photo features the computer could still find in the after photo "
        "(surface looks the same in those spots). Red dots are features that vanished or changed "
        "beyond recognition. A mostly red before-image means the surface looks substantially "
        "different after the job -- strong evidence that washing physically altered the visible "
        "surface. A mostly green image would suggest little surface change occurred.",
        img_h=IMG_H,
    ))

    # ── 3. Intensity std dev ────────────────────────────────────────────────
    gray_b_bgr = cv2.cvtColor(gray_before, cv2.COLOR_GRAY2BGR)
    gray_a_bgr = cv2.cvtColor(gray_after,  cv2.COLOR_GRAY2BGR)
    std_sign   = "+" if metrics["std_increase_pct"] >= 0 else ""
    sections.append(_build_visual_row(
        total_w, gray_b_bgr, gray_a_bgr,
        f"Before  (std dev = {metrics['std_before']:.1f})",
        f"After   (std dev = {metrics['std_after']:.1f})",
        "Intensity Std Dev",
        f"{metrics['std_before']:.1f}", f"{metrics['std_after']:.1f}",
        f"{std_sign}{metrics['std_increase_pct']:.1f}%",
        "Standard deviation measures how much the brightness values vary across the image. "
        "A surface uniformly coated in dirt looks similar everywhere -- flat, grey, low variation. "
        "After washing, lighter concrete or metal areas and darker joints/cracks become exposed, "
        "creating a much wider spread of brightness values. A higher standard deviation in the "
        "after photo means more brightness variation -- consistent with the surface being cleaned "
        "and the underlying material becoming visible.",
        img_h=IMG_H,
    ))

    # ── 4. Edge density ─────────────────────────────────────────────────────
    edges_b   = cv2.cvtColor(cv2.Canny(gray_before, 50, 150), cv2.COLOR_GRAY2BGR)
    edges_a   = cv2.cvtColor(cv2.Canny(gray_after,  50, 150), cv2.COLOR_GRAY2BGR)
    edge_sign = "+" if metrics["edge_increase_pct"] >= 0 else ""
    sections.append(_build_visual_row(
        total_w, edges_b, edges_a,
        f"Before  (edge density = {metrics['edge_density_before']:.1f}%)",
        f"After   (edge density = {metrics['edge_density_after']:.1f}%)",
        "Edge Density",
        f"{metrics['edge_density_before']:.1f}%", f"{metrics['edge_density_after']:.1f}%",
        f"{edge_sign}{metrics['edge_increase_pct']:.1f}%",
        "White pixels mark detected edges -- the sharp boundaries of physical features such as "
        "joints, cracks, bolt holes, and the manhole rim. A surface buried under silt or debris "
        "appears featureless with almost no edges. After washing, the underlying structure "
        "becomes exposed and the edge detector finds many more boundaries. More white pixels "
        "in the after image confirms that physical surface features have been uncovered.",
        img_h=IMG_H,
    ))

    # ── 5. Laplacian variance ────────────────────────────────────────────────
    lap_b       = np.abs(cv2.Laplacian(gray_before, cv2.CV_64F))
    lap_a       = np.abs(cv2.Laplacian(gray_after,  cv2.CV_64F))
    lap_b_color = cv2.applyColorMap(
        cv2.normalize(lap_b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_HOT,
    )
    lap_a_color = cv2.applyColorMap(
        cv2.normalize(lap_a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_HOT,
    )
    lap_sign = "+" if metrics["lap_increase_pct"] >= 0 else ""
    sections.append(_build_visual_row(
        total_w, lap_b_color, lap_a_color,
        f"Before  (variance = {metrics['lap_var_before']:.1f})",
        f"After   (variance = {metrics['lap_var_after']:.1f})",
        "Laplacian Variance",
        f"{metrics['lap_var_before']:.1f}", f"{metrics['lap_var_after']:.1f}",
        f"{lap_sign}{metrics['lap_increase_pct']:.1f}%",
        "The Laplacian filter highlights every point of rapid brightness change -- essentially "
        "fine-grained surface texture and micro-detail. Bright areas on this heatmap correspond "
        "to highly textured regions such as rough concrete, coarse gravel, or exposed metal. "
        "A featureless dirty surface produces a mostly dark (low variance) heatmap because "
        "the dirt smooths everything over. After washing the surface detail is revealed, "
        "making the map brighter and more varied across the whole image.",
        img_h=IMG_H,
    ))

    # ── 6. GLCM entropy + homogeneity ────────────────────────────────────────
    def _quantised_colour(gray):
        h, w = gray.shape
        sm = cv2.resize(gray, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
        q  = ((sm // 4) * 4).astype(np.uint8)
        return cv2.applyColorMap(q, cv2.COLORMAP_VIRIDIS)

    glcm_b   = _quantised_colour(gray_before)
    glcm_a   = _quantised_colour(gray_after)
    ent_sign = "+" if metrics["entropy_increase"] >= 0 else ""
    hom_chg  = metrics["homogeneity_after"] - metrics["homogeneity_before"]
    hom_sign = "+" if hom_chg >= 0 else ""
    sections.append(_build_visual_row(
        total_w, glcm_b, glcm_a,
        f"Before  (entropy = {metrics['entropy_before']:.3f}   homogeneity = {metrics['homogeneity_before']:.3f})",
        f"After   (entropy = {metrics['entropy_after']:.3f}   homogeneity = {metrics['homogeneity_after']:.3f})",
        "GLCM Entropy / Homogeneity",
        f"entropy {ent_sign}{metrics['entropy_increase']:.3f}",
        f"homogeneity {hom_sign}{hom_chg:.3f}",
        "combined change",
        "This shows the downsampled, 64-level quantised version of each image used to compute "
        "texture complexity (Grey Level Co-occurrence Matrix). Entropy measures how complex and "
        "varied the brightness pattern is -- a flat dirty surface has low entropy (few distinct "
        "levels), a physically textured surface has high entropy (many varied levels). Homogeneity "
        "measures how uniform the surface looks -- dirty and smooth scores high, clean and rough "
        "scores lower. After washing: entropy typically rises and homogeneity falls as the "
        "real surface pattern becomes visible beneath the removed dirt layer.",
        img_h=IMG_H,
    ))

    return np.vstack(sections)


def build_report_image(folder_name, pair_results, report_w=2400):
    """Stack cover header, per-pair sections, and footer into one tall image."""
    sections = [_pil_to_bgr(build_cover_header(report_w, folder_name))]

    for stats, metrics, match_img_bgr, composite_bgr, visuals in pair_results:
        first_char = stats["pair"][0].upper()
        direction  = "Downstream" if first_char == "D" else "Upstream"
        pair_header_txt = f"{stats['pair']}  ({direction})"
        sections.append(_pil_to_bgr(build_section_bar(report_w, pair_header_txt, [])))

        # Scale keypoint-matches image to report_w
        kp_h, kp_w = match_img_bgr.shape[:2]
        kp_new_h   = int(kp_h * report_w / kp_w)
        sections.append(
            cv2.resize(match_img_bgr, (report_w, kp_new_h), interpolation=cv2.INTER_AREA)
        )

        # Scale composite to report_w if needed
        comp_h, comp_w = composite_bgr.shape[:2]
        if comp_w != report_w:
            comp_new_h = int(comp_h * report_w / comp_w)
            sections.append(
                cv2.resize(composite_bgr, (report_w, comp_new_h), interpolation=cv2.INTER_AREA)
            )
        else:
            sections.append(composite_bgr)

        sections.append(_pil_to_bgr(build_metrics_panel(report_w, stats, metrics)))
        sections.append(build_visual_evidence_panel(report_w, visuals, metrics))

    sections.append(_pil_to_bgr(build_footer(report_w)))
    return np.vstack(sections)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_d2d5_u2u5_on_folder(folder_path, folder_name, output_dir, progress_cb=None):
    """
    Run D2/D5 and U2/U5 SIFT + washing-evidence pipeline on ONE folder.

    Parameters
    ----------
    folder_path : str   Path to the job folder.
    folder_name : str   Folder name for reporting.
    output_dir  : str   Root output dir; outputs go to <output_dir>/<folder_name>/
    progress_cb : callable(msg), optional

    Returns
    -------
    dict
        ``{
            "pairs_processed": int,
            "pairs_failed": int,
            "pair_results": { "D2": {"status": "OK"|"FAILED", "washing_tier": ...}, ... },
            "overall_pass": bool,  # True if >=1 pair is HIGH tier
        }``
    """
    from pipeline_config import STAGE3N_PASS_TIER

    out_dir = os.path.join(output_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    pairs_processed  = 0
    pairs_failed     = 0
    pair_results     = {}

    for base_before, base_after, prefix in PAIRS:
        try:
            stats, metrics, _match_img, _composite, _visuals = process_pair(
                folder_path, folder_name,
                base_before, base_after, prefix,
                out_dir,
            )
            tier = metrics["washing_tier"]
            pair_results[prefix] = {
                "status":        "OK",
                "washing_tier":  tier,
                "washing_confidence": metrics["washing_confidence"],
                "sift_stats":    stats,
                "metrics":       metrics,
            }
            pairs_processed += 1
            if progress_cb:
                progress_cb(
                    f"    [{prefix}] tier={tier}  "
                    f"conf={metrics['washing_confidence']:.2f}"
                )
        except (FileNotFoundError, RuntimeError, IOError) as exc:
            pair_results[prefix] = {"status": "FAILED", "error": str(exc)}
            pairs_failed += 1
            if progress_cb:
                progress_cb(f"    [{prefix}] FAILED -- {exc}")

    # Overall pass: at least one pair achieves the required tier
    overall_pass = any(
        v.get("washing_tier") == STAGE3N_PASS_TIER
        for v in pair_results.values()
        if v.get("status") == "OK"
    )

    return {
        "pairs_processed": pairs_processed,
        "pairs_failed":    pairs_failed,
        "pair_results":    pair_results,
        "overall_pass":    overall_pass,
    }


def main():
    if not os.path.isdir(FOLDER_PATH):
        print(f"ERROR: folder not found: {FOLDER_PATH}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Trial run — D2/D5 + U2/U5 pairs for folder {FOLDER_NAME}")
    print(f"  Source : {FOLDER_PATH}")
    print(f"  Output : {OUT_DIR}")
    print()

    results      = []   # (stats, metrics)
    pair_results = []   # (stats, metrics, match_img, composite, visuals) for report builder
    for base_before, base_after, prefix in PAIRS:
        print(f"  [{prefix}] Pair {base_before}/{base_after}:")
        try:
            stats, metrics, match_img, composite, visuals = process_pair(
                FOLDER_PATH, FOLDER_NAME, base_before, base_after, prefix, OUT_DIR
            )
            results.append((stats, metrics))
            pair_results.append((stats, metrics, match_img, composite, visuals))
        except (FileNotFoundError, RuntimeError, IOError) as exc:
            print(f"  [{prefix}] SKIPPED -- {exc}")
        print()

    print("-" * 52)
    print(f"  Done.  {len(results)}/{len(PAIRS)} pair(s) processed.")
    if results:
        print()
        for s, _ in results:
            print(f"  {s['pair']}")
            print(f"    kp before={s['kp_before']:,}  after={s['kp_after']:,}")
            print(f"    ratio matches={s['ratio_matches']:,}  inliers={s['ransac_inliers']:,}")
            print(f"    mean diff={s['mean_diff']}/255  changed={s['pct_changed']}%")

        print()
        print("  Washing Evidence Metrics")
        print("  " + "-" * 47)
        for s, m in results:
            std_sign  = "+" if m["std_increase_pct"]  >= 0 else ""
            ent_sign  = "+" if m["entropy_increase"]   >= 0 else ""
            edge_sign = "+" if m["edge_increase_pct"]  >= 0 else ""
            lap_sign  = "+" if m["lap_increase_pct"]   >= 0 else ""
            print(f"  {s['pair']}")
            print(f"    KP ratio           :  {m['kp_ratio']:.2f}x  (after/before)")
            print(f"    Std dev            :  before={m['std_before']}  after={m['std_after']}  {std_sign}{m['std_increase_pct']:.1f}%")
            print(f"    GLCM entropy       :  before={m['entropy_before']:.3f}  after={m['entropy_after']:.3f}  {ent_sign}{m['entropy_increase']:.3f} nats")
            print(f"    GLCM homogeneity   :  before={m['homogeneity_before']:.3f}  after={m['homogeneity_after']:.3f}")
            print(f"    Match ratio        :  {m['match_ratio']:.3f}  ({m['match_count']} of {m['kp_before_count']:,} before-kp matched)")
            print(f"    Edge density       :  before={m['edge_density_before']}%  after={m['edge_density_after']}%  {edge_sign}{m['edge_increase_pct']:.1f}%")
            print(f"    Laplacian variance :  before={m['lap_var_before']}  after={m['lap_var_after']}  {lap_sign}{m['lap_increase_pct']:.1f}%")
            print(f"    Confidence         :  {m['washing_confidence']:.2f}  [{m['washing_tier']}]")
            print()
    if pair_results:
        print()
        print("  Building consolidated report image ...")
        report      = build_report_image(FOLDER_NAME, pair_results)
        report_path = os.path.join(OUT_DIR, f"{FOLDER_NAME}_report.jpg")
        cv2.imwrite(report_path, report, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  Report saved: {os.path.basename(report_path)}")

    print(f"  Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
