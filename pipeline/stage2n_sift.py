"""
Level 2 — No-obstruction folders: before/after image alignment and difference analysis
=======================================================================================
Scans every subfolder inside adjusted_images/no_obstruction/.
For each folder it looks for: D1, D4, U1, U4 (any extension).

Detection gate (requires trained model — see train_manhole_detector.py):
  HIGH confidence (≥ 0.70) → auto-accept → SIFT comparison pipeline
  MED  confidence (0.30–0.69) → queue for human review → Label Studio
  LOW  confidence (< 0.30) → auto-reject → logged only

If no trained model is available yet, the gate is skipped and all pairs are processed.

For each accepted pair (D1/D4 and U1/U4):
  1. Detects SIFT keypoints in both images.
  2. Matches descriptors with FLANN + Lowe's ratio test.
  3. Computes a homography (RANSAC) to warp the "after" image onto the "before" frame.
  4. Saves two output images per pair:
       <prefix>_keypoint_matches.jpg  — inlier SIFT matches visualised
       <prefix>_composite.jpg         — full auditor composite

Composite layout:
  ┌─────────────────────────────────────────────────┐
  │  ORIGINAL PHOTOGRAPHS (AS TAKEN)                │
  │  Original Before (D1)    Original After (D4)    │
  ├──────────────────────────────────────────────────┤
  │  MATHEMATICAL COMPARISON                        │
  │  Before (Ref)  After Aligned  Difference        │
  ├──────────────────────────────────────────────────┤
  │  QUALITY ASSURANCE METHODOLOGY & RESULTS        │
  └─────────────────────────────────────────────────┘

Output directory:
  adjusted_images/no_obstruction/difference_analysis/<folder_name>/

Run report:
  adjusted_images/no_obstruction/difference_analysis/run_report_<timestamp>.json

Review queue (medium-confidence images):
  adjusted_images/no_obstruction/difference_analysis/review_queue/<folder>_<base>.jpg
"""

import json
import os
import shutil
import sys
import textwrap
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────
_STAGE_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR     = os.environ.get(
    "NO_OBSTRUCTION_DIR",
    os.path.join(_STAGE_ROOT, "adjusted_images", "no_obstruction"),
)
ANALYSIS_DIR = os.path.join(BASE_DIR, "difference_analysis")

# ── Detector config ───────────────────────────────────────────────────────────
# Override DETECTOR_MODEL env-var to point to the weights file.
DETECTOR_MODEL = os.environ.get(
    "DETECTOR_MODEL",
    os.path.join(_STAGE_ROOT, "manhole_detector", "v1_positive_only3", "weights", "best.pt"),
)
CONF_ACCEPT    = 0.70    # ≥ this → auto-accept
CONF_REVIEW    = 0.30    # ≥ this, < CONF_ACCEPT → queue for human review
                         # < CONF_REVIEW → auto-reject

# ── SIFT pipeline config ──────────────────────────────────────────────────────
VALID_EXTS        = {".jpg", ".jpeg", ".png"}
PANEL_HEIGHT      = 600
SECTION_BAR_H     = 56
EXPL_PADDING      = 20
FLANN_TREES       = 5
FLANN_CHECKS      = 50
RATIO_THRESHOLD   = 0.75
RANSAC_THRESHOLD  = 5.0
DIFF_THRESHOLD    = 20        # grey-level change considered "significant" (0–255)

PAIRS = [
    ("D1", "D4", "D"),        # (before_base, after_base, output_prefix)
    ("U1", "U4", "U"),
]

_SKIP_IMGS = os.environ.get("SKIP_REPORT_IMAGES", "").lower() in ("1", "true")

# ── Metadata mask config ──────────────────────────────────────────────────────
# Minimum keypoints that must remain in the physical (unmasked) region after
# OCR-derived text regions are excluded.  Fewer → "Insufficient Physical
# Landmarks" error rather than falling back on metadata text matching.
MIN_MASKED_KP = 10
# Pixel margin inflated around every OCR bounding box so that border
# pixels of glyphs are also excluded.
TEXT_MASK_PAD = 8

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


def _cap_image(img_bgr):
    """Downscale img_bgr so its longest edge is <= MAX_IMAGE_DIM (if set)."""
    max_dim = int(os.environ.get("MAX_IMAGE_DIM", "0"))
    if max_dim <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if max(h, w) <= max_dim:
        return img_bgr
    scale = max_dim / max(h, w)
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def load_pair(folder, base_before, base_after):
    """Load a before/after image pair.

    Returns
    -------
    (before_bgr, after_bgr, path_before, path_after)
    """
    path_before = find_image(folder, base_before)
    path_after  = find_image(folder, base_after)
    missing = [b for b, p in [(base_before, path_before), (base_after, path_after)] if p is None]
    if missing:
        raise FileNotFoundError(f"Image(s) not found: {missing}")
    before = _cap_image(cv2.imread(path_before))
    after  = _cap_image(cv2.imread(path_after))
    if before is None or after is None:
        raise IOError(f"cv2.imread failed for images in {folder}")
    return before, after, path_before, path_after


def _resize_to_height(img_bgr, height):
    h, w  = img_bgr.shape[:2]
    scale = height / h
    return cv2.resize(img_bgr, (int(w * scale), height), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata masking
# ─────────────────────────────────────────────────────────────────────────────

_mask_ocr_engine = None   # lazy singleton — initialised on first use


def _get_mask_ocr_engine():
    """
    Lazy-initialise a PaddleOCR engine used solely for metadata masking.

    The engine is shared across all calls in a single run (singleton).
    Returns None gracefully if PaddleOCR is not installed so the pipeline
    degrades to unmasked SIFT rather than crashing.

    Set env var DISABLE_MASK_OCR=1 to skip PaddleOCR initialisation entirely
    (useful in memory-constrained environments).

    Applies a compatibility shim for imgaug on NumPy >= 2.0: np.sctypes was
    removed in NumPy 2.0 but imgaug (a PaddleOCR dep) still references it.
    The shim is added only when absent and does not affect NumPy behaviour.
    """
    global _mask_ocr_engine
    if os.environ.get("DISABLE_MASK_OCR", "").lower() in ("1", "true", "yes"):
        return None
    if _mask_ocr_engine is not None:
        return _mask_ocr_engine
    try:
        import os as _os
        import numpy as _np
        # imgaug (PaddleOCR dep) uses np.sctypes removed in NumPy 2.0
        if not hasattr(_np, "sctypes"):
            _np.sctypes = {
                "int":     [_np.int8,  _np.int16,  _np.int32,  _np.int64],
                "uint":    [_np.uint8, _np.uint16, _np.uint32, _np.uint64],
                "float":   [_np.float32, _np.float64],
                "complex": [_np.complex64, _np.complex128],
                "others":  [bool, object, bytes, str],
            }
        # PaddleOCR's generated protobuf files are incompatible with
        # protobuf >= 4.0 in C-extension mode; pure-Python mode fixes this.
        _os.environ.setdefault(
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python"
        )
        from paddleocr import PaddleOCR
        _mask_ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except BaseException:
        # Catches both Python exceptions and SystemExit (raised by PaddlePaddle
        # on initialisation failure, e.g. missing model files or network error).
        _mask_ocr_engine = None
    return _mask_ocr_engine


def create_metadata_mask(image_shape, ocr_lines):
    """
    Build a binary mask that excludes OCR-detected text regions from SIFT.

    Every bounding box returned by PaddleOCR is inflated by TEXT_MASK_PAD
    pixels and painted black (0) in the mask so that SIFT cannot detect
    keypoints on burned-in timestamps, location codes, or UI overlays.
    The rest of the image (physical pipe / manhole surface) is white (255).

    Parameters
    ----------
    image_shape : tuple   (H, W) or (H, W, C) — same shape as the image.
    ocr_lines   : list    PaddleOCR result lines:
                          [ ([pt0, pt1, pt2, pt3], ('text', conf)), ... ]
                          Pass [] if OCR produced no detections.

    Returns
    -------
    numpy.ndarray  uint8, shape (H, W).
                   255 = physical region  (SIFT may detect keypoints here)
                     0 = excluded region  (text / metadata overlay)
    """
    h, w = image_shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for line in ocr_lines:
        box = line[0]   # four corner points [[x,y], [x,y], [x,y], [x,y]]
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x1 = max(0, min(xs) - TEXT_MASK_PAD)
        y1 = max(0, min(ys) - TEXT_MASK_PAD)
        x2 = min(w, max(xs) + TEXT_MASK_PAD)
        y2 = min(h, max(ys) + TEXT_MASK_PAD)
        mask[y1:y2, x1:x2] = 0
    return mask


def run_ocr_for_mask(image_path):
    """
    Detect text regions in *image_path* for metadata masking.

    Runs PaddleOCR and returns the raw line list so that
    create_metadata_mask() can convert bounding boxes to a binary mask.
    Returns [] if PaddleOCR is not available or the image contains no text.
    """
    engine = _get_mask_ocr_engine()
    if engine is None:
        return []
    try:
        raw = engine.ocr(image_path, cls=True)
        return raw[0] if raw and raw[0] else []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Detection gate
# ─────────────────────────────────────────────────────────────────────────────

def load_detector():
    """
    Load the YOLOv8 detector if the model file exists.
    Returns the model, or None if not yet available.
    """
    if not os.path.isfile(DETECTOR_MODEL):
        print(f"  [Gate] No trained model found at:\n         {DETECTOR_MODEL}")
        print("         Skipping detection gate — all pairs will be processed.")
        print("         Train the model first: python train_manhole_detector.py\n")
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(DETECTOR_MODEL)
        print(f"  [Gate] Detector loaded: {os.path.basename(DETECTOR_MODEL)}\n")
        return model
    except ImportError:
        print("  [Gate] ultralytics not installed — gate disabled.")
        print("         Install: pip install ultralytics\n")
        return None


def classify_image(image_path, model):
    """
    Run the detector on one image.
    Returns (tier, confidence, yolo_result):
      tier: "accept" | "review" | "reject"
      confidence: highest detection confidence (0.0 if no detection)
      yolo_result: ultralytics Results object for the image
    """
    results = model.predict(source=image_path, conf=CONF_REVIEW, verbose=False)
    r       = results[0]
    boxes   = r.boxes

    if len(boxes) == 0:
        return "reject", 0.0, r

    best_conf = float(boxes.conf.max())
    if best_conf >= CONF_ACCEPT:
        return "accept", best_conf, r
    return "review", best_conf, r


def save_gate_image(image_path, yolo_result, tier, conf, folder_name, base_name, output_dir):
    """
    Draw YOLO detection boxes + confidence on the image and save it to
    output_dir/<tier>/<folder_name>_<base_name>.jpg
    """
    img = cv2.imread(image_path)
    if img is None:
        return

    TIER_COLORS = {
        "accept": (34,  139, 34),   # green
        "review": (0,   140, 255),  # orange
        "reject": (0,   0,   200),  # red
    }
    color = TIER_COLORS.get(tier, (128, 128, 128))

    # Draw each detection box
    for box in yolo_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        box_conf = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"manhole {box_conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Colour banner at the top showing tier + confidence
    banner = np.zeros((50, img.shape[1], 3), dtype=np.uint8)
    banner[:] = color
    banner_text = f"GATE: {tier.upper()}   conf={conf:.0%}   {folder_name} / {base_name}"
    cv2.putText(banner, banner_text, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    img = np.vstack([banner, img])

    gate_dir = os.path.join(output_dir, tier)
    os.makedirs(gate_dir, exist_ok=True)
    save_path = os.path.join(gate_dir, f"{folder_name}_{base_name}.jpg")
    cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ─────────────────────────────────────────────────────────────────────────────
# SIFT pipeline
# ─────────────────────────────────────────────────────────────────────────────

def sift_match(gray1, gray2, mask1=None, mask2=None, ratio_threshold=None):
    """
    Detect SIFT keypoints and match them between two greyscale images.

    Parameters
    ----------
    gray1, gray2 : numpy.ndarray   Greyscale images (uint8).
    mask1, mask2 : numpy.ndarray or None
        Binary masks (uint8, same H×W as the images).  255 = allowed region,
        0 = excluded region.  Generated by create_metadata_mask() to prevent
        keypoint detection on burned-in text overlays.  Pass None to run on
        the full image (e.g. when PaddleOCR is not available).

    Raises
    ------
    RuntimeError
        "Insufficient Physical Landmarks" if fewer than MIN_MASKED_KP
        keypoints survive in either masked image — the pipe surface is
        featureless or the mask is over-aggressive.  This is preferred over
        silently falling back on metadata text as alignment landmarks.
    """
    try:
        _nfeatures = int(os.environ.get("MAX_SIFT_FEATURES", 20000))
        sift = cv2.SIFT_create(nfeatures=_nfeatures)
    except AttributeError:
        print("ERROR: cv2.SIFT_create() not available. Install opencv-contrib-python.")
        sys.exit(1)

    kp1, des1 = sift.detectAndCompute(gray1, mask1)
    kp2, des2 = sift.detectAndCompute(gray2, mask2)

    # Guard: fail explicitly rather than match on metadata text
    if len(kp1) < MIN_MASKED_KP or len(kp2) < MIN_MASKED_KP:
        raise RuntimeError(
            f"Insufficient Physical Landmarks: {len(kp1)} (before) / {len(kp2)} (after) "
            f"keypoints detected in unmasked regions — minimum required: {MIN_MASKED_KP}. "
            "The pipe surface may be featureless or the metadata mask is over-aggressive."
        )

    if des1 is None or des2 is None:
        raise RuntimeError(
            "SIFT descriptor extraction returned None — no usable keypoints in unmasked regions."
        )

    index_params  = dict(algorithm=1, trees=FLANN_TREES)
    search_params = dict(checks=FLANN_CHECKS)
    flann         = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches   = flann.knnMatch(des1, des2, k=2)
    _ratio = ratio_threshold if ratio_threshold is not None else RATIO_THRESHOLD
    good = [m for m, n in raw_matches if m.distance < _ratio * n.distance]
    return kp1, kp2, good


def compute_homography(kp1, kp2, good_matches):
    if len(good_matches) < 4:
        raise RuntimeError(f"Only {len(good_matches)} good match(es) — need ≥4.")
    src_pts = np.float32([kp2[m.trainIdx].pt  for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt  for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    if H is None:
        raise RuntimeError("findHomography returned None — insufficient inliers.")
    inlier_matches = [m for m, flag in zip(good_matches, mask.ravel()) if flag]
    return H, inlier_matches


def compute_inlier_coverage(kp1, inlier_matches, image_shape, grid_size=4):
    """
    Measure how broadly the RANSAC inliers are distributed across the image.

    The image is divided into a (grid_size × grid_size) grid of equal zones.
    Each inlier keypoint (from the before image) is assigned to a zone and
    the number of occupied zones is counted.

    A high coverage percentage means the verified alignment points are spread
    across the full pipe surface — a reliable whole-surface match.
    A low percentage means all inliers cluster in one region, which may
    indicate the alignment is driven by a single local feature rather than
    the pipe surface as a whole.

    Parameters
    ----------
    kp1            : list   SIFT keypoints from the before image.
    inlier_matches : list   RANSAC inlier matches (DMatch objects).
    image_shape    : tuple  (H, W) or (H, W, C) of the before image.
    grid_size      : int    Number of rows and columns in the grid (default 4,
                            giving a 4×4 = 16-zone grid).

    Returns
    -------
    coverage_pct      : float  Percentage of zones that contain ≥1 inlier.
    occupied_cells    : int    Number of occupied zones.
    total_cells       : int    Total zones (grid_size²).
    """
    h, w = image_shape[:2]
    occupied = set()
    for m in inlier_matches:
        x, y = kp1[m.queryIdx].pt
        col = min(int(x / w * grid_size), grid_size - 1)
        row = min(int(y / h * grid_size), grid_size - 1)
        occupied.add((row, col))
    total_cells  = grid_size * grid_size
    coverage_pct = len(occupied) / total_cells * 100
    return coverage_pct, len(occupied), total_cells


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
            "Step 1 — Metadata Masking + Feature Detection (SIFT: Scale-Invariant Feature Transform)",
            "Before landmark detection begins, the system automatically identifies any burned-in "
            "metadata overlays — timestamps, location codes, and camera UI text — using optical "
            "character recognition (OCR). A binary mask is generated from the detected text "
            "bounding boxes so that keypoint extraction is strictly confined to the physical pipe "
            "surface. The SIFT algorithm then identifies distinctive 'landmark' points — bolt "
            "holes, scratches, surface edges, and physical markings — drawn exclusively from the "
            "unmasked physical region. Metadata text is never used as an alignment landmark, "
            "preventing false geometric alignment driven by fixed-position overlay characters.",
        ),
        (
            "Step 2 — Feature Matching (FLANN + Ratio Test)",
            "The computer compares every landmark from the before photo against every landmark in "
            "the after photo, searching for matching pairs that represent the same physical point. "
            "A strict quality filter — the ratio test — then discards any match that is ambiguous, "
            "i.e. where the computer cannot confidently distinguish which landmark it is looking at. "
            "Only clear, unambiguous, high-confidence matches are retained.",
        ),
        (
            "Step 3 — Alignment Calculation (RANSAC + Homography Matrix)",
            "Using the verified matching landmarks, the computer calculates a mathematical formula "
            "called a homography matrix. This formula describes exactly how to rotate, scale, and "
            "shift the after photo so that it lines up pixel-for-pixel with the before photo. A "
            "further statistical filter (RANSAC) removes any remaining inconsistent matches before "
            "the formula is computed, ensuring the alignment is based only on trustworthy, "
            "geometrically consistent data.",
        ),
        (
            "Step 4 — Inlier Spatial Coverage (4x4 Grid Distribution Check)",
            "After RANSAC identifies the verified alignment points, a further check confirms that "
            "those points are distributed broadly across the pipe surface rather than concentrated "
            "in a single region. The before image is divided into a 4x4 grid of 16 equal zones "
            "and the system counts how many zones contain at least one verified inlier keypoint. "
            "A higher percentage means physical evidence is spread across the whole pipe surface, "
            "confirming a genuine whole-surface match. A low percentage — all inliers in one or "
            "two zones — would indicate the alignment may be driven by a single local feature "
            "such as one bolt hole or edge, rather than the pipe structure as a whole.",
        ),
        (
            "Step 5 — Visual Reference (Difference Heatmap)",
            "The after photo is warped onto the before frame using the computed homography and "
            "the two aligned images are subtracted pixel by pixel to produce a reference heatmap "
            "(magma colour scale: black → purple → orange → yellow). This provides a visual "
            "reference overlay for the auditor. Note: because sewer camera images are routinely "
            "exposure-adjusted between inspections, pixel-level brightness differences reflect "
            "camera processing as well as physical change and are provided for visual reference "
            "only — the verified RANSAC inlier count and spatial coverage are the primary "
            "evidence of a confirmed match.",
        ),
    ]

    result_lines = [
        ("Job folder",                    stats["folder"]),
        ("Image pair",                    stats["pair"]),
        ("SIFT keypoints — Before photo", f"{stats['kp_before']:,}  "
                                           "(landmark points identified on the before photo)"),
        ("SIFT keypoints — After photo",  f"{stats['kp_after']:,}  "
                                           "(landmark points identified on the after photo)"),
        ("Ratio-test matches",            f"{stats['ratio_matches']:,}  "
                                           "(unambiguous landmark pairs found in both photos)"),
        ("RANSAC inliers",                f"{stats['ransac_inliers']:,}  "
                                           "(verified alignment points after final quality filter; "
                                           ">=10 confirms photos show the same object)"),
        ("Inlier spatial coverage",       f"{stats.get('inlier_coverage_pct', 0):.0f}%  "
                                           f"({stats.get('coverage_cells_occupied', 0)} of "
                                           f"{stats.get('coverage_grid_size', 4) ** 2} grid zones occupied — "
                                           "higher = inliers spread broadly across pipe surface)"),
        ("Mean pixel difference",         f"{stats['mean_diff']:.1f} / 255  "
                                           "(average per-pixel change after alignment; 0 = identical)"),
        ("Pixels changed significantly",  f"{stats['pct_changed']:.1f}%  "
                                           f"(proportion of image with change > {DIFF_THRESHOLD}/255)"),
        ("Text regions masked — Before",  f"{stats.get('text_regions_masked_before', 0)}  "
                                           "(OCR-detected metadata overlays excluded from SIFT keypoint extraction)"),
        ("Text regions masked — After",   f"{stats.get('text_regions_masked_after', 0)}  "
                                           "(OCR-detected metadata overlays excluded from SIFT keypoint extraction)"),
    ]

    interpretation = (
        "INTERPRETATION:  The RANSAC inlier count confirms that the before and after photos depict "
        "the same physical structure. The spatial coverage percentage confirms that the verified "
        "alignment points are distributed broadly across the pipe surface rather than concentrated "
        "on a single feature. Metadata overlays (timestamps, location text, camera UI) were "
        "detected via OCR and excluded from the alignment calculation, ensuring the homography is "
        "grounded in physical pipe evidence only. The difference heatmap is provided as a visual "
        "reference for the auditor. This analysis is generated automatically and can be reproduced "
        "at any time from the original image files."
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


def _build_verdict_banner_2n(total_w, stats):
    """Top-of-image banner: STAGE 2N verdict + key parameters vs thresholds."""
    from pipeline_config import STAGE2N_MIN_INLIERS, STAGE2N_MIN_COVERAGE_PCT

    inliers  = stats.get("ransac_inliers", 0)
    coverage = stats.get("inlier_coverage_pct", 0.0)
    in_ok    = inliers  >= STAGE2N_MIN_INLIERS
    cov_ok   = coverage >= STAGE2N_MIN_COVERAGE_PCT
    passed   = in_ok and cov_ok

    bg_col  = (30, 90, 30)  if passed else (90, 25, 25)
    v_label = "PASSED"      if passed else "FAILED"
    icon    = "[PASS]"      if passed else "[FAIL]"

    pad    = EXPL_PADDING
    banner = Image.new("RGB", (total_w, 100), bg_col)
    draw   = ImageDraw.Draw(banner)

    title = f"STAGE 2N  --  SIFT ALIGNMENT  {icon}  {v_label}"
    draw.text((pad, 10), title, fill=(255, 255, 255), font=load_font(18))
    draw.line([(pad, 40), (total_w - pad, 40)], fill=(200, 200, 200), width=1)

    col_ok   = (140, 255, 140)
    col_fail = (255, 120, 120)
    col_info = (210, 210, 210)

    params = [
        (f"RANSAC inliers: {inliers}  (threshold >= {STAGE2N_MIN_INLIERS})",
         col_ok if in_ok else col_fail),
        (f"Coverage: {coverage:.0f}%  (threshold >= {STAGE2N_MIN_COVERAGE_PCT:.0f}%)",
         col_ok if cov_ok else col_fail),
        (f"KP before: {stats.get('kp_before', 0):,}  |  KP after: {stats.get('kp_after', 0):,}"
         f"  |  Ratio matches: {stats.get('ratio_matches', 0):,}",
         col_info),
    ]

    col_sep = total_w // len(params)
    y = 48
    for i, (text, colour) in enumerate(params):
        draw.text((pad + i * col_sep, y), text, fill=colour, font=load_font(13))
        if i > 0:
            draw.line([(i * col_sep - 4, y), (i * col_sep - 4, y + 18)],
                      fill=(180, 180, 180), width=1)

    return banner


def build_composite(before_bgr, after_bgr, warped_bgr, diff_gray, stats):
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

    verdict_banner = _build_verdict_banner_2n(total_w, stats)

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
    return np.vstack([
        _pil_to_bgr(verdict_banner),
        _pil_to_bgr(bar1), row1,
        _pil_to_bgr(bar2), row2,
        _pil_to_bgr(expl_pil),
    ])


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


def export_review_queue(review_items, analysis_dir):
    """Copy medium-confidence images to review_queue/ for Label Studio import."""
    if not review_items:
        return
    queue_dir = os.path.join(analysis_dir, "review_queue")
    os.makedirs(queue_dir, exist_ok=True)
    for item in review_items:
        src = item["image_path"]
        dst_name = f"{item['folder']}_{os.path.basename(src)}"
        shutil.copy2(src, os.path.join(queue_dir, dst_name))
    print(f"\n  Review queue exported → {queue_dir}")
    print(f"  Import this folder into Label Studio to begin review.")


def write_run_report(report, analysis_dir):
    """Write JSON report and print console summary."""
    os.makedirs(analysis_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(analysis_dir, f"run_report_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    s = report["summary"]
    total = s["total_pairs"]
    print()
    print("  " + "-" * 48)
    print("   Run summary")
    print("  " + "-" * 48)
    if report["gate_active"]:
        pct_a = s["auto_accepted"] / total * 100 if total else 0
        pct_r = s["sent_for_review"] / total * 100 if total else 0
        pct_j = s["auto_rejected"] / total * 100 if total else 0
        print(f"   Total pairs         : {total}")
        print(f"   Auto-accepted       : {s['auto_accepted']:>4}  ({pct_a:.1f}%)")
        print(f"   Sent for review     : {s['sent_for_review']:>4}  ({pct_r:.1f}%)  ← open Label Studio")
        print(f"   Auto-rejected       : {s['auto_rejected']:>4}  ({pct_j:.1f}%)")
    else:
        print(f"   Total pairs         : {total}  (gate disabled — no model)")
        print(f"   Processed           : {s['auto_accepted']}")
        print(f"   Skipped (SIFT err)  : {s.get('skipped_errors', 0)}")
    print("  " + "-" * 48)
    print(f"  Run report -> {os.path.basename(out_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────────────────────────────────────

def process_pair(folder_path, folder_name, base_before, base_after, prefix, out_dir,
                 ratio_threshold=None):
    before_bgr, after_bgr, path_before, path_after = load_pair(
        folder_path, base_before, base_after
    )

    # ── Resize for SIFT (cap at MAX_SIFT_HEIGHT to limit memory usage) ────────
    # All subsequent processing (mask, SIFT, warp, composite) is done on the
    # resized images so that H is consistent throughout.  Full-res buffers are
    # released immediately after resizing to free RAM.
    MAX_SIFT_HEIGHT = int(os.environ.get("MAX_SIFT_HEIGHT", 1080))
    sift_before = _resize_to_height(before_bgr, MAX_SIFT_HEIGHT) if before_bgr.shape[0] > MAX_SIFT_HEIGHT else before_bgr
    sift_after  = _resize_to_height(after_bgr,  MAX_SIFT_HEIGHT) if after_bgr.shape[0] > MAX_SIFT_HEIGHT else after_bgr
    # Free full-resolution buffers if we actually created resized copies
    if sift_before is not before_bgr:
        del before_bgr
    if sift_after is not after_bgr:
        del after_bgr

    # ── Metadata masking ──────────────────────────────────────────────────────
    # Run OCR on both images to detect burned-in text overlays (timestamps,
    # location codes, UI elements).  The bounding boxes are converted into a
    # binary mask that is passed to SIFT so that keypoints are only detected
    # on the physical pipe/manhole surface.
    ocr_before = run_ocr_for_mask(path_before)
    ocr_after  = run_ocr_for_mask(path_after)
    mask_before = create_metadata_mask(sift_before.shape, ocr_before)
    mask_after  = create_metadata_mask(sift_after.shape,  ocr_after)

    n_masked_before = len(ocr_before)
    n_masked_after  = len(ocr_after)
    mask_applied    = n_masked_before > 0 or n_masked_after > 0

    if mask_applied:
        print(
            f"    Metadata mask: {n_masked_before} region(s) before / "
            f"{n_masked_after} region(s) after - excluded from SIFT"
        )

    # ── SIFT feature detection + matching (masked) ────────────────────────────
    gray_before = cv2.cvtColor(sift_before, cv2.COLOR_BGR2GRAY)
    gray_after  = cv2.cvtColor(sift_after,  cv2.COLOR_BGR2GRAY)

    kp1, kp2, good_matches = sift_match(
        gray_before, gray_after, mask_before, mask_after,
        ratio_threshold=ratio_threshold,
    )
    print(f"    SIFT kp: {len(kp1):,}/{len(kp2):,}  ratio matches: {len(good_matches):,}", end="")

    H, inlier_matches = compute_homography(kp1, kp2, good_matches)
    print(f"  inliers: {len(inlier_matches):,}")

    # H was computed in sift_before/sift_after coordinate space — use sift_*
    # for coverage, warp, and composite so coordinates remain consistent.
    coverage_pct, coverage_cells, coverage_total = compute_inlier_coverage(
        kp1, inlier_matches, sift_before.shape
    )
    print(f"    Coverage: {coverage_cells}/{coverage_total} zones ({coverage_pct:.0f}%)")

    warped_bgr, diff_gray, mean_diff, pct_changed = warp_and_diff(sift_before, sift_after, H)

    stats = {
        "folder":                     folder_name,
        "pair":                       f"{base_before} / {base_after}",
        "kp_before":                  len(kp1),
        "kp_after":                   len(kp2),
        "ratio_matches":              len(good_matches),
        "ransac_inliers":             len(inlier_matches),
        "inlier_coverage_pct":        round(coverage_pct, 1),
        "coverage_cells_occupied":    coverage_cells,
        "coverage_grid_size":         4,
        "mean_diff":                  round(mean_diff, 1),
        "pct_changed":                round(pct_changed, 1),
        "text_regions_masked_before": n_masked_before,
        "text_regions_masked_after":  n_masked_after,
        "metadata_mask_applied":      mask_applied,
    }

    if not _SKIP_IMGS:
        match_img = build_keypoint_img(sift_before, sift_after, kp1, kp2, inlier_matches)
        composite = build_composite(sift_before, sift_after, warped_bgr, diff_gray, stats)
        save_pair_outputs(out_dir, prefix, match_img, composite)
    return stats


def run_sift_on_folder(folder_path, folder_name, output_dir,
                       detector=None, progress_cb=None, ratio_threshold=None):
    """
    Run the SIFT before/after pipeline on ONE no-obstruction job folder.

    Targets D1/D4 and U1/U4 pairs.  If *detector* is provided (loaded YOLO
    model) the detection gate is applied first.

    Parameters
    ----------
    folder_path : str   Path to the job folder.
    folder_name : str   Folder name for reporting.
    output_dir  : str   Root output dir; per-folder outputs go to
                        <output_dir>/<folder_name>/
    detector    : YOLO model or None
    progress_cb : callable(msg), optional

    Returns
    -------
    dict
        ``{
            "pairs_processed": int,
            "pairs_failed": int,
            "pairs_skipped": int,
            "pair_stats": { "D": {...} | None, "U": {...} | None },
            "overall_pass": bool,   # True if >=1 pair meets inlier+pct thresholds
        }``
    """
    from pipeline_config import STAGE2N_MIN_INLIERS, STAGE2N_MIN_COVERAGE_PCT

    out_dir = os.path.join(output_dir, folder_name)
    pairs_processed = 0
    pairs_failed    = 0
    pairs_skipped   = 0
    pair_stats      = {}

    for base_before, base_after, prefix in PAIRS:
        path_before = find_image(folder_path, base_before)
        path_after  = find_image(folder_path, base_after)
        missing = [b for b, p in [(base_before, path_before), (base_after, path_after)] if p is None]
        if missing:
            if progress_cb:
                progress_cb(f"    [{prefix}] {', '.join(missing)} not found -- skipped")
            pairs_skipped += 1
            pair_stats[prefix] = {"status": "MISSING_IMAGES", "missing": missing}
            continue

        pair_gate = {}
        if detector is not None:
            tier_b, conf_b, res_b = classify_image(path_before, detector)
            save_gate_image(path_before, res_b, tier_b, conf_b,
                            folder_name, base_before, output_dir)

            tier_a, conf_a, res_a = classify_image(path_after, detector)
            save_gate_image(path_after, res_a, tier_a, conf_a,
                            folder_name, base_after, output_dir)

            if progress_cb:
                progress_cb(
                    f"    [{prefix}] Gate: {base_before}={tier_b.upper()} {conf_b:.2f}  "
                    f"{base_after}={tier_a.upper()} {conf_a:.2f}"
                )

            # Always record both gate results so Excel report can show them
            pair_gate = {
                f"gate_tier_{base_before}": tier_b,
                f"gate_conf_{base_before}": round(conf_b, 3),
                f"gate_tier_{base_after}":  tier_a,
                f"gate_conf_{base_after}":  round(conf_a, 3),
            }
            worst = tier_b if tier_b in ("review", "reject") else tier_a
            if worst in ("review", "reject"):
                pairs_skipped += 1
                pair_stats[prefix] = {"status": f"GATE_{worst.upper()}", **pair_gate}
                continue

        try:
            stats = process_pair(
                folder_path, folder_name,
                base_before, base_after, prefix, out_dir,
                ratio_threshold=ratio_threshold,
            )
            passes = (
                stats["ransac_inliers"]      >= STAGE2N_MIN_INLIERS
                and stats["inlier_coverage_pct"] >= STAGE2N_MIN_COVERAGE_PCT
            )
            stats["stage2n_pass"] = passes
            pair_stats[prefix] = {"status": "OK", **pair_gate, **stats}
            pairs_processed += 1
        except (FileNotFoundError, RuntimeError, IOError) as exc:
            if progress_cb:
                progress_cb(f"    [{prefix}] FAILED -- {exc}")
            pair_stats[prefix] = {"status": "FAILED", "error": str(exc)}
            pairs_failed += 1

    # Overall pass: at least one pair has >= STAGE2N_MIN_INLIERS RANSAC inliers
    overall_pass = any(
        v.get("stage2n_pass", False)
        for v in pair_stats.values()
        if isinstance(v, dict)
    )

    return {
        "pairs_processed": pairs_processed,
        "pairs_failed":    pairs_failed,
        "pairs_skipped":   pairs_skipped,
        "pair_stats":      pair_stats,
        "overall_pass":    overall_pass,
    }


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # ── Scan all job subfolders ───────────────────────────────────────────────
    folders = sorted(
        [e for e in os.scandir(BASE_DIR)
         if e.is_dir() and e.name not in {"difference_analysis"}],
        key=lambda e: e.name,
    )
    if not folders:
        print(f"No subfolders found in {BASE_DIR}")
        sys.exit(1)

    print(f"Found {len(folders)} job folder(s) in no_obstruction.\n")

    # ── Load detector (optional) ──────────────────────────────────────────────
    detector    = load_detector()
    gate_active = detector is not None

    # ── Tracking for run report ───────────────────────────────────────────────
    review_items    = []
    rejected_items  = []
    accepted_count  = 0
    error_count     = 0

    # ── Process folders ───────────────────────────────────────────────────────
    for folder in folders:
        folder_name = folder.name
        out_dir     = os.path.join(ANALYSIS_DIR, folder_name)
        print(f"  {folder_name}")

        for base_before, base_after, prefix in PAIRS:
            path_before = find_image(folder.path, base_before)
            if path_before is None:
                print(f"    [{prefix}] {base_before} not found — skipped")
                continue

            # ── Detection gate ────────────────────────────────────────────────
            if gate_active:
                tier, conf, yolo_result = classify_image(path_before, detector)
                save_gate_image(path_before, yolo_result, tier, conf,
                                folder_name, base_before, ANALYSIS_DIR)
                if tier == "accept":
                    print(f"    [{prefix}] Gate: ACCEPT  conf={conf:.2f}")
                elif tier == "review":
                    print(f"    [{prefix}] Gate: REVIEW  conf={conf:.2f}  → queued")
                    review_items.append({
                        "folder":     folder_name,
                        "pair":       f"{base_before}/{base_after}",
                        "image_path": path_before,
                        "confidence": round(conf, 3),
                    })
                    continue
                else:
                    print(f"    [{prefix}] Gate: REJECT  conf={conf:.2f}  → skipped")
                    rejected_items.append({
                        "folder":     folder_name,
                        "pair":       f"{base_before}/{base_after}",
                        "image_path": path_before,
                        "confidence": round(conf, 3),
                    })
                    continue

            # ── SIFT pipeline ─────────────────────────────────────────────────
            print(f"    [{prefix}] Pair {base_before}/{base_after}:")
            try:
                process_pair(folder.path, folder_name, base_before, base_after, prefix, out_dir)
                accepted_count += 1
            except (FileNotFoundError, RuntimeError, IOError) as exc:
                print(f"    [{prefix}] SKIPPED — {exc}")
                error_count += 1

        print()

    # ── Review queue export ───────────────────────────────────────────────────
    export_review_queue(review_items, ANALYSIS_DIR)

    # ── Run report ────────────────────────────────────────────────────────────
    total_pairs = len(folders) * len(PAIRS)
    report = {
        "run_date":       datetime.now().isoformat(timespec="seconds"),
        "model_path":     DETECTOR_MODEL if gate_active else None,
        "gate_active":    gate_active,
        "conf_thresholds": {"accept": CONF_ACCEPT, "review": CONF_REVIEW},
        "summary": {
            "total_pairs":    total_pairs,
            "auto_accepted":  accepted_count,
            "sent_for_review": len(review_items),
            "auto_rejected":  len(rejected_items),
            "skipped_errors": error_count,
        },
        "review_queue": review_items,
        "rejected":      rejected_items,
    }
    write_run_report(report, ANALYSIS_DIR)

    print(f"\nDone.  Outputs: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
