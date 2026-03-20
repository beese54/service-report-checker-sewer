"""
Contact sheet — D1/D4 + U1/U4 SIFT failures.
Each row = one folder, 4 image columns + parameters panel.

PASS CRITERIA (both must be met per pair):
  RANSAC inliers >= 10   (confirms same physical location)
  Cover %        >= 25%  (inliers spread across pipe surface, not clustered)
A folder passes Stage 1 if at least one pair (D or U) satisfies both.

Results are cached in sift_metrics_cache.json so re-runs are fast.
"""

import os, sys, json, glob, traceback
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker"
ORIG_DIR     = os.path.join(BASE, "original_images")
REPORT_DIR   = os.path.join(BASE, "adjusted_images", "no_obstruction", "difference_analysis")
CACHE_FILE   = os.path.join(BASE, "sift_metrics_cache.json")
OUT_FILE     = os.path.join(BASE, "failed_D1D4_U1U4_contact_sheet.jpg")

# ── Pass/fail thresholds (from pipeline_config.py) ───────────────────────────
MIN_INLIERS   = 10
MIN_COVERAGE  = 25.0

FAILED_FOLDERS = [
    "1159099","1159101","1159103","140883","152593","152602","152604",
    "152605","157252","157253","160155","172131","172142","172179",
    "172303","172311","178210","182216","1852128","185223","185230",
    "185243","187805","187806","187807","187808","187810","187813",
    "187815","190464","204032","204033","204034","206646",
    "3164501","3164502","3189151","792180",
]

# ── SIFT parameters (mirror stage2n_sift.py) ─────────────────────────────────
FLANN_TREES      = 5
FLANN_CHECKS     = 50
RATIO_THRESHOLD  = 0.75
RANSAC_THRESHOLD = 5.0
DIFF_THRESHOLD   = 20
MIN_KP           = 10
GRID_SIZE        = 4

# ── Layout ────────────────────────────────────────────────────────────────────
THUMB_W    = 360
THUMB_H    = 270
LABEL_H    = 22
PARAMS_W   = 460
HEADER_H   = 36
ROW_PAD    = 8
COL_PAD    = 6
FOLD_W     = 120   # left sidebar

IMG_KEYS   = ["D1.jpg", "D4.jpg", "U1.jpg", "U4.jpg"]
COL_LABELS = ["D1  (before)", "D4  (after)", "U1  (before)", "U4  (after)"]

# ── Colours ───────────────────────────────────────────────────────────────────
BG        = (30, 30, 30)
HEADER_BG = (50, 50, 80)
ROW_BG    = (45, 45, 45)
ROW_BG2   = (38, 38, 38)
MISS_BG   = (80, 40, 40)
OK_BG     = (40, 60, 40)
WARN_BG   = (80, 70, 20)
PARAM_BG  = (22, 22, 40)
DIVIDER   = (70, 70, 90)
TEXT_W    = (230, 230, 230)
TEXT_Y    = (230, 220, 80)
TEXT_R    = (230, 100, 100)
TEXT_G    = (100, 210, 100)
TEXT_B    = (130, 180, 240)
TEXT_DIM  = (140, 140, 170)

# ── Font helpers ──────────────────────────────────────────────────────────────
def try_font(size):
    for name in ["arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf",
                 "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()

F_FOLDER  = try_font(14)
F_LABEL   = try_font(12)
F_PARAM   = try_font(12)
F_HEADER  = try_font(14)
F_MISSING = try_font(18)

# ── SIFT metric extraction ────────────────────────────────────────────────────
def extract_sift_metrics(img_before_path, img_after_path):
    """Return dict with kp_before, kp_after, flann_matches, ransac_inliers,
    coverage_pct, or raise RuntimeError with a reason string."""
    before_bgr = cv2.imread(img_before_path)
    after_bgr  = cv2.imread(img_after_path)
    if before_bgr is None:
        raise RuntimeError(f"Cannot read: {img_before_path}")
    if after_bgr is None:
        raise RuntimeError(f"Cannot read: {img_after_path}")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(after_bgr,  cv2.COLOR_BGR2GRAY), None)

    if len(kp1) < MIN_KP or len(kp2) < MIN_KP:
        raise RuntimeError(
            f"Too few KP: {len(kp1)} before / {len(kp2)} after (min {MIN_KP})"
        )
    if des1 is None or des2 is None:
        raise RuntimeError("SIFT descriptor extraction returned None")

    index_params  = dict(algorithm=1, trees=FLANN_TREES)
    search_params = dict(checks=FLANN_CHECKS)
    flann         = cv2.FlannBasedMatcher(index_params, search_params)
    raw           = flann.knnMatch(des1, des2, k=2)
    good          = [m for m, n in raw if m.distance < RATIO_THRESHOLD * n.distance]

    if len(good) < 4:
        raise RuntimeError(f"Too few FLANN matches: {len(good)} (need ≥4)")

    src_pts = np.float32([kp2[m.trainIdx].pt  for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt  for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    if H is None:
        raise RuntimeError("findHomography returned None — insufficient inliers")

    inlier_matches = [m for m, flag in zip(good, mask.ravel()) if flag]

    # Spatial coverage
    h, w = before_bgr.shape[:2]
    occupied = set()
    for m in inlier_matches:
        x, y  = kp1[m.queryIdx].pt
        col   = min(int(x / w * GRID_SIZE), GRID_SIZE - 1)
        row   = min(int(y / h * GRID_SIZE), GRID_SIZE - 1)
        occupied.add((row, col))
    total        = GRID_SIZE * GRID_SIZE
    coverage_pct = len(occupied) / total * 100

    return {
        "kp_before":    len(kp1),
        "kp_after":     len(kp2),
        "flann_matches": len(good),
        "ransac_inliers": len(inlier_matches),
        "coverage_pct": round(coverage_pct, 1),
    }


def compute_all_metrics(folders, orig_dir, run_data, cache):
    """For each folder compute D-pair and U-pair metrics. Use cache where available."""
    updated = False
    for folder_id in folders:
        if folder_id in cache:
            continue  # already cached
        folder_path = os.path.join(orig_dir, folder_id)
        result = {}

        for prefix, before_key, after_key in [
            ("D", "D1.jpg", "D4.jpg"),
            ("U", "U1.jpg", "U4.jpg"),
        ]:
            bp = os.path.join(folder_path, before_key)
            ap = os.path.join(folder_path, after_key)

            # Check if detector already rejected this pair
            pair_tag  = f"{before_key[:-4]}/{after_key[:-4]}"  # D1/D4 or U1/U4
            det_entry = next(
                (e for e in run_data.get(folder_id, []) if e["pair"] == pair_tag), None
            )
            if det_entry and det_entry["reason"] == "detector_rejected":
                result[prefix] = {
                    "error": f"Detector rejected (conf={det_entry['conf']:.3f})"
                }
                continue

            if not os.path.isfile(bp) or not os.path.isfile(ap):
                missing = []
                if not os.path.isfile(bp): missing.append(before_key)
                if not os.path.isfile(ap): missing.append(after_key)
                result[prefix] = {"error": f"Missing: {', '.join(missing)}"}
                continue

            try:
                stats = extract_sift_metrics(bp, ap)
                result[prefix] = stats
                print(f"  {folder_id}/{prefix}: KP {stats['kp_before']}/{stats['kp_after']} "
                      f"FLANN {stats['flann_matches']} inliers {stats['ransac_inliers']} "
                      f"cover {stats['coverage_pct']}%")
            except RuntimeError as e:
                result[prefix] = {"error": str(e)}
                print(f"  {folder_id}/{prefix}: FAILED — {e}")
            except Exception as e:
                result[prefix] = {"error": f"Unexpected: {e}"}
                print(f"  {folder_id}/{prefix}: ERROR — {e}")

        cache[folder_id] = result
        updated = True

    if updated:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"Cache saved: {CACHE_FILE}")

    return cache


# ── Load detector run report data ─────────────────────────────────────────────
def load_run_data():
    """Returns dict: folder_id -> list of {pair, conf, reason}"""
    data = {}
    for rfile in sorted(glob.glob(os.path.join(REPORT_DIR, "run_report_*.json"))):
        with open(rfile) as f:
            d = json.load(f)
        for entry in d.get("rejected", []):
            fid = entry["folder"]
            data.setdefault(fid, []).append({
                "pair": entry["pair"], "conf": entry.get("confidence", 0.0),
                "reason": "detector_rejected"
            })
        for entry in d.get("review_queue", []):
            fid = entry["folder"]
            data.setdefault(fid, []).append({
                "pair": entry["pair"], "conf": entry.get("confidence", 0.0),
                "reason": "detector_review"
            })
    return data


# ── Image helpers ─────────────────────────────────────────────────────────────
def load_thumb(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), (20, 20, 20))
    canvas.paste(img, ((THUMB_W - img.width) // 2, (THUMB_H - img.height) // 2))
    return canvas

def missing_thumb(label):
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), MISS_BG)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, THUMB_W-1, THUMB_H-1], outline=(160, 60, 60), width=2)
    d.text((THUMB_W//2, THUMB_H//2 - 12), "MISSING", font=F_MISSING,
           fill=(255, 100, 100), anchor="mm")
    d.text((THUMB_W//2, THUMB_H//2 + 14), label, font=F_LABEL,
           fill=(200, 130, 130), anchor="mm")
    return canvas

def draw_label_strip(draw, x, y, w, text, bg, fg):
    draw.rectangle([x, y, x+w-1, y+LABEL_H-1], fill=bg)
    draw.text((x + w//2, y + LABEL_H//2), text, font=F_LABEL, fill=fg, anchor="mm")


# ── Parameters panel builder ──────────────────────────────────────────────────
def build_param_lines(folder_id, missing_imgs, run_data, sift_cache):
    """Returns list of (indent, text, colour) lines for the params panel."""
    lines = []
    folder_det = run_data.get(folder_id, [])
    folder_sift = sift_cache.get(folder_id, {})

    # Header: folder ID
    lines.append((0, f"FOLDER  {folder_id}", TEXT_B))

    # Missing image summary
    if missing_imgs:
        lines.append((0, f"MISSING  {', '.join(missing_imgs)}", TEXT_R))
    lines.append((0, "", TEXT_DIM))  # spacer

    for prefix, pair_label in [("D", "D1/D4"), ("U", "U1/U4")]:
        lines.append((0, f"-- {pair_label} pair --", TEXT_DIM))

        det_entry = next((e for e in folder_det if e["pair"] == pair_label), None)
        sift_data = folder_sift.get(prefix, {})

        # Detector confidence (if available)
        if det_entry:
            conf   = det_entry.get("conf", 0.0)
            reason = det_entry["reason"]
            if reason == "detector_rejected":
                lines.append((2, f"Detector conf : {conf:.3f}  REJECTED", TEXT_R))
            else:
                lines.append((2, f"Detector conf : {conf:.3f}  REVIEW", TEXT_Y))

        # SIFT metrics
        if "error" in sift_data:
            lines.append((2, f"SIFT error    : {sift_data['error']}", TEXT_R))
        elif sift_data:
            kpb = sift_data["kp_before"]
            kpa = sift_data["kp_after"]
            fl  = sift_data["flann_matches"]
            inl = sift_data["ransac_inliers"]
            cov = sift_data["coverage_pct"]

            # Supporting metrics (informational)
            kpb_col = TEXT_G if kpb >= 500 else (TEXT_Y if kpb >= 100 else TEXT_R)
            kpa_col = TEXT_G if kpa >= 500 else (TEXT_Y if kpa >= 100 else TEXT_R)
            fl_col  = TEXT_G if fl  >= 50  else (TEXT_Y if fl  >= 10  else TEXT_R)

            lines.append((2, f"KP before     : {kpb:,}", kpb_col))
            lines.append((2, f"KP after      : {kpa:,}", kpa_col))
            lines.append((2, f"FLANN matches : {fl:,}", fl_col))

            # ── Pass/fail determinants ────────────────────────────────────────
            inl_pass = inl >= MIN_INLIERS
            cov_pass = cov >= MIN_COVERAGE
            pair_pass = inl_pass and cov_pass

            inl_col = TEXT_G if inl_pass else TEXT_R
            cov_col = TEXT_G if cov_pass else TEXT_R
            verdict_col = TEXT_G if pair_pass else TEXT_R
            verdict_txt = "PAIR PASS" if pair_pass else "PAIR FAIL"

            lines.append((2, f"RANSAC inliers: {inl:,}  [need >={MIN_INLIERS}]  {'PASS' if inl_pass else 'FAIL'}", inl_col))
            lines.append((2, f"Cover %       : {cov:.1f}%  [need >={MIN_COVERAGE:.0f}%]  {'PASS' if cov_pass else 'FAIL'}", cov_col))
            lines.append((2, f">>> {verdict_txt} <<<", verdict_col))
        else:
            lines.append((2, "SIFT: not run / no data", TEXT_DIM))

        lines.append((0, "", TEXT_DIM))  # spacer between pairs

    return lines


# ── Main: build contact sheet ─────────────────────────────────────────────────
def main():
    print("Loading run report data...")
    run_data = load_run_data()

    # Load or init cache
    if os.path.isfile(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"Cache loaded: {len(cache)} folders already computed")
    else:
        cache = {}
        print("No cache found — computing SIFT metrics for all folders")

    print("Computing SIFT metrics (skipping cached entries)...")
    cache = compute_all_metrics(FAILED_FOLDERS, ORIG_DIR, run_data, cache)

    # ── Layout ────────────────────────────────────────────────────────────────
    cell_h  = THUMB_H + LABEL_H
    row_h   = cell_h + ROW_PAD
    n_cols  = len(IMG_KEYS)
    cell_w  = THUMB_W + COL_PAD
    total_w = FOLD_W + n_cols * cell_w + PARAMS_W + COL_PAD * 2
    total_h = HEADER_H + len(FAILED_FOLDERS) * row_h + ROW_PAD

    print(f"Canvas: {total_w} x {total_h}")
    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([0, 0, total_w, HEADER_H], fill=HEADER_BG)
    draw.text((FOLD_W // 2, HEADER_H // 2), "FOLDER", font=F_HEADER,
              fill=TEXT_W, anchor="mm")
    for ci, col_label in enumerate(COL_LABELS):
        cx = FOLD_W + ci * cell_w + THUMB_W // 2
        draw.text((cx, HEADER_H // 2), col_label, font=F_HEADER, fill=TEXT_W, anchor="mm")
    px_hdr = FOLD_W + n_cols * cell_w + COL_PAD
    draw.text((px_hdr + PARAMS_W // 2, HEADER_H // 2), "SIFT PARAMETERS",
              font=F_HEADER, fill=TEXT_W, anchor="mm")

    # Rows
    for ri, folder_id in enumerate(FAILED_FOLDERS):
        row_top = HEADER_H + ri * row_h
        draw.rectangle([0, row_top, total_w, row_top + row_h - 1],
                       fill=ROW_BG if ri % 2 == 0 else ROW_BG2)

        # Folder ID sidebar
        draw.text((FOLD_W // 2, row_top + cell_h // 2), folder_id,
                  font=F_FOLDER, fill=TEXT_B, anchor="mm")

        # Determine which images exist
        folder_path  = os.path.join(ORIG_DIR, folder_id)
        present      = set(os.listdir(folder_path)) if os.path.isdir(folder_path) else set()
        missing_imgs = [k for k in IMG_KEYS if k not in present]
        folder_det   = run_data.get(folder_id, [])

        # Image cells
        for ci, img_key in enumerate(IMG_KEYS):
            cell_x   = FOLD_W + ci * cell_w
            img_path = os.path.join(folder_path, img_key)
            pair_tag = "D1/D4" if img_key.startswith("D") else "U1/U4"
            det_fail = any(e["pair"] == pair_tag and e["reason"] == "detector_rejected"
                           for e in folder_det)

            if img_key in present and os.path.isfile(img_path):
                try:
                    thumb = load_thumb(img_path)
                    canvas.paste(thumb, (cell_x, row_top))
                    strip_bg = WARN_BG if det_fail else OK_BG
                    strip_fg = TEXT_Y  if det_fail else TEXT_G
                    strip_lbl = f"{img_key}  [DET FAIL]" if det_fail else img_key
                    draw_label_strip(draw, cell_x, row_top + THUMB_H, THUMB_W,
                                     strip_lbl, strip_bg, strip_fg)
                except Exception as e:
                    canvas.paste(missing_thumb(f"ERR: {e}"), (cell_x, row_top))
                    draw_label_strip(draw, cell_x, row_top + THUMB_H, THUMB_W,
                                     img_key + " [READ ERR]", MISS_BG, TEXT_R)
            else:
                canvas.paste(missing_thumb(img_key), (cell_x, row_top))
                draw_label_strip(draw, cell_x, row_top + THUMB_H, THUMB_W,
                                 img_key + " [NOT FOUND]", MISS_BG, TEXT_R)

        # Parameters panel
        px   = FOLD_W + n_cols * cell_w + COL_PAD
        draw.rectangle([px, row_top, px + PARAMS_W - 2, row_top + cell_h - 1],
                       fill=PARAM_BG, outline=DIVIDER)

        param_lines = build_param_lines(folder_id, missing_imgs, run_data, cache)
        line_h = 16
        py     = row_top + 6
        for indent, text, col in param_lines:
            if text:
                lx = px + 8 + indent * 6
                draw.text((lx, py), text, font=F_PARAM, fill=col)
            py += line_h

        # Row divider
        draw.line([(0, row_top + row_h - 1), (total_w, row_top + row_h - 1)],
                  fill=DIVIDER, width=1)

    canvas.save(OUT_FILE, quality=90, optimize=True)
    print(f"\nSaved: {OUT_FILE}")
    print(f"Size : {canvas.width} x {canvas.height} px")


if __name__ == "__main__":
    main()
