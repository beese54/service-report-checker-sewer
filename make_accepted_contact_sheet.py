"""
Generate a contact sheet for all ACCEPTED folders showing all 12 images
(D1-D6 and U1-U6) plus pipeline parameters from Stage 1, 2 and 3.

Layout per folder:
  D-row: [D1][D2][D3][D4][D5][D6] + D params panel
  U-row: [U1][U2][U3][U4][U5][U6] + U params panel

Output: accepted_contact_sheet.jpg
"""

import os
import json
import glob as glob_mod

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE       = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker"
ORIG_DIR   = os.path.join(BASE, "original_images")
DIFF_DIR   = os.path.join(BASE, "adjusted_images", "no_obstruction", "difference_analysis")
CACHE_FILE = os.path.join(BASE, "sift_metrics_cache.json")
WASH_FILE  = os.path.join(DIFF_DIR, "D2D5U2U5_confidence_summary.json")
GF_FILE    = os.path.join(DIFF_DIR, "gf_run_20260304_214712.json")
OUT_FILE   = os.path.join(BASE, "accepted_contact_sheet.jpg")

# ---------------------------------------------------------------------------
# Accepted folder list
# ---------------------------------------------------------------------------
ACCEPTED_FOLDERS = [
    "1159097","1159109","1159115","1159119","1159121","1159122",
    "1296709","1296710","1296711","140504","140888","146915",
    "152609","152610","152836","156125","160389","160518","160519",
    "162944","163100","169287","169292","169297","172128","172130",
    "172143","172165","172173","172175","172176","172178","172290",
    "172291","172292","172299","178259","178260","185221","185224",
    "185225","185226","185227","185228","185231","185232","185233",
    "185251","185252","185253","187811","187999","188386","188493",
    "188494","188495","188496","190456","190457","190461","191226",
    "195161","195172","203548","203561","203567","203568","203570",
    "203571","203587","203590","203591","203592","203998","203999",
    "204027","204028","204029","204031","204035","204036","204037",
    "204038","204182","204183","204184","204408","204409","204533",
    "204930","206035","206037","206041","206639","206640","206641",
    "206644","206645","206647","206648","206812","206813","206853",
    "206854","208891","208892","208893","208894","208896","209056",
    "209106","209456","3103199","3231057","3273387","3273388",
    "3347821","3400650","3400652","3400655","3474111","380490",
]

# ---------------------------------------------------------------------------
# SIFT parameters (mirrors stage2n_sift.py)
# ---------------------------------------------------------------------------
FLANN_TREES      = 5
FLANN_CHECKS     = 50
RATIO_THRESHOLD  = 0.75
RANSAC_THRESHOLD = 5.0
GRID_SIZE        = 4
MIN_KP           = 10

# Stage pass thresholds
SIFT_MIN_INLIERS  = 10
SIFT_MIN_COVERAGE = 25.0
WASH_MIN_CONF     = 0.50
WASH_PASS_TIER    = "HIGH"
GF_MIN_SCORE      = 3

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
THUMB_W    = 220
THUMB_H    = 165
LABEL_H    = 20
SUB_ROW_H  = THUMB_H + LABEL_H   # 185
FOLDER_HDR = 24
GAP        = 6
SIDEBAR_W  = 100
COL_PAD    = 4
PARAMS_W   = 440
HEADER_H   = 30

N_IMGS     = 6
TOTAL_IMG_W = N_IMGS * (THUMB_W + COL_PAD)   # 6 * 224 = 1344
TOTAL_W    = SIDEBAR_W + TOTAL_IMG_W + PARAMS_W  # 100 + 1344 + 440 = 1884

# Per-folder height: header + D-row + U-row + gap
FOLDER_H   = FOLDER_HDR + SUB_ROW_H + SUB_ROW_H + GAP

D_KEYS = ["D1", "D2", "D3", "D4", "D5", "D6"]
U_KEYS = ["U1", "U2", "U3", "U4", "U5", "U6"]

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG          = (30,  30,  30)
HEADER_BG   = (50,  50,  80)
FOLDER_HDR_BG = (35, 35,  55)
ROW_BG_D    = (42,  42,  52)
ROW_BG_U    = (38,  45,  42)
MISS_BG     = (80,  40,  40)
OK_BG       = (40,  60,  40)
PARAM_BG    = (22,  22,  40)
DIVIDER     = (70,  70,  90)
TEXT_W      = (230, 230, 230)
TEXT_Y      = (230, 220,  80)
TEXT_R      = (230, 100, 100)
TEXT_G      = (100, 210, 100)
TEXT_B      = (130, 180, 240)
TEXT_DIM    = (140, 140, 170)

# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------
def try_font(size):
    for name in ["arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf",
                 "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_FOLDER  = try_font(12)
F_LABEL   = try_font(11)
F_PARAM   = try_font(10)
F_HEADER  = try_font(12)
F_MISSING = try_font(14)
F_FHDR    = try_font(12)

# ---------------------------------------------------------------------------
# SIFT metric extraction
# ---------------------------------------------------------------------------
def extract_sift_metrics(img_before_path, img_after_path):
    """Return dict with kp_before, kp_after, flann_matches, ransac_inliers,
    coverage_pct, or raise RuntimeError with a reason string."""
    before_bgr = cv2.imread(img_before_path)
    after_bgr  = cv2.imread(img_after_path)
    if before_bgr is None:
        raise RuntimeError(f"Cannot read: {img_before_path}")
    if after_bgr is None:
        raise RuntimeError(f"Cannot read: {img_after_path}")

    sift      = cv2.SIFT_create()
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
        raise RuntimeError(f"Too few FLANN matches: {len(good)} (need >=4)")

    src_pts = np.float32([kp2[m.trainIdx].pt  for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt  for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    if H is None:
        raise RuntimeError("findHomography returned None -- insufficient inliers")

    inlier_matches = [m for m, flag in zip(good, mask.ravel()) if flag]

    h, w     = before_bgr.shape[:2]
    occupied = set()
    for m in inlier_matches:
        x, y = kp1[m.queryIdx].pt
        col  = min(int(x / w * GRID_SIZE), GRID_SIZE - 1)
        row  = min(int(y / h * GRID_SIZE), GRID_SIZE - 1)
        occupied.add((row, col))
    coverage_pct = len(occupied) / (GRID_SIZE * GRID_SIZE) * 100

    return {
        "kp_before":      len(kp1),
        "kp_after":       len(kp2),
        "flann_matches":  len(good),
        "ransac_inliers": len(inlier_matches),
        "coverage_pct":   round(coverage_pct, 1),
    }


def ensure_sift_cache(folders, cache):
    """Compute SIFT metrics for any folders not yet in cache. Saves cache."""
    updated = False
    for folder_id in folders:
        if folder_id in cache:
            continue
        folder_path = os.path.join(ORIG_DIR, folder_id)
        result = {}
        for prefix, bkey, akey in [("D", "D1.jpg", "D4.jpg"), ("U", "U1.jpg", "U4.jpg")]:
            bp = os.path.join(folder_path, bkey)
            ap = os.path.join(folder_path, akey)
            if not os.path.isfile(bp) or not os.path.isfile(ap):
                missing = []
                if not os.path.isfile(bp): missing.append(bkey)
                if not os.path.isfile(ap): missing.append(akey)
                result[prefix] = {"error": f"Missing: {', '.join(missing)}"}
                continue
            try:
                stats = extract_sift_metrics(bp, ap)
                result[prefix] = stats
            except RuntimeError as e:
                result[prefix] = {"error": str(e)}
            except Exception as e:
                result[prefix] = {"error": f"Unexpected: {e}"}
        cache[folder_id] = result
        updated = True

    if updated:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  Cache updated: {CACHE_FILE}")

    return cache


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_sift_cache():
    if os.path.isfile(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def load_washing():
    """Returns dict: folder_id -> washing entry (tier, confidence, pairs list)."""
    if not os.path.isfile(WASH_FILE):
        print(f"WARNING: washing file not found: {WASH_FILE}")
        return {}
    with open(WASH_FILE) as f:
        return json.load(f)


def load_geometry():
    """Returns dict: (folder_id, prefix) -> result entry."""
    if not os.path.isfile(GF_FILE):
        print(f"WARNING: geometry file not found: {GF_FILE}")
        return {}
    with open(GF_FILE) as f:
        raw = json.load(f)
    results = raw if isinstance(raw, list) else raw.get("results", [])
    index = {}
    for entry in results:
        key = (str(entry.get("folder", "")), str(entry.get("prefix", "")))
        index[key] = entry
    return index


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def load_thumb(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), (20, 20, 20))
    canvas.paste(img, ((THUMB_W - img.width) // 2, (THUMB_H - img.height) // 2))
    return canvas


def missing_thumb():
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), MISS_BG)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, THUMB_W - 1, THUMB_H - 1], outline=(160, 60, 60), width=2)
    d.text((THUMB_W // 2, THUMB_H // 2 - 8), "MISSING", font=F_MISSING,
           fill=(255, 100, 100), anchor="mm")
    return canvas


def draw_label_strip(draw, x, y, w, text, bg, fg):
    draw.rectangle([x, y, x + w - 1, y + LABEL_H - 1], fill=bg)
    draw.text((x + w // 2, y + LABEL_H // 2), text, font=F_LABEL,
              fill=fg, anchor="mm")


# ---------------------------------------------------------------------------
# Signal score helpers (washing stage)
# ---------------------------------------------------------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_signal_scores(pair_entry):
    """Return dict of 0-1 scores for each washing signal."""
    kp_ratio         = pair_entry.get("kp_ratio",         0.0)
    std_inc          = pair_entry.get("std_increase_pct",  0.0)
    ent_inc          = pair_entry.get("entropy_increase",  0.0)
    match_ratio      = pair_entry.get("match_ratio",       0.5)
    edge_inc         = pair_entry.get("edge_increase_pct", 0.0)
    lap_inc          = pair_entry.get("lap_increase_pct",  0.0)
    return {
        "kp":    clamp((kp_ratio - 1.0) / 0.5,              0, 1),
        "std":   clamp(std_inc / 20.0,                       0, 1),
        "ent":   clamp(ent_inc / 0.5,                        0, 1),
        "mat":   clamp((0.5 - match_ratio) / (0.5 - 0.05),  0, 1),
        "edge":  clamp(edge_inc / 20.0,                      0, 1),
        "lap":   clamp(lap_inc / 30.0,                       0, 1),
    }


# ---------------------------------------------------------------------------
# Params panel text builder
# ---------------------------------------------------------------------------
def build_param_lines(folder_id, prefix, sift_cache, washing, geometry):
    """
    Build list of (text, colour) tuples for one sub-row (D or U).
    prefix is "D" or "U".
    """
    lines = []
    NA    = "N/A"

    # Pair labels
    s1_pair  = f"{prefix}1/{prefix}4"   # e.g. D1/D4
    s2_pair  = f"{prefix}2/{prefix}5"   # e.g. D2/D5
    s3_pair  = f"{prefix}3/{prefix}6"   # e.g. D3/D6
    s3_pfx   = f"{prefix}3"             # geometry result prefix key e.g. D3

    # ---- Stage 1: SIFT -------------------------------------------------------
    sift_data = sift_cache.get(folder_id, {}).get(prefix, {})
    if "error" in sift_data:
        s1_status = "ERROR"
        s1_col    = TEXT_R
        s1_detail = f"inliers=- cover=-%  [{sift_data['error'][:28]}]"
    elif sift_data:
        inl = sift_data["ransac_inliers"]
        cov = sift_data["coverage_pct"]
        ok  = inl >= SIFT_MIN_INLIERS and cov >= SIFT_MIN_COVERAGE
        s1_status = "PASS" if ok else "FAIL"
        s1_col    = TEXT_G if ok else TEXT_R
        s1_detail = f"inliers={inl} cover={cov:.1f}%"
    else:
        s1_status = "N/A"
        s1_col    = TEXT_DIM
        s1_detail = "no data"

    thresh_s1 = f"thresholds: >={SIFT_MIN_INLIERS} inliers, >={SIFT_MIN_COVERAGE}% cover"
    lines.append((f"Stage1 {s1_pair}: {s1_detail} [{s1_status}]   {thresh_s1}",
                  s1_col if s1_status != "N/A" else TEXT_DIM))

    # ---- Stage 2: Washing confidence -----------------------------------------
    wash_folder = washing.get(folder_id, {})
    # Find the pair for this prefix
    wash_pair_entry = None
    for pe in wash_folder.get("pairs", []):
        p = pe.get("pair", "")
        if p == s2_pair:
            wash_pair_entry = pe
            break

    if wash_pair_entry is not None:
        wconf = wash_pair_entry.get("washing_confidence",
                wash_folder.get("confidence", 0.0))
        wtier = wash_pair_entry.get("washing_tier",
                wash_folder.get("tier", ""))
        ok2   = wconf >= WASH_MIN_CONF and wtier == WASH_PASS_TIER
        s2_status = "PASS" if ok2 else "FAIL"
        s2_col    = TEXT_G if ok2 else TEXT_R

        scores = compute_signal_scores(wash_pair_entry)
        sig_str = (f"kp={scores['kp']:.2f} std={scores['std']:.2f} "
                   f"ent={scores['ent']:.2f} mat={scores['mat']:.2f} "
                   f"edge={scores['edge']:.2f} lap={scores['lap']:.2f}")
        s2_detail = f"wash_conf={wconf:.2f} {wtier} [{s2_status}]   signals: {sig_str}"
    else:
        s2_col    = TEXT_DIM
        s2_detail = f"wash_conf=N/A N/A [N/A]   signals: N/A"

    lines.append((f"Stage2 {s2_pair}: {s2_detail}", s2_col))

    # ---- Stage 3: Geometry ---------------------------------------------------
    geo_entry = geometry.get((folder_id, s3_pfx), None)
    if geo_entry is not None:
        score  = geo_entry.get("score", 0)
        status = geo_entry.get("status", "?")
        ok3    = score >= GF_MIN_SCORE
        s3_col = TEXT_G if ok3 else TEXT_R

        gate  = geo_entry.get("gate", {})
        sist  = geo_entry.get("sift_stats", {})
        grease= geo_entry.get("grease", {})
        tex   = geo_entry.get("texture", {})
        water = geo_entry.get("water", {})

        # Sub-signal pass/fail tags
        # S1: blur gate (blur_score present and circle_found)
        blur_ok    = gate.get("acceptable", False)
        s1_tag     = "P" if blur_ok else "F"
        # S2: pct_changed >= 25
        pct_chg    = sist.get("pct_changed", 0.0)
        s2_tag     = "P" if pct_chg >= 25.0 else "F"
        # S3: texture confirmed
        tex_conf   = tex.get("confirmed", False)
        s3_tag     = "P" if tex_conf else "F"
        # S4: water detected
        water_det  = water.get("water_detected", False)
        s4_tag     = "P" if water_det else "F"
        # S5: not greasy
        grease_flg = grease.get("flagged", True)
        s5_tag     = "P" if not grease_flg else "F"

        inl3   = sist.get("ransac_inliers", 0)
        detail = (f"score={score}/5 [{status}]   "
                  f"S1:inliers={inl3}({'P' if blur_ok else 'F'}) "
                  f"S2:chg={pct_chg:.1f}%({s2_tag}) "
                  f"S3:tex({s3_tag}) "
                  f"S4:water({s4_tag}) "
                  f"S5:{'ngrease' if not grease_flg else 'grease'}({s5_tag})")
    else:
        s3_col = TEXT_DIM
        detail = f"score=N/A [N/A]   N/A"

    lines.append((f"Stage3 {s3_pair}: {detail}", s3_col))

    return lines


# ---------------------------------------------------------------------------
# Draw one sub-row (D or U) images + params panel
# ---------------------------------------------------------------------------
def draw_sub_row(canvas, draw, folder_id, prefix, keys, row_bg, row_top,
                 sift_cache, washing, geometry):
    """
    Draw the image cells and params panel for one prefix row (D or U).
    row_top: y pixel where this sub-row starts (after folder header offset).
    """
    # Background
    draw.rectangle([0, row_top, TOTAL_W - 1, row_top + SUB_ROW_H - 1], fill=row_bg)

    folder_path = os.path.join(ORIG_DIR, folder_id)
    present     = set(os.listdir(folder_path)) if os.path.isdir(folder_path) else set()

    for ci, key in enumerate(keys):
        cell_x   = SIDEBAR_W + ci * (THUMB_W + COL_PAD)
        img_path = os.path.join(folder_path, key + ".jpg")
        img_path_png = os.path.join(folder_path, key + ".png")

        found_path = None
        if key + ".jpg" in present and os.path.isfile(img_path):
            found_path = img_path
        elif key + ".png" in present and os.path.isfile(img_path_png):
            found_path = img_path_png
        else:
            # Try case-insensitive
            for fname in present:
                if fname.lower() == (key + ".jpg").lower() or \
                   fname.lower() == (key + ".png").lower():
                    found_path = os.path.join(folder_path, fname)
                    break

        if found_path:
            try:
                thumb = load_thumb(found_path)
                canvas.paste(thumb, (cell_x, row_top))
                draw_label_strip(draw, cell_x, row_top + THUMB_H, THUMB_W,
                                 key, OK_BG, TEXT_G)
            except Exception:
                canvas.paste(missing_thumb(), (cell_x, row_top))
                draw_label_strip(draw, cell_x, row_top + THUMB_H, THUMB_W,
                                 key + " [ERR]", MISS_BG, TEXT_R)
        else:
            canvas.paste(missing_thumb(), (cell_x, row_top))
            draw_label_strip(draw, cell_x, row_top + THUMB_H, THUMB_W,
                             key + " [MISSING]", MISS_BG, TEXT_R)

    # Params panel
    px = SIDEBAR_W + TOTAL_IMG_W
    draw.rectangle([px, row_top, px + PARAMS_W - 1, row_top + SUB_ROW_H - 1],
                   fill=PARAM_BG, outline=DIVIDER)

    param_lines = build_param_lines(folder_id, prefix, sift_cache, washing, geometry)
    line_h = 14
    py     = row_top + 5
    for text, col in param_lines:
        # Wrap long lines to fit in panel (approx 58 chars for 10pt font in 440px)
        max_chars = 66
        if len(text) > max_chars:
            # Split at first "   " separator if possible to get main + detail
            parts = text.split("   ", 1)
            draw.text((px + 6, py), parts[0], font=F_PARAM, fill=col)
            py += line_h
            if len(parts) > 1:
                draw.text((px + 12, py), parts[1], font=F_PARAM, fill=TEXT_DIM)
                py += line_h
        else:
            draw.text((px + 6, py), text, font=F_PARAM, fill=col)
            py += line_h
        py += 2  # small extra gap between stage lines

    # Divider at bottom of sub-row
    draw.line([(SIDEBAR_W, row_top + SUB_ROW_H - 1),
               (TOTAL_W - 1, row_top + SUB_ROW_H - 1)],
              fill=DIVIDER, width=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data files...")
    sift_cache = load_sift_cache()
    print(f"  SIFT cache: {len(sift_cache)} folders")
    washing = load_washing()
    print(f"  Washing:    {len(washing)} folders")
    geometry = load_geometry()
    print(f"  Geometry:   {len(geometry)} entries")

    print("Ensuring SIFT cache for all accepted folders...")
    sift_cache = ensure_sift_cache(ACCEPTED_FOLDERS, sift_cache)

    # ---- Canvas setup --------------------------------------------------------
    n_folders = len(ACCEPTED_FOLDERS)
    total_h   = HEADER_H + n_folders * FOLDER_H

    print(f"Canvas: {TOTAL_W} x {total_h} px ({n_folders} folders)")
    canvas = Image.new("RGB", (TOTAL_W, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # ---- Global header -------------------------------------------------------
    draw.rectangle([0, 0, TOTAL_W - 1, HEADER_H - 1], fill=HEADER_BG)
    draw.text((SIDEBAR_W // 2, HEADER_H // 2), "ID", font=F_HEADER,
              fill=TEXT_W, anchor="mm")

    d_col_labels = ["D1", "D2", "D3", "D4", "D5", "D6"]
    u_col_labels = ["U1", "U2", "U3", "U4", "U5", "U6"]
    # Use D labels as header (same positions for U)
    for ci, lbl in enumerate(d_col_labels):
        cx = SIDEBAR_W + ci * (THUMB_W + COL_PAD) + THUMB_W // 2
        draw.text((cx, HEADER_H // 2), lbl, font=F_HEADER, fill=TEXT_W, anchor="mm")
    px_hdr = SIDEBAR_W + TOTAL_IMG_W
    draw.text((px_hdr + PARAMS_W // 2, HEADER_H // 2), "PARAMETERS",
              font=F_HEADER, fill=TEXT_W, anchor="mm")

    # ---- Folder rows ---------------------------------------------------------
    for fi, folder_id in enumerate(ACCEPTED_FOLDERS):
        print(f"Processing {fi + 1}/{n_folders}: {folder_id}...")

        folder_top = HEADER_H + fi * FOLDER_H

        # Folder header bar
        draw.rectangle([0, folder_top, TOTAL_W - 1, folder_top + FOLDER_HDR - 1],
                       fill=FOLDER_HDR_BG)
        draw.text((TOTAL_W // 2, folder_top + FOLDER_HDR // 2),
                  f"Folder: {folder_id}",
                  font=F_FHDR, fill=TEXT_B, anchor="mm")

        # Sidebar: folder ID vertically centred across D+U rows
        side_top = folder_top + FOLDER_HDR
        side_h   = SUB_ROW_H * 2
        draw.text((SIDEBAR_W // 2, side_top + side_h // 2),
                  folder_id, font=F_FOLDER, fill=TEXT_B, anchor="mm")

        # D row
        d_row_top = folder_top + FOLDER_HDR
        draw_sub_row(canvas, draw, folder_id, "D", D_KEYS, ROW_BG_D,
                     d_row_top, sift_cache, washing, geometry)

        # U row
        u_row_top = folder_top + FOLDER_HDR + SUB_ROW_H
        draw_sub_row(canvas, draw, folder_id, "U", U_KEYS, ROW_BG_U,
                     u_row_top, sift_cache, washing, geometry)

        # Folder separator gap (filled with BG already, draw a thicker divider)
        sep_y = folder_top + FOLDER_HDR + SUB_ROW_H * 2
        draw.rectangle([0, sep_y, TOTAL_W - 1, sep_y + GAP - 1], fill=BG)
        draw.line([(0, sep_y), (TOTAL_W - 1, sep_y)], fill=DIVIDER, width=1)

    # ---- Save ----------------------------------------------------------------
    print(f"Saving to {OUT_FILE} ...")
    canvas.save(OUT_FILE, quality=88, optimize=True)
    print(f"Done. Image size: {canvas.width} x {canvas.height} px")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
