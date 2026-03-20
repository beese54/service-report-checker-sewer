"""
Generate a contact sheet for folders that failed the D3/D6 + U3/U6 geometry stage.

Each row = one folder.
Columns : D3, D6, U3, U6 (360x270 thumbnails).
Left sidebar (120 px) : folder ID.
Right panel  (500 px) : D3/D6 and U3/U6 signal breakdown with colour-coded verdicts.

Data source : gf_run_20260304_214712.json in the difference_analysis directory.
"""

import json
import os

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE        = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker"
ORIG_DIR    = os.path.join(BASE, "original_images")
GF_RUN_JSON = os.path.join(
    BASE,
    "adjusted_images", "no_obstruction", "difference_analysis",
    "gf_run_20260304_214712.json",
)
OUT_FILE    = os.path.join(BASE, "failed_D3D6_U3U6_contact_sheet.jpg")

# ---------------------------------------------------------------------------
# Failed folders
# ---------------------------------------------------------------------------
FAILED_FOLDERS = ["1159111", "152834", "152835", "190455", "206038", "206039"]

# ---------------------------------------------------------------------------
# Thresholds (mirror stage4n_geometry.py)
# ---------------------------------------------------------------------------
BLUR_REJECT_THRESHOLD  = 35.0
RANSAC_INLIERS_PASS    = 10
PCT_CHANGED_PASS       = 5.0
GREASE_FLAG_THRESHOLD  = 2.0    # grease_pct > this -> flagged (used in stage)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
THUMB_W   = 360
THUMB_H   = 270
LABEL_H   = 22
FOLD_W    = 120     # left sidebar
PARAMS_W  = 500     # right parameters panel
HEADER_H  = 36
ROW_PAD   = 8
COL_PAD   = 6

IMG_KEYS   = ["D3.jpg", "D6.jpg", "U3.jpg", "U6.jpg"]
COL_LABELS = ["D3  (before)", "D6  (after)", "U3  (before)", "U6  (after)"]

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG        = (30,  30,  30)
HEADER_BG = (50,  50,  80)
ROW_BG    = (45,  45,  45)
ROW_BG2   = (38,  38,  38)
MISS_BG   = (80,  40,  40)
OK_BG     = (40,  60,  40)
WARN_BG   = (80,  70,  20)
PARAM_BG  = (22,  22,  40)
DIVIDER   = (70,  70,  90)
TEXT_W    = (230, 230, 230)
TEXT_Y    = (230, 220,  80)
TEXT_R    = (230, 100, 100)
TEXT_G    = (100, 210, 100)
TEXT_B    = (130, 180, 240)
TEXT_DIM  = (140, 140, 170)

# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------
def try_font(size):
    for name in ["arialbd.ttf", "arial.ttf",
                 "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
                 "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_FOLDER  = try_font(14)
F_LABEL   = try_font(12)
F_PARAM   = try_font(11)
F_HEADER  = try_font(14)
F_MISSING = try_font(18)

# ---------------------------------------------------------------------------
# Load gf_run JSON and build lookup: folder -> {prefix -> entry}
# ---------------------------------------------------------------------------
def load_gf_lookup(json_path):
    """Return dict: folder_id -> {prefix -> entry_dict}."""
    with open(json_path, "r") as fh:
        data = json.load(fh)
    results = data.get("results", [])
    lookup = {}
    for entry in results:
        fid    = entry.get("folder", "")
        prefix = entry.get("prefix", "")   # "D3" or "U3"
        if fid and prefix:
            lookup.setdefault(fid, {})[prefix] = entry
    return lookup


# ---------------------------------------------------------------------------
# Verdict / signal helpers
# ---------------------------------------------------------------------------
def _signal_pass_fail(entry):
    """
    Derive the five boolean signals from a full-results entry.
    Returns dict: {s1, s2, s3, s4, s5, score, verdict} or None if entry is
    not a full-results entry (e.g. GATE_REJECTED, ALIGNMENT_FAILED, MISSING).
    """
    if "sift_stats" not in entry:
        return None
    ss      = entry["sift_stats"]
    grease  = entry.get("grease", {})
    texture = entry.get("texture", {})
    water   = entry.get("water", {})

    s1 = ss.get("ransac_inliers", 0) >= RANSAC_INLIERS_PASS
    s2 = ss.get("pct_changed",    0) >= PCT_CHANGED_PASS
    s3 = bool(texture.get("confirmed", False))
    s4 = bool(water.get("water_detected", False))
    s5 = not bool(grease.get("flagged", False))

    score = sum([s1, s2, s3, s4, s5])
    if score >= 3:
        verdict = "PASS"
    elif score >= 2:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"

    # Hard overrides (mirror stage4n_geometry.py)
    if grease.get("grease_pct", 0) > 15.0 and not water.get("water_detected", False):
        verdict = "FAIL"
    if ss.get("ransac_inliers", 0) < 4 and verdict == "PASS":
        verdict = "REVIEW"

    return dict(s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, score=score, verdict=verdict)


def verdict_colour(v):
    if v == "PASS":
        return TEXT_G
    if v == "REVIEW":
        return TEXT_Y
    return TEXT_R


def pass_col(ok):
    return TEXT_G if ok else TEXT_R


# ---------------------------------------------------------------------------
# Build the parameter panel lines for one pair (D3/D6 or U3/U6)
# ---------------------------------------------------------------------------
def build_pair_lines(pair_label, entry):
    """
    Returns list of (indent_px, text, colour) for one pair.
    entry may be None (not in JSON) or have various status values.
    """
    lines = []

    # ---- Section header ------------------------------------------------
    lines.append((0, f"-- {pair_label} pair --", TEXT_DIM))

    if entry is None:
        lines.append((8, "No data in gf_run JSON", TEXT_R))
        lines.append((0, "", TEXT_DIM))
        return lines

    status = entry.get("status", "UNKNOWN")

    # ---- MISSING_IMAGES ------------------------------------------------
    if status == "MISSING_IMAGES":
        err = entry.get("error", "Images missing")
        lines.append((8, f"Status: MISSING_IMAGES", TEXT_R))
        lines.append((8, f"{err}", TEXT_R))
        lines.append((0, "", TEXT_DIM))
        return lines

    # ---- Gate info (present for GATE_REJECTED, ALIGNMENT_FAILED, full) --
    gate = entry.get("gate", {})
    if gate:
        blur   = gate.get("blur_score", None)
        circ   = gate.get("circle_found", None)
        cent   = gate.get("centering_ok", None)
        blur_ok = (blur is not None and blur >= BLUR_REJECT_THRESHOLD)

        blur_str  = f"{blur:.1f}" if blur is not None else "n/a"
        circ_str  = "Yes" if circ else ("No" if circ is not None else "n/a")
        cent_str  = "Yes" if cent else ("No" if cent is not None else "n/a")

        lines.append((8,
            f"Gate  blur_score   : {blur_str}   [pass >= {BLUR_REJECT_THRESHOLD:.0f}]",
            pass_col(blur_ok)))
        lines.append((8,
            f"Gate  circle found : {circ_str}  centred: {cent_str}",
            TEXT_G if (circ and cent) else TEXT_Y if circ else TEXT_R))

    # ---- GATE_REJECTED -------------------------------------------------
    if status == "GATE_REJECTED":
        reasons = entry.get("reject_reasons", entry.get("gate", {}).get("reject_reasons", []))
        lines.append((8, f"Status: GATE_REJECTED", TEXT_R))
        for r in reasons:
            lines.append((14, r, TEXT_R))
        lines.append((0, "", TEXT_DIM))
        return lines

    # ---- ALIGNMENT_FAILED ----------------------------------------------
    if status == "ALIGNMENT_FAILED":
        err = entry.get("error", "Alignment failed")
        lines.append((8, f"Status: ALIGNMENT_FAILED", TEXT_R))
        lines.append((14, err, TEXT_R))
        lines.append((0, "", TEXT_DIM))
        return lines

    # ---- Full results entry (status == PASS / REVIEW / FAIL) -----------
    signals = _signal_pass_fail(entry)
    score   = entry.get("score", signals["score"] if signals else "?")
    verdict = entry.get("status", signals["verdict"] if signals else "?")

    # Use our re-derived verdict for colouring (JSON status may differ)
    v_colour = verdict_colour(verdict)
    lines.append((8,
        f"Verdict: {verdict}  Score: {score}/5",
        v_colour))

    if signals is None:
        lines.append((8, "Signal data unavailable", TEXT_R))
        lines.append((0, "", TEXT_DIM))
        return lines

    ss      = entry.get("sift_stats", {})
    grease  = entry.get("grease", {})
    texture = entry.get("texture", {})
    water   = entry.get("water", {})

    # S1: RANSAC inliers
    inliers = ss.get("ransac_inliers", 0)
    s1_ok   = signals["s1"]
    s1_lbl  = "PASS" if s1_ok else "FAIL"
    lines.append((8,
        f"S1  RANSAC inliers : {inliers}     [{s1_lbl} {'>' if s1_ok else '<'} {RANSAC_INLIERS_PASS}]",
        pass_col(s1_ok)))

    # S2: pct_changed
    pct  = ss.get("pct_changed", 0)
    s2_ok = signals["s2"]
    s2_lbl = "PASS" if s2_ok else "FAIL"
    lines.append((8,
        f"S2  Pct changed    : {pct:.1f}% [{s2_lbl} >= {PCT_CHANGED_PASS:.0f}%]",
        pass_col(s2_ok)))

    # S3: texture entropy
    delta = texture.get("entropy_delta", 0.0)
    s3_ok = signals["s3"]
    s3_lbl = "PASS" if s3_ok else "FAIL"
    lines.append((8,
        f"S3  Texture/entropy: {delta:.3f} delta [{s3_lbl}]",
        pass_col(s3_ok)))

    # S4: water detected
    w_det  = water.get("water_detected", False)
    w_conf = water.get("water_confidence", 0.0)
    s4_ok  = signals["s4"]
    s4_lbl = "PASS" if s4_ok else "FAIL"
    w_det_str = "Yes" if w_det else "No"
    lines.append((8,
        f"S4  Water detected : {w_det_str} {w_conf*100:.1f}% conf [{s4_lbl}]",
        pass_col(s4_ok)))

    # S5: grease not flagged
    g_flagged = grease.get("flagged", False)
    g_pct     = grease.get("grease_pct", 0.0)
    s5_ok     = signals["s5"]
    s5_lbl    = "PASS" if s5_ok else "FAIL"
    g_flag_str = "Yes" if g_flagged else "No"
    lines.append((8,
        f"S5  Grease flagged : {g_flag_str}  {g_pct:.2f}% [{s5_lbl}]",
        pass_col(s5_ok)))

    # Extra stats
    ssim  = ss.get("ssim_score", None)
    inlr  = ss.get("inlier_ratio", None)
    if ssim is not None:
        lines.append((8, f"SSIM score         : {ssim:.2f}", TEXT_DIM))
    if inlr is not None:
        lines.append((8, f"Inlier ratio       : {inlr*100:.1f}%", TEXT_DIM))

    lines.append((0, "", TEXT_DIM))   # spacer
    return lines


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def load_thumb(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), (20, 20, 20))
    canvas.paste(img, ((THUMB_W - img.width) // 2, (THUMB_H - img.height) // 2))
    return canvas


def missing_thumb(label):
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), MISS_BG)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, THUMB_W - 1, THUMB_H - 1], outline=(160, 60, 60), width=2)
    d.text((THUMB_W // 2, THUMB_H // 2 - 12), "MISSING",
           font=F_MISSING, fill=(255, 100, 100), anchor="mm")
    d.text((THUMB_W // 2, THUMB_H // 2 + 14), label,
           font=F_LABEL, fill=(200, 130, 130), anchor="mm")
    return canvas


def draw_label_strip(draw, x, y, w, text, bg, fg):
    draw.rectangle([x, y, x + w - 1, y + LABEL_H - 1], fill=bg)
    draw.text((x + w // 2, y + LABEL_H // 2), text,
              font=F_LABEL, fill=fg, anchor="mm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading gf_run JSON...")
    gf_lookup = load_gf_lookup(GF_RUN_JSON)
    print(f"  {sum(len(v) for v in gf_lookup.values())} entries loaded "
          f"for {len(gf_lookup)} folders")

    # -- Layout maths -------------------------------------------------------
    n_cols  = len(IMG_KEYS)
    cell_h  = THUMB_H + LABEL_H
    row_h   = cell_h + ROW_PAD
    cell_w  = THUMB_W + COL_PAD
    total_w = FOLD_W + n_cols * cell_w + PARAMS_W + COL_PAD * 2
    total_h = HEADER_H + len(FAILED_FOLDERS) * row_h + ROW_PAD

    print(f"Canvas: {total_w} x {total_h} px")
    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # -- Header -------------------------------------------------------------
    draw.rectangle([0, 0, total_w, HEADER_H], fill=HEADER_BG)
    draw.text((FOLD_W // 2, HEADER_H // 2),
              "FOLDER", font=F_HEADER, fill=TEXT_W, anchor="mm")
    for ci, col_label in enumerate(COL_LABELS):
        cx = FOLD_W + ci * cell_w + THUMB_W // 2
        draw.text((cx, HEADER_H // 2), col_label,
                  font=F_HEADER, fill=TEXT_W, anchor="mm")
    px_hdr = FOLD_W + n_cols * cell_w + COL_PAD
    draw.text((px_hdr + PARAMS_W // 2, HEADER_H // 2),
              "D3/D6 + U3/U6 PARAMETERS",
              font=F_HEADER, fill=TEXT_W, anchor="mm")

    # -- Rows ---------------------------------------------------------------
    for ri, folder_id in enumerate(FAILED_FOLDERS):
        row_top = HEADER_H + ri * row_h
        print(f"  Row {ri+1}/{len(FAILED_FOLDERS)}: folder {folder_id}")

        # Alternating row background
        draw.rectangle([0, row_top, total_w, row_top + row_h - 1],
                       fill=ROW_BG if ri % 2 == 0 else ROW_BG2)

        # Folder ID sidebar
        draw.text((FOLD_W // 2, row_top + cell_h // 2),
                  folder_id, font=F_FOLDER, fill=TEXT_B, anchor="mm")

        folder_path   = os.path.join(ORIG_DIR, folder_id)
        folder_exists = os.path.isdir(folder_path)
        present       = set(os.listdir(folder_path)) if folder_exists else set()

        # Image cells
        for ci, img_key in enumerate(IMG_KEYS):
            cell_x   = FOLD_W + ci * cell_w
            img_path = os.path.join(folder_path, img_key)

            if img_key in present and os.path.isfile(img_path):
                try:
                    thumb = load_thumb(img_path)
                    canvas.paste(thumb, (cell_x, row_top))
                    draw_label_strip(draw, cell_x, row_top + THUMB_H, THUMB_W,
                                     img_key, OK_BG, TEXT_G)
                except Exception as exc:
                    canvas.paste(missing_thumb(f"ERR"), (cell_x, row_top))
                    draw_label_strip(draw, cell_x, row_top + THUMB_H, THUMB_W,
                                     f"{img_key} [READ ERR]", MISS_BG, TEXT_R)
                    print(f"    WARNING: could not load {img_path}: {exc}")
            else:
                canvas.paste(missing_thumb(img_key), (cell_x, row_top))
                draw_label_strip(draw, cell_x, row_top + THUMB_H, THUMB_W,
                                 img_key + " [NOT FOUND]", MISS_BG, TEXT_R)

        # Parameters panel
        px = FOLD_W + n_cols * cell_w + COL_PAD
        draw.rectangle([px, row_top, px + PARAMS_W - 2, row_top + cell_h - 1],
                       fill=PARAM_BG, outline=DIVIDER)

        folder_entries = gf_lookup.get(folder_id, {})

        # Build lines for D3/D6 pair and U3/U6 pair
        param_lines = []
        param_lines.append((0, f"FOLDER  {folder_id}", TEXT_B))
        param_lines.append((0, "", TEXT_DIM))

        for before_prefix, pair_label in [("D3", "D3/D6"), ("U3", "U3/U6")]:
            entry = folder_entries.get(before_prefix, None)
            param_lines.extend(build_pair_lines(pair_label, entry))

        # Draw parameter lines
        line_h = 15
        py     = row_top + 5
        for indent_px, text, col in param_lines:
            if text:
                lx = px + 6 + indent_px
                draw.text((lx, py), text, font=F_PARAM, fill=col)
            py += line_h

        # Row divider
        draw.line([(0, row_top + row_h - 1), (total_w, row_top + row_h - 1)],
                  fill=DIVIDER, width=1)

    canvas.save(OUT_FILE, quality=90, optimize=True)
    print(f"\nSaved : {OUT_FILE}")
    print(f"Size  : {canvas.width} x {canvas.height} px")


if __name__ == "__main__":
    main()
