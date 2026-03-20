"""
make_failed_contact_sheet_d2d5u2u5.py

Generates a contact sheet for folders that failed the D2/D5 + U2/U5 washing
confidence stage.  No cv2 required — all data comes from the JSON summary and
images are loaded with Pillow.

Usage:
    python make_failed_contact_sheet_d2d5u2u5.py
"""

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker")
ORIG_DIR        = BASE_DIR / "original_images"
JSON_PATH       = (
    BASE_DIR
    / "adjusted_images"
    / "no_obstruction"
    / "difference_analysis"
    / "D2D5U2U5_confidence_summary.json"
)
OUTPUT_PATH     = BASE_DIR / "failed_D2D5_U2U5_contact_sheet.jpg"

# ── Pass threshold (mirrors pipeline_config.py) ───────────────────────────────
STAGE3N_HIGH_CONFIDENCE = 0.50
PASS_TIER               = "HIGH"

# ── Failed folder list ────────────────────────────────────────────────────────
FAILED_FOLDERS = [
    "1159105","1159107","1159113","140914","152634","152635","152667","152668",
    "152837","157082","160159","160505","160522","160523","160526","160595",
    "163016","169280","169281","169282","172129","172144","172301","172308",
    "178257","178258","185217","185218","185220","185222","185229","185248",
    "185249","185250","187812","187996","188355","188356","188362","190462",
    "190463","196861","203427","203428","203558","203559","203565","203566",
    "203569","203588","203589","203593","204189","204534","204928","204929",
    "206638","208890","208895","208917","209057","3198332","321228","3400653","3474112",
]

# ── Colours (BGR-free; PIL uses RGB) ─────────────────────────────────────────
BG          = (30, 30, 30)
HEADER_BG   = (50, 50, 80)
ROW_BG      = (45, 45, 45)
ROW_BG2     = (38, 38, 38)
MISS_BG     = (80, 40, 40)
OK_BG       = (40, 60, 40)
WARN_BG     = (80, 70, 20)
PARAM_BG    = (22, 22, 40)
DIVIDER     = (70, 70, 90)
TEXT_W      = (230, 230, 230)
TEXT_Y      = (230, 220,  80)
TEXT_R      = (230, 100, 100)
TEXT_G      = (100, 210, 100)
TEXT_B      = (130, 180, 240)
TEXT_DIM    = (140, 140, 170)

# ── Layout constants ──────────────────────────────────────────────────────────
THUMB_W     = 360
THUMB_H     = 270
LABEL_H     = 22          # image label strip height
SIDEBAR_W   = 120         # folder-id sidebar
PARAM_W     = 480         # parameters panel
IMG_COLS    = 4           # D2, D5, U2, U5
TOTAL_IMG_W = THUMB_W * IMG_COLS
ROW_H       = THUMB_H + LABEL_H
FULL_W      = SIDEBAR_W + TOTAL_IMG_W + PARAM_W
HEADER_H    = 40

# ── Signal weights (must match stage3n_washing.py) ────────────────────────────
SIGNAL_WEIGHTS = [0.10, 0.15, 0.25, 0.10, 0.25, 0.15]

# ── Font helper ───────────────────────────────────────────────────────────────

def try_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Try common font names; fall back to Pillow default."""
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


# ── Score computation ─────────────────────────────────────────────────────────

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_scores(pair_data: dict) -> list[float]:
    """Return list of 6 per-signal scores mirroring stage3n_washing.py logic."""
    kp_ratio          = pair_data.get("kp_ratio",          0.0)
    std_increase_pct  = pair_data.get("std_increase_pct",  0.0)
    entropy_increase  = pair_data.get("entropy_increase",  0.0)
    match_ratio       = pair_data.get("match_ratio",        0.5)
    edge_increase_pct = pair_data.get("edge_increase_pct", 0.0)
    lap_increase_pct  = pair_data.get("lap_increase_pct",  0.0)

    kp_score      = clamp((kp_ratio - 1.0) / 0.5,                   0, 1)
    std_score     = clamp(std_increase_pct / 20.0,                   0, 1)
    entropy_score = clamp(entropy_increase / 0.5,                    0, 1)
    match_score   = clamp((0.5 - match_ratio) / (0.5 - 0.05),       0, 1)
    edge_score    = clamp(edge_increase_pct / 20.0,                  0, 1)
    lap_score     = clamp(lap_increase_pct / 30.0,                   0, 1)
    return [kp_score, std_score, entropy_score, match_score, edge_score, lap_score]


def score_colour(score: float) -> tuple:
    if score >= 0.7:
        return TEXT_G
    if score >= 0.30:
        return TEXT_Y
    return TEXT_R


def pair_passes(pair_data: dict) -> bool:
    return (
        pair_data.get("washing_confidence", 0.0) >= STAGE3N_HIGH_CONFIDENCE
        and pair_data.get("washing_tier", "") == PASS_TIER
    )


# ── Image loading ─────────────────────────────────────────────────────────────

def load_thumb(folder: str, image_name: str) -> Image.Image | None:
    """Load <folder>/<image_name>.jpg (or .png) and resize to THUMB_W×THUMB_H."""
    folder_path = ORIG_DIR / folder
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
        candidate = folder_path / (image_name + ext)
        if candidate.exists():
            try:
                img = Image.open(candidate).convert("RGB")
                img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
                return img
            except Exception:
                return None
    return None


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_text_left(draw: ImageDraw.ImageDraw, xy: tuple, text: str,
                   font: ImageFont.FreeTypeFont, fill: tuple) -> int:
    """Draw text at xy (left-aligned); return y after the line."""
    draw.text(xy, text, font=font, fill=fill)
    bbox = font.getbbox(text)
    return xy[1] + (bbox[3] - bbox[1]) + 2


def draw_params_panel(draw: ImageDraw.ImageDraw,
                      x0: int, y0: int,
                      folder_data: dict | None,
                      font_sm: ImageFont.FreeTypeFont,
                      font_bold: ImageFont.FreeTypeFont) -> None:
    """Render the right-hand parameters panel for one folder row."""
    x = x0 + 8
    y = y0 + 6

    if folder_data is None:
        draw.text((x, y), "No JSON data for this folder", font=font_sm, fill=TEXT_R)
        return

    pairs_data = {p["pair"]: p for p in folder_data.get("pairs", [])}

    signal_labels = [
        ("KP ratio",     "wt=10%"),
        ("Std dev incr", "wt=15%"),
        ("Entropy incr", "wt=25%"),
        ("Match ratio",  "wt=10%"),
        ("Edge incr",    "wt=25%"),
        ("Lap var incr", "wt=15%"),
    ]
    signal_fmt = [
        lambda v: f"{v:.2f}",
        lambda v: f"{v:.1f}%",
        lambda v: f"{v:.2f}",
        lambda v: f"{v:.3f}",
        lambda v: f"{v:.1f}%",
        lambda v: f"{v:.1f}%",
    ]
    signal_keys = [
        "kp_ratio", "std_increase_pct", "entropy_increase",
        "match_ratio", "edge_increase_pct", "lap_increase_pct",
    ]
    note_labels = [
        "",
        "",
        "",
        "(lower=better)",
        "",
        "",
    ]

    for pair_label in ["D2/D5", "U2/U5"]:
        # Section header
        draw.text((x, y), f"-- {pair_label} pair --", font=font_bold, fill=TEXT_B)
        y += 16

        pd = pairs_data.get(pair_label)
        if pd is None:
            draw.text((x + 4, y), "  No data", font=font_sm, fill=TEXT_R)
            y += 14
        else:
            scores = compute_scores(pd)
            draw.text((x + 4, y), "  6 Washing Signals (with weights):", font=font_sm, fill=TEXT_DIM)
            y += 13

            for i, (lbl, wt_str) in enumerate(signal_labels):
                raw_val = pd.get(signal_keys[i], 0.0)
                val_str = signal_fmt[i](raw_val)
                sc      = scores[i]
                sc_col  = score_colour(sc)
                note    = note_labels[i]

                line = f"  {i+1}. {lbl:<14} ({wt_str}) : {val_str:<8}  score={sc:.2f}"
                if note:
                    line += f"  {note}"
                draw.text((x + 4, y), line, font=font_sm, fill=sc_col)
                y += 13

            # Wash confidence
            wc    = pd.get("washing_confidence", 0.0)
            wt    = pd.get("washing_tier", "?")
            ok    = pair_passes(pd)
            col   = TEXT_G if ok else TEXT_R
            verdict = "PASS" if ok else "FAIL"
            draw.text(
                (x + 4, y),
                f"  Wash confidence   : {wc:.2f}  [{wt}]",
                font=font_sm, fill=col,
            )
            y += 13
            draw.text(
                (x + 4, y),
                f"  PASS threshold    : >= {STAGE3N_HIGH_CONFIDENCE} HIGH tier  --> {verdict}",
                font=font_sm, fill=col,
            )
            y += 16

        # Divider between pairs
        draw.line([(x, y), (x0 + PARAM_W - 8, y)], fill=DIVIDER, width=1)
        y += 4


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load JSON summary
    print(f"Loading JSON from {JSON_PATH} ...")
    if JSON_PATH.exists():
        with open(JSON_PATH, "r", encoding="utf-8") as fh:
            summary: dict = json.load(fh)
        print(f"  Loaded {len(summary)} folder entries from JSON.")
    else:
        print(f"  WARNING: JSON not found at {JSON_PATH}. Proceeding without metrics.")
        summary = {}

    font_sm   = try_font(11)
    font_bold = try_font(11, bold=True)
    font_hdr  = try_font(13, bold=True)
    font_id   = try_font(12, bold=True)

    n_rows    = len(FAILED_FOLDERS)
    total_h   = HEADER_H + n_rows * ROW_H
    print(f"Canvas size: {FULL_W} x {total_h}  ({n_rows} rows)")

    canvas = Image.new("RGB", (FULL_W, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # ── Header ────────────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (FULL_W, HEADER_H)], fill=HEADER_BG)
    draw.text(
        (SIDEBAR_W + 8, 10),
        "D2/D5 + U2/U5 Washing Confidence — FAILED Folders Contact Sheet",
        font=font_hdr, fill=TEXT_W,
    )
    col_labels = ["D2 (before)", "D5 (after)", "U2 (before)", "U5 (after)"]
    for ci, lbl in enumerate(col_labels):
        cx = SIDEBAR_W + ci * THUMB_W + THUMB_W // 2
        bbox = font_sm.getbbox(lbl)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, HEADER_H - 18), lbl, font=font_sm, fill=TEXT_DIM)

    # ── Rows ──────────────────────────────────────────────────────────────────
    img_names = ["D2", "D5", "U2", "U5"]

    for row_idx, folder in enumerate(FAILED_FOLDERS):
        y_top   = HEADER_H + row_idx * ROW_H
        row_col = ROW_BG if row_idx % 2 == 0 else ROW_BG2

        # Background
        draw.rectangle([(0, y_top), (FULL_W, y_top + ROW_H)], fill=row_col)
        # Param panel background
        px0 = SIDEBAR_W + TOTAL_IMG_W
        draw.rectangle([(px0, y_top), (FULL_W, y_top + ROW_H)], fill=PARAM_BG)

        # Sidebar — folder ID
        draw.rectangle([(0, y_top), (SIDEBAR_W, y_top + ROW_H)], fill=HEADER_BG)
        id_bbox = font_id.getbbox(folder)
        id_w    = id_bbox[2] - id_bbox[0]
        id_x    = max(4, (SIDEBAR_W - id_w) // 2)
        id_y    = y_top + (ROW_H - (id_bbox[3] - id_bbox[1])) // 2
        draw.text((id_x, id_y), folder, font=font_id, fill=TEXT_B)

        # Retrieve JSON data for this folder
        folder_data = summary.get(folder)

        # Build pass/fail map per image pair
        pairs_data = {}
        if folder_data:
            for pd in folder_data.get("pairs", []):
                pairs_data[pd["pair"]] = pd

        # Determine label strip colour per image
        def label_bg_for_image(img_name: str) -> tuple:
            """Return strip colour based on whether that pair passes."""
            pair_key = "D2/D5" if img_name in ("D2", "D5") else "U2/U5"
            if folder_data is None:
                return WARN_BG      # no JSON at all
            pd = pairs_data.get(pair_key)
            if pd is None:
                return WARN_BG      # missing pair in JSON
            return OK_BG if pair_passes(pd) else MISS_BG

        # Images
        for ci, img_name in enumerate(img_names):
            ix = SIDEBAR_W + ci * THUMB_W
            iy = y_top

            thumb = load_thumb(folder, img_name)
            if thumb is not None:
                canvas.paste(thumb, (ix, iy))
            else:
                # Grey placeholder
                draw.rectangle([(ix, iy), (ix + THUMB_W, iy + THUMB_H)], fill=(60, 60, 60))
                draw.text(
                    (ix + THUMB_W // 2 - 25, iy + THUMB_H // 2 - 7),
                    "NO IMAGE",
                    font=font_sm, fill=TEXT_R,
                )

            # Label strip at bottom of thumbnail
            strip_y = iy + THUMB_H
            strip_bg = label_bg_for_image(img_name)
            draw.rectangle(
                [(ix, strip_y), (ix + THUMB_W, strip_y + LABEL_H)],
                fill=strip_bg,
            )
            lbl_text = img_name
            lb = font_sm.getbbox(lbl_text)
            lw = lb[2] - lb[0]
            draw.text(
                (ix + (THUMB_W - lw) // 2, strip_y + 4),
                lbl_text,
                font=font_bold, fill=TEXT_W,
            )

        # Parameters panel
        draw_params_panel(
            draw, px0, y_top, folder_data, font_sm, font_bold
        )

        # Horizontal divider
        draw.line([(0, y_top + ROW_H - 1), (FULL_W, y_top + ROW_H - 1)], fill=DIVIDER)

        if (row_idx + 1) % 10 == 0 or (row_idx + 1) == n_rows:
            print(f"  Rendered {row_idx + 1}/{n_rows} rows ...")

    print(f"Saving to {OUTPUT_PATH} ...")
    canvas.save(str(OUTPUT_PATH), "JPEG", quality=90)
    print("Done.")


if __name__ == "__main__":
    main()
