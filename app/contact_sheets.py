"""
Contact sheet generator for web pipeline results.

Creates thumbnail grids grouped by outcome (ACCEPTED, failed Stage 1/2/3).
Uses only Pillow — no cv2 required.
"""

import os

from PIL import Image, ImageDraw, ImageFont

# ── Layout constants ──────────────────────────────────────────────────────────
THUMB_W  = 120
THUMB_H  = 90
LABEL_H  = 18
HEADER_H = 30
FOLD_W   = 100
GAP      = 3
PANEL_W  = 280   # width of the right-hand metrics panel
LINE_H   = 10    # line height inside panels

# Max folders rendered per sheet
MAX_SHEET_FOLDERS = 100

# ── Colours ───────────────────────────────────────────────────────────────────
BG          = (30,  30,  30)
HEADER_BG   = (50,  50,  80)
ROW_BG_EVEN = (45,  45,  45)
ROW_BG_ODD  = (38,  38,  38)
MISS_BG     = (80,  40,  40)
OK_BG       = (40,  60,  40)
TEXT_W      = (230, 230, 230)
TEXT_G      = (100, 210, 100)
TEXT_R      = (230, 100, 100)
TEXT_B      = (130, 180, 240)
TEXT_Y      = (255, 200,  80)
TEXT_DIM    = (140, 140, 140)

# ── Image keys shown per outcome group ───────────────────────────────────────
_GROUP_KEYS = {
    "accepted":    ["D1", "D4", "U1", "U4", "D2", "D5", "U2", "U5", "D3", "D6", "U3", "U6"],
    "failed_d1d4": ["D1", "D4", "U1", "U4"],
    "failed_d2d5": ["D2", "D5", "U2", "U5"],
    "failed_d3d6": ["D3", "D6", "U3", "U6"],
}

_GROUP_LABELS = {
    "accepted":    "ACCEPTED",
    "failed_d1d4": "FAILED -- Stage 1 (D1/D4 + U1/U4 SIFT)",
    "failed_d2d5": "FAILED -- Stage 2 (D2/D5 + U2/U5 Washing)",
    "failed_d3d6": "FAILED -- Stage 3 (D3/D6 + U3/U6 Geometry)",
}

# ── Stage 3N washing signal metadata ─────────────────────────────────────────
_WASHING_SIGNAL_ROWS = [
    ("KP ratio",     "wt=10%", lambda m: f"{m.get('kp_ratio', 0):.2f}"),
    ("Std dev incr", "wt=15%", lambda m: f"{m.get('std_increase_pct', 0):.1f}%"),
    ("Entropy incr", "wt=25%", lambda m: f"{m.get('entropy_increase', m.get('entropy_after', 0) - m.get('entropy_before', 0)):.2f}"),
    ("Match ratio",  "wt=10%", lambda m: f"{m.get('match_ratio', 0):.3f}"),
    ("Edge incr",    "wt=25%", lambda m: f"{m.get('edge_increase_pct', 0):.1f}%"),
    ("Lap var incr", "wt=15%", lambda m: f"{m.get('lap_increase_pct', 0):.1f}%"),
]
_WASHING_NOTES = ["", "", "", " (lower=better)", "", ""]


def _try_font(size: int):
    for name in ["arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf",
                 "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _find_image(folder_path: str, key: str) -> str | None:
    """Return full path to key image (e.g. D1, U4) in folder_path, any extension."""
    if not os.path.isdir(folder_path):
        return None
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(folder_path, key + ext)
        if os.path.isfile(p):
            return p
    for fname in os.listdir(folder_path):
        base, _ = os.path.splitext(fname)
        if base.upper() == key.upper():
            return os.path.join(folder_path, fname)
    return None


def _load_thumb(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), (20, 20, 20))
    canvas.paste(img, ((THUMB_W - img.width) // 2, (THUMB_H - img.height) // 2))
    return canvas


def _missing_thumb() -> Image.Image:
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), MISS_BG)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, THUMB_W - 1, THUMB_H - 1], outline=(160, 60, 60), width=2)
    d.text((THUMB_W // 2, THUMB_H // 2), "MISSING",
           font=_try_font(14), fill=(255, 100, 100), anchor="mm")
    return canvas


def _resolve_folder_path(input_dir: str, folder_name: str) -> str:
    direct = os.path.join(input_dir, folder_name)
    if os.path.isdir(direct):
        return direct
    for sub in ("no_obstruction", "obstruction"):
        alt = os.path.join(input_dir, sub, folder_name)
        if os.path.isdir(alt):
            return alt
    return direct


# ── Shared panel helpers ──────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _score_colour(score: float):
    if score >= 0.6:
        return TEXT_G
    if score >= 0.25:
        return TEXT_Y
    return TEXT_R


def _tier_colour(tier: str):
    if tier == "HIGH":
        return TEXT_G
    if tier == "MEDIUM":
        return TEXT_Y
    return TEXT_R


def _verdict_colour(verdict: str):
    if verdict in ("PASS", "ACCEPTED"):
        return TEXT_G
    if verdict in ("REVIEW", "NEEDS_REVIEW"):
        return TEXT_Y
    return TEXT_R


def _tline(draw, cx: int, cy: int, left: str, right: str,
           font, left_col=None, right_col=None) -> int:
    """Draw left+right parts on the same line; return next cy."""
    draw.text((cx, cy), left, font=font, fill=left_col or TEXT_W)
    w = draw.textlength(left, font=font)
    draw.text((cx + w, cy), right, font=font, fill=right_col or TEXT_W)
    return cy + LINE_H


def _section_header(draw, cx: int, cy: int, text: str, font) -> int:
    draw.text((cx, cy), text, font=font, fill=(180, 220, 255))
    return cy + LINE_H + 1


# ── Stage 2N SIFT panel (failed_d1d4) ────────────────────────────────────────

def _draw_sift_panel(draw, x: int, y: int, pair_stats: dict,
                     min_inliers: int, min_coverage: float, font):
    """Draw RANSAC inlier / coverage metrics for D and U pairs."""
    cx, cy = x + 5, y + 5

    for pair_key, pair_label in [("D", "D1/D4 pair"), ("U", "U1/U4 pair")]:
        ps = pair_stats.get(pair_key, {})
        cy = _section_header(draw, cx, cy, f"-- {pair_label} --", font)

        if not ps:
            draw.text((cx, cy), "  no data", font=font, fill=TEXT_R)
            cy += LINE_H * 4 + 5
            continue

        status = ps.get("status", "?")
        if status in ("MISSING_IMAGES", "FAILED"):
            draw.text((cx, cy), f"  Status: {status}", font=font, fill=TEXT_R)
            cy += LINE_H * 3 + 5
            continue
        if status.startswith("GATE_"):
            draw.text((cx, cy), f"  Status: {status}", font=font, fill=TEXT_Y)
            cy += LINE_H * 3 + 5
            continue

        inliers  = ps.get("ransac_inliers", 0)
        coverage = ps.get("inlier_coverage_pct", 0.0)
        in_pass  = inliers  >= min_inliers
        cov_pass = coverage >= min_coverage

        inlier_str = f"RANSAC inliers : {inliers:>4}  (need >= {min_inliers}) "
        cy = _tline(draw, cx, cy, inlier_str,
                    "PASS" if in_pass else "FAIL", font,
                    right_col=TEXT_G if in_pass else TEXT_R)

        cov_str = f"Spatial coverage: {coverage:>5.1f}%  (need >= {min_coverage:.0f}%) "
        cy = _tline(draw, cx, cy, cov_str,
                    "PASS" if cov_pass else "FAIL", font,
                    right_col=TEXT_G if cov_pass else TEXT_R)

        # Gate confidence (if detector was applied)
        for base in ("D1", "D4", "U1", "U4"):
            tier_key = f"gate_tier_{base}"
            conf_key = f"gate_conf_{base}"
            if tier_key in ps:
                tier = ps[tier_key]
                conf = ps.get(conf_key, 0.0)
                draw.text((cx, cy), f"  Gate {base}: {tier.upper()} ({conf:.2f})",
                          font=font, fill=_tier_colour(tier))
                cy += LINE_H

        cy += 5   # gap between pairs


# ── Stage 3N Washing panel (failed_d2d5) ─────────────────────────────────────

def _recompute_washing_scores(m: dict) -> list:
    entropy_inc = m.get("entropy_increase",
                        m.get("entropy_after", 0) - m.get("entropy_before", 0))
    return [
        _clamp((m.get("kp_ratio", 0) - 1.0) / 0.5),
        _clamp(m.get("std_increase_pct", 0) / 20.0),
        _clamp(entropy_inc / 0.5),
        _clamp((0.5 - m.get("match_ratio", 0.5)) / 0.45),
        _clamp(m.get("edge_increase_pct", 0) / 20.0),
        _clamp(m.get("lap_increase_pct", 0) / 30.0),
    ]


def _draw_washing_panel(draw, x: int, y: int, pair_results: dict,
                        pass_threshold: float, font):
    """Draw washing metrics for both pairs (D2 and U2)."""
    cx, cy = x + 5, y + 5

    for pair_key, pair_label in [("D2", "D2/D5 pair"), ("U2", "U2/U5 pair")]:
        pr      = pair_results.get(pair_key, {})
        metrics = pr.get("metrics") if pr.get("status") == "OK" else None
        cy = _section_header(draw, cx, cy, f"-- {pair_label} --", font)

        if not metrics:
            err = pr.get("error", pr.get("status", "no data"))
            draw.text((cx, cy), f"  {err}", font=font, fill=TEXT_R)
            cy += LINE_H * 9 + 5
            continue

        draw.text((cx, cy), "6 Washing Signals (with weights):",
                  font=font, fill=TEXT_DIM)
        cy += LINE_H

        scores = _recompute_washing_scores(metrics)
        for i, (label, wt_str, fmt_fn) in enumerate(_WASHING_SIGNAL_ROWS):
            val_str   = fmt_fn(metrics)
            score     = scores[i]
            note      = _WASHING_NOTES[i]
            left_part = f"{i+1}. {label:<13} ({wt_str}): {val_str:<9}"
            score_str = f" score={score:.2f}{note}"
            draw.text((cx, cy), left_part, font=font, fill=TEXT_W)
            draw.text((cx + draw.textlength(left_part, font=font), cy),
                      score_str, font=font, fill=_score_colour(score))
            cy += LINE_H

        conf    = metrics.get("washing_confidence", 0)
        tier    = metrics.get("washing_tier", "?")
        verdict = "PASS" if tier == "HIGH" else "FAIL"

        cy = _tline(draw, cx, cy, f"Wash confidence : {conf:.2f} ",
                    f"[{tier}]", font, right_col=_tier_colour(tier))
        cy = _tline(draw, cx, cy, f"PASS threshold  : >= {pass_threshold:.2f} HIGH tier --> ",
                    verdict, font, right_col=_verdict_colour(verdict))
        cy += 5


# ── Stage 4N Geometry panel (failed_d3d6) ────────────────────────────────────

def _draw_geometry_panel(draw, x: int, y: int, pair_results: dict, font):
    """Draw geometry-first verdict metrics for D3 and U3 pairs."""
    cx, cy = x + 5, y + 5

    for pair_key, pair_label in [("D3", "D3/D6 pair"), ("U3", "U3/U6 pair")]:
        pr = pair_results.get(pair_key, {})
        cy = _section_header(draw, cx, cy, f"-- {pair_label} --", font)

        if not pr:
            draw.text((cx, cy), "  no data", font=font, fill=TEXT_R)
            cy += LINE_H * 4 + 5
            continue

        status = pr.get("status", "?")
        score  = pr.get("score", 0)

        cy = _tline(draw, cx, cy, f"Status : {status}",
                    f"  ({score}/5 signals)", font,
                    left_col=_verdict_colour(status))

        grease  = pr.get("grease") or {}
        texture = pr.get("texture") or {}
        water   = pr.get("water") or {}

        g_pct     = grease.get("grease_pct", 0.0)
        g_flagged = grease.get("flagged", False)
        cy = _tline(draw, cx, cy,
                    f"Grease : {g_pct:.1f}%  ",
                    "[FLAGGED]" if g_flagged else "[ok]", font,
                    right_col=TEXT_R if g_flagged else TEXT_G)

        t_delta     = texture.get("entropy_delta", 0.0)
        t_confirmed = texture.get("confirmed", False)
        cy = _tline(draw, cx, cy,
                    f"Texture: delta={t_delta:+.3f}  ",
                    "[confirmed]" if t_confirmed else "[not confirmed]", font,
                    right_col=TEXT_G if t_confirmed else TEXT_Y)

        w_detected = water.get("water_detected", False)
        w_spec     = water.get("specular_pct", 0.0)
        w_blue     = water.get("blue_water_pct", 0.0)
        cy = _tline(draw, cx, cy,
                    f"Water  : spec={w_spec:.1f}%  blue={w_blue:.1f}%  ",
                    "[detected]" if w_detected else "[not detected]", font,
                    right_col=TEXT_G if w_detected else TEXT_DIM)

        cy += 5   # gap between pairs


# ── Accepted summary panel ────────────────────────────────────────────────────

def _draw_accepted_panel(draw, x: int, y: int, stage_data: dict, font):
    """Draw a compact all-stages summary for accepted folders."""
    cx, cy = x + 5, y + 5

    # Stage 2N
    s2 = stage_data.get("stage2n") or {}
    pair_stats = s2.get("pair_stats", {})
    cy = _section_header(draw, cx, cy, "Stage 2N (SIFT)", font)
    for pair_key, pair_label in [("D", "D1/D4"), ("U", "U1/U4")]:
        ps = pair_stats.get(pair_key, {})
        if ps.get("status") == "OK":
            inliers  = ps.get("ransac_inliers", 0)
            coverage = ps.get("inlier_coverage_pct", 0.0)
            draw.text((cx, cy),
                      f"  {pair_label}: {inliers} inliers  {coverage:.0f}% cov",
                      font=font, fill=TEXT_G)
        else:
            status = ps.get("status", "N/A")
            draw.text((cx, cy), f"  {pair_label}: {status}", font=font, fill=TEXT_DIM)
        cy += LINE_H

    # Stage 3N
    s3 = stage_data.get("stage3n") or {}
    pair_results_3 = s3.get("pair_results", {})
    cy += 2
    cy = _section_header(draw, cx, cy, "Stage 3N (Washing)", font)
    for pair_key, pair_label in [("D2", "D2/D5"), ("U2", "U2/U5")]:
        pr = pair_results_3.get(pair_key, {})
        metrics = pr.get("metrics") if pr.get("status") == "OK" else None
        if metrics:
            conf = metrics.get("washing_confidence", 0)
            tier = metrics.get("washing_tier", "?")
            left = f"  {pair_label}: conf={conf:.2f} "
            draw.text((cx, cy), left, font=font, fill=TEXT_W)
            draw.text((cx + draw.textlength(left, font=font), cy),
                      f"[{tier}]", font=font, fill=_tier_colour(tier))
        else:
            draw.text((cx, cy), f"  {pair_label}: {pr.get('status', 'N/A')}",
                      font=font, fill=TEXT_DIM)
        cy += LINE_H

    # Stage 4N
    s4 = stage_data.get("stage4n") or {}
    pair_results_4 = s4.get("pair_results", {})
    cy += 2
    cy = _section_header(draw, cx, cy, "Stage 4N (Geometry)", font)
    for pair_key, pair_label in [("D3", "D3/D6"), ("U3", "U3/U6")]:
        pr = pair_results_4.get(pair_key, {})
        status = pr.get("status", "N/A")
        score  = pr.get("score")
        if score is not None:
            left = f"  {pair_label}: score={score}/5  "
            draw.text((cx, cy), left, font=font, fill=TEXT_W)
            draw.text((cx + draw.textlength(left, font=font), cy),
                      status, font=font, fill=_verdict_colour(status))
        else:
            draw.text((cx, cy), f"  {pair_label}: {status}",
                      font=font, fill=TEXT_DIM)
        cy += LINE_H


# ── Row-height calculator ─────────────────────────────────────────────────────

def _row_height_for(panel_type: str | None) -> int:
    """Return the row height needed for the given panel type."""
    if panel_type == "washing":
        # 2 pairs × (header + signals-title + 6 signals + confidence + threshold)
        # = 2 × 10 lines + 1 gap + top/bottom padding
        return (2 * 10 + 1) * LINE_H + 22
    # All other panels fit in the default row height
    return THUMB_H + LABEL_H + GAP


# ── Core sheet builder ────────────────────────────────────────────────────────

def _make_contact_sheet(
    folders: list,
    input_dir: str,
    keys: list,
    title: str,
    out_path: str,
    panel_type: str | None = None,
    per_folder_data: dict | None = None,
) -> str:
    """Render a thumbnail grid and save to out_path. Returns out_path."""
    try:
        from pipeline_config import (
            STAGE2N_MIN_INLIERS, STAGE2N_MIN_COVERAGE_PCT,
            STAGE3N_HIGH_CONFIDENCE,
        )
    except ImportError:
        STAGE2N_MIN_INLIERS      = 10
        STAGE2N_MIN_COVERAGE_PCT = 25.0
        STAGE3N_HIGH_CONFIDENCE  = 0.50

    has_panel = panel_type is not None and bool(per_folder_data is not None)
    n_cols    = len(keys)
    col_w     = THUMB_W + GAP
    images_w  = FOLD_W + n_cols * col_w
    total_w   = images_w + (PANEL_W if has_panel else 0)
    row_h     = _row_height_for(panel_type) if has_panel else (THUMB_H + LABEL_H + GAP)
    total_h   = HEADER_H + len(folders) * row_h

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    F_hdr    = _try_font(13)
    F_lbl    = _try_font(11)
    F_fdr    = _try_font(11)
    F_metric = _try_font(9)

    draw.rectangle([0, 0, total_w - 1, HEADER_H - 1], fill=HEADER_BG)
    draw.text((total_w // 2, HEADER_H // 2), title, font=F_hdr, fill=TEXT_W, anchor="mm")

    for ri, folder_name in enumerate(folders):
        row_top = HEADER_H + ri * row_h
        row_bg  = ROW_BG_EVEN if ri % 2 == 0 else ROW_BG_ODD
        draw.rectangle([0, row_top, total_w - 1, row_top + row_h - 2], fill=row_bg)

        draw.text(
            (FOLD_W // 2, row_top + row_h // 2),
            folder_name, font=F_fdr, fill=TEXT_B, anchor="mm",
        )

        folder_path = _resolve_folder_path(input_dir, folder_name)
        img_top     = row_top + (row_h - THUMB_H - LABEL_H) // 2

        for ci, key in enumerate(keys):
            cell_x   = FOLD_W + ci * col_w
            img_path = _find_image(folder_path, key)

            if img_path:
                try:
                    thumb = _load_thumb(img_path)
                    canvas.paste(thumb, (cell_x, img_top))
                    draw.rectangle(
                        [cell_x, img_top + THUMB_H,
                         cell_x + THUMB_W - 1, img_top + THUMB_H + LABEL_H - 1],
                        fill=OK_BG,
                    )
                    draw.text(
                        (cell_x + THUMB_W // 2, img_top + THUMB_H + LABEL_H // 2),
                        key, font=F_lbl, fill=TEXT_G, anchor="mm",
                    )
                except Exception:
                    canvas.paste(_missing_thumb(), (cell_x, img_top))
                    draw.rectangle(
                        [cell_x, img_top + THUMB_H,
                         cell_x + THUMB_W - 1, img_top + THUMB_H + LABEL_H - 1],
                        fill=MISS_BG,
                    )
                    draw.text(
                        (cell_x + THUMB_W // 2, img_top + THUMB_H + LABEL_H // 2),
                        key + " [ERR]", font=F_lbl, fill=TEXT_R, anchor="mm",
                    )
            else:
                canvas.paste(_missing_thumb(), (cell_x, img_top))
                draw.rectangle(
                    [cell_x, img_top + THUMB_H,
                     cell_x + THUMB_W - 1, img_top + THUMB_H + LABEL_H - 1],
                    fill=MISS_BG,
                )
                draw.text(
                    (cell_x + THUMB_W // 2, img_top + THUMB_H + LABEL_H // 2),
                    key + " [MISSING]", font=F_lbl, fill=TEXT_R, anchor="mm",
                )

        if has_panel:
            folder_data = (per_folder_data or {}).get(folder_name, {})
            if panel_type == "sift":
                _draw_sift_panel(draw, images_w, row_top,
                                 folder_data.get("pair_stats", {}),
                                 STAGE2N_MIN_INLIERS, STAGE2N_MIN_COVERAGE_PCT,
                                 F_metric)
            elif panel_type == "washing":
                _draw_washing_panel(draw, images_w, row_top,
                                    folder_data,
                                    STAGE3N_HIGH_CONFIDENCE, F_metric)
            elif panel_type == "geometry":
                _draw_geometry_panel(draw, images_w, row_top,
                                     folder_data.get("pair_results", {}),
                                     F_metric)
            elif panel_type == "accepted":
                _draw_accepted_panel(draw, images_w, row_top, folder_data, F_metric)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, quality=85, optimize=True)
    canvas.close()
    del canvas
    return out_path


# ── Public API ────────────────────────────────────────────────────────────────

def generate_contact_sheets(
    pipeline_results: dict,
    output_dir: str,
    input_dir: str,
) -> dict:
    """
    Generate contact sheets for each outcome group from pipeline results.

    Parameters
    ----------
    pipeline_results : dict   Full results dict returned by run_pipeline().
    output_dir       : str   Job output directory (sheets saved under contact_sheets/).
    input_dir        : str   Directory containing the extracted job folders.

    Returns
    -------
    dict with keys: "accepted", "failed_d1d4", "failed_d2d5", "failed_d3d6"
    Each value is either an absolute path string or None (group had 0 folders).
    """
    groups: dict[str, list[str]] = {
        "accepted":    [],
        "failed_d1d4": [],
        "failed_d2d5": [],
        "failed_d3d6": [],
    }

    # Per-folder stage data for panels
    panel_data: dict[str, dict[str, dict]] = {
        "accepted":    {},
        "failed_d1d4": {},
        "failed_d2d5": {},
        "failed_d3d6": {},
    }

    for f in pipeline_results.get("folders", []):
        status = f.get("status", "")
        stage  = f.get("failed_stage") or ""
        name   = f.get("name", "")
        detail = f.get("detail", {})

        # Resolve stage2n / stage3n / stage4n regardless of no_obstruction vs obstruction path
        sift   = detail.get("sift_stages") or {}
        s2     = detail.get("stage2n") or sift.get("stage2n")
        s3     = detail.get("stage3n") or sift.get("stage3n")
        s4     = detail.get("stage4n") or sift.get("stage4n")

        if status == "ACCEPTED":
            groups["accepted"].append(name)
            panel_data["accepted"][name] = {
                "stage2n": s2, "stage3n": s3, "stage4n": s4,
            }
        elif status in ("REJECTED", "NEEDS_REVIEW"):
            if "D1D4" in stage or "U1U4" in stage:
                groups["failed_d1d4"].append(name)
                if s2:
                    panel_data["failed_d1d4"][name] = s2
            elif "D2D5" in stage or "U2U5" in stage:
                groups["failed_d2d5"].append(name)
                if s3:
                    panel_data["failed_d2d5"][name] = s3.get("pair_results", {})
            elif "D3D6" in stage or "U3U6" in stage:
                groups["failed_d3d6"].append(name)
                if s4:
                    panel_data["failed_d3d6"][name] = s4
            else:
                groups["failed_d1d4"].append(name)
                if s2:
                    panel_data["failed_d1d4"][name] = s2

    sheets_dir = os.path.join(output_dir, "contact_sheets")
    result: dict[str, str | None] = {k: None for k in groups}
    file_names = {
        "accepted":    "accepted_contact_sheet.jpg",
        "failed_d1d4": "failed_D1D4_U1U4_contact_sheet.jpg",
        "failed_d2d5": "failed_D2D5_U2U5_contact_sheet.jpg",
        "failed_d3d6": "failed_D3D6_U3U6_contact_sheet.jpg",
    }
    panel_types = {
        "accepted":    "accepted",
        "failed_d1d4": "sift",
        "failed_d2d5": "washing",
        "failed_d3d6": "geometry",
    }

    for group_key, folder_list in groups.items():
        if not folder_list:
            continue
        # Cap to avoid excessive memory use on large batches
        if len(folder_list) > MAX_SHEET_FOLDERS:
            folder_list = folder_list[:MAX_SHEET_FOLDERS]
        out_path = os.path.join(sheets_dir, file_names[group_key])
        try:
            _make_contact_sheet(
                folder_list,
                input_dir,
                _GROUP_KEYS[group_key],
                _GROUP_LABELS[group_key],
                out_path,
                panel_type=panel_types[group_key],
                per_folder_data=panel_data[group_key],
            )
            result[group_key] = out_path
        except Exception:
            import traceback
            traceback.print_exc()

    return result
