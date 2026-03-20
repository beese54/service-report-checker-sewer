"""
generate_triage_jpeg.py — Failure Triage JPEG Generator
=========================================================
Stitches the top N failure cases into a single JPEG for rapid visual triage.
Each row shows: folder ID + stage label | Before image | After image | Metric panel.

Usage:
    python analytics/generate_triage_jpeg.py \\
        --run results/pipeline_run_<ts>.json \\
        --originals original_images/ \\
        [--output triage_<ts>.jpg] \\
        [--top-n 3]

Failure ranking:
    1. Stage priority: Stage 2N < Stage 3N < Stage 4N  (earlier = more informative)
    2. Within stage: worst metric value first (fewest inliers / lowest confidence / worst blur)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from PIL import Image, ImageDraw, ImageFont

# ── Layout ────────────────────────────────────────────────────────────────────
IMG_W      = 360
IMG_H      = 270
LABEL_H    = 22
ROW_H      = IMG_H + LABEL_H          # 292 px
SIDEBAR_W  = 120
METRIC_W   = 200
HEADER_H   = 60
CANVAS_W   = SIDEBAR_W + IMG_W + IMG_W + METRIC_W   # 1040 px

# ── Colours ───────────────────────────────────────────────────────────────────
BG          = (30,  30,  30)
HEADER_BG   = (50,  50,  80)
ROW_BG_EVEN = (45,  45,  45)
ROW_BG_ODD  = (38,  38,  38)
MISS_BG     = (80,  40,  40)
LABEL_BG    = (60,  40,  40)
METRIC_BG   = (35,  35,  55)
TEXT_W      = (230, 230, 230)
TEXT_G      = (100, 210, 100)
TEXT_R      = (230, 100, 100)
TEXT_Y      = (255, 200,  60)
TEXT_B      = (130, 180, 240)

# ── Stage priority (lower = ranked first) ─────────────────────────────────────
_STAGE_PRIORITY = {
    "D1D4_U1U4": 0,
    "U1U4_D1D4": 0,
    "D2D5_U2U5": 1,
    "U2U5_D2D5": 1,
    "D3D6_U3U6": 2,
    "U3U6_D3D6": 2,
    "OCR_FAILED": 3,
}

# Before/after image keys per stage
_STAGE_IMAGES = {
    "D1D4_U1U4": [("D1", "D4"), ("U1", "U4")],
    "D2D5_U2U5": [("D2", "D5"), ("U2", "U5")],
    "D3D6_U3U6": [("D3", "D6"), ("U3", "U6")],
    "OCR_FAILED": [("D1", "D4"), ("U1", "U4")],
}


# =============================================================================
# Helpers
# =============================================================================

def _try_font(size: int) -> ImageFont.ImageFont:
    for name in ["arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf",
                 "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _find_image(folder_path: str, key: str) -> str | None:
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
    img.thumbnail((IMG_W, IMG_H), Image.LANCZOS)
    canvas = Image.new("RGB", (IMG_W, IMG_H), (20, 20, 20))
    canvas.paste(img, ((IMG_W - img.width) // 2, (IMG_H - img.height) // 2))
    return canvas


def _missing_thumb(label: str = "MISSING") -> Image.Image:
    canvas = Image.new("RGB", (IMG_W, IMG_H), MISS_BG)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, IMG_W - 1, IMG_H - 1], outline=(160, 60, 60), width=2)
    d.text((IMG_W // 2, IMG_H // 2), label, font=_try_font(14),
           fill=(255, 100, 100), anchor="mm")
    return canvas


def _wrap_text(text: str, max_chars: int = 22) -> list[str]:
    """Naive word-wrap for the metric panel."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text]


# =============================================================================
# Failure extraction + ranking
# =============================================================================

def _worst_metric(folder: dict) -> float:
    """Return a sort key (lower = worse = ranked first) for a rejected folder."""
    stage    = folder.get("failed_stage", "")
    detail   = folder.get("detail") or {}
    priority = _STAGE_PRIORITY.get(stage, 9)

    if priority == 0:   # Stage 2N — use max ransac inliers (higher inliers = better alignment)
        s2n = detail.get("stage2n") or {}
        ps  = s2n.get("pair_stats") or {}
        vals = [
            ps.get("D", {}).get("ransac_inliers") or 0,
            ps.get("U", {}).get("ransac_inliers") or 0,
        ]
        return priority * 1000 + max(vals)

    if priority == 1:   # Stage 3N — use max washing confidence
        s3n = detail.get("stage3n") or {}
        pr  = s3n.get("pair_results") or {}
        vals = [
            pr.get("D2", {}).get("washing_confidence") or 0.0,
            pr.get("U2", {}).get("washing_confidence") or 0.0,
        ]
        return priority * 1000 + max(vals)

    if priority == 2:   # Stage 4N — use max blur score
        s4n = detail.get("stage4n") or {}
        pr  = s4n.get("pair_results") or {}
        blur_D = (pr.get("D3") or {}).get("gate", {}).get("blur_score") or 0.0
        blur_U = (pr.get("U3") or {}).get("gate", {}).get("blur_score") or 0.0
        return priority * 1000 + max(blur_D, blur_U)

    return priority * 1000


def _pick_images(folder: dict, originals_dir: str) -> tuple[Image.Image, Image.Image, str]:
    """Return (before_img, after_img, pair_label) for the most informative pair."""
    stage       = folder.get("failed_stage", "")
    folder_name = folder.get("name", "")
    folder_path = os.path.join(originals_dir, folder_name)
    pairs = _STAGE_IMAGES.get(stage, [("D1", "D4")])

    for before_key, after_key in pairs:
        b_path = _find_image(folder_path, before_key)
        a_path = _find_image(folder_path, after_key)
        if b_path or a_path:
            b_img = _load_thumb(b_path) if b_path else _missing_thumb(before_key)
            a_img = _load_thumb(a_path) if a_path else _missing_thumb(after_key)
            return b_img, a_img, f"{before_key}→{after_key}"

    return _missing_thumb("BEFORE"), _missing_thumb("AFTER"), "?→?"


def _metric_lines(folder: dict) -> list[str]:
    """Build metric panel text lines for this folder."""
    stage  = folder.get("failed_stage", "")
    detail = folder.get("detail") or {}

    if _STAGE_PRIORITY.get(stage, 9) == 0:   # Stage 2N
        s2n = detail.get("stage2n") or {}
        ps  = s2n.get("pair_stats") or {}
        lines = [f"Stage 2N REJECTED", ""]
        for prefix, pair_key in [("D", "D"), ("U", "U")]:
            p = ps.get(prefix) or {}
            if p.get("status") == "OK":
                inliers  = p.get("ransac_inliers", "?")
                coverage = p.get("inlier_coverage_pct", "?")
                lines += [
                    f"{pair_key} Inliers: {inliers}",
                    f"  (need >= 10)",
                    f"{pair_key} Coverage: {coverage}%",
                    f"  (need >= 25%)",
                    "",
                ]
            elif p.get("status") == "GATE_REJECT":
                conf = p.get(f"gate_conf_{prefix}1", 0.0)
                lines += [f"{pair_key} Gate: REJECTED", f"  conf={conf:.2f}", ""]
        return lines

    if _STAGE_PRIORITY.get(stage, 9) == 1:   # Stage 3N
        s3n = detail.get("stage3n") or {}
        pr  = s3n.get("pair_results") or {}
        lines = ["Stage 3N REJECTED", ""]
        for pair_key, result_key in [("D2", "D2"), ("U2", "U2")]:
            p = pr.get(result_key) or {}
            if p.get("status") == "OK":
                conf = p.get("washing_confidence", 0.0)
                tier = p.get("washing_tier", "?")
                lines += [
                    f"{pair_key} Conf: {conf:.3f}",
                    f"  Tier: {tier}",
                    f"  Need: HIGH (>=0.50)",
                    "",
                ]
            elif p.get("status") == "FAILED":
                lines += [f"{pair_key}: FAILED", f"  {p.get('error','')[:20]}", ""]
        return lines

    if _STAGE_PRIORITY.get(stage, 9) == 2:   # Stage 4N
        s4n = detail.get("stage4n") or {}
        pr  = s4n.get("pair_results") or {}
        lines = ["Stage 4N REJECTED", ""]
        for pair_key in ("D3", "U3"):
            p = pr.get(pair_key) or {}
            status = p.get("status", "")
            if status in ("PASS", "REVIEW", "FAIL", "GATE_REJECTED"):
                blur  = (p.get("gate") or {}).get("blur_score", "?")
                score = p.get("score", "?")
                lines += [
                    f"{pair_key} Status: {status}",
                    f"  Blur: {blur:.1f}" if isinstance(blur, float) else f"  Blur: {blur}",
                    f"  Score: {score}/5",
                    "",
                ]
        return lines

    return [f"Stage: {stage}", "REJECTED"]


# =============================================================================
# Triage JPEG builder
# =============================================================================

def generate_triage_jpeg(
    run_path: str,
    originals_dir: str,
    output_path: str | None = None,
    top_n: int = 3,
):
    with open(run_path, encoding="utf-8") as f:
        run_json = json.load(f)

    # Collect rejected folders
    rejected = [
        fld for fld in run_json.get("folders", [])
        if fld.get("status") == "REJECTED"
    ]

    if not rejected:
        print("No REJECTED folders found in run JSON.")
        return None

    # Sort: stage priority first, then worst metric within stage
    rejected.sort(key=_worst_metric)
    top_failures = rejected[:top_n]

    # Build canvas
    n_rows    = len(top_failures)
    canvas_h  = HEADER_H + n_rows * ROW_H
    canvas    = Image.new("RGB", (CANVAS_W, canvas_h), BG)
    draw      = ImageDraw.Draw(canvas)

    F_large = _try_font(16)
    F_med   = _try_font(13)
    F_small = _try_font(11)

    # Header
    draw.rectangle([0, 0, CANVAS_W - 1, HEADER_H - 1], fill=HEADER_BG)
    run_date = run_json.get("run_date", "")
    draw.text(
        (CANVAS_W // 2, HEADER_H // 2 - 6),
        "WRN Pipeline — Failure Triage",
        font=F_large, fill=TEXT_W, anchor="mm",
    )
    draw.text(
        (CANVAS_W // 2, HEADER_H // 2 + 12),
        f"Run: {run_date}  |  Top {n_rows} failures  |  Before → After",
        font=F_small, fill=(150, 150, 180), anchor="mm",
    )

    # Column header labels
    col_xs = {
        "sidebar": 0,
        "before":  SIDEBAR_W,
        "after":   SIDEBAR_W + IMG_W,
        "metrics": SIDEBAR_W + IMG_W + IMG_W,
    }

    for ri, folder in enumerate(top_failures):
        row_top = HEADER_H + ri * ROW_H
        row_bg  = ROW_BG_EVEN if ri % 2 == 0 else ROW_BG_ODD
        draw.rectangle([0, row_top, CANVAS_W - 1, row_top + ROW_H - 1], fill=row_bg)

        # Sidebar: folder ID + stage
        stage = folder.get("failed_stage", "?")
        stage_label = stage.replace("_", " ")
        draw.text(
            (SIDEBAR_W // 2, row_top + IMG_H // 2 - 10),
            folder.get("name", "?"),
            font=F_med, fill=TEXT_B, anchor="mm",
        )
        draw.text(
            (SIDEBAR_W // 2, row_top + IMG_H // 2 + 10),
            stage_label,
            font=_try_font(10), fill=TEXT_Y, anchor="mm",
        )

        # Before + After images
        b_img, a_img, pair_label = _pick_images(folder, originals_dir)
        canvas.paste(b_img, (col_xs["before"], row_top))
        canvas.paste(a_img, (col_xs["after"],  row_top))

        # Image labels
        label_y = row_top + IMG_H
        for x_off, lbl, col in [
            (col_xs["before"], "BEFORE", TEXT_G),
            (col_xs["after"],  "AFTER",  TEXT_R),
        ]:
            draw.rectangle(
                [x_off, label_y, x_off + IMG_W - 1, label_y + LABEL_H - 1],
                fill=LABEL_BG,
            )
            draw.text(
                (x_off + IMG_W // 2, label_y + LABEL_H // 2),
                f"{lbl}  ({pair_label})",
                font=F_small, fill=col, anchor="mm",
            )

        # Metric panel
        mx = col_xs["metrics"]
        draw.rectangle([mx, row_top, mx + METRIC_W - 1, row_top + ROW_H - 1], fill=METRIC_BG)
        lines = _metric_lines(folder)
        y_text = row_top + 10
        for line in lines:
            if not line:
                y_text += 8
                continue
            is_title = not line.startswith(" ") and ":" not in line
            font  = F_med   if is_title else F_small
            color = TEXT_Y  if is_title else TEXT_W
            draw.text((mx + 8, y_text), line, font=font, fill=color)
            y_text += (16 if is_title else 14)
            if y_text > row_top + ROW_H - 10:
                break

    # Output path
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"triage_{ts}.jpg"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path, quality=90, optimize=True)
    print(f"Triage JPEG saved: {output_path}  ({n_rows} failure rows, {CANVAS_W}x{canvas_h}px)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate failure triage JPEG")
    parser.add_argument("--run",       required=True, help="Pipeline run JSON path")
    parser.add_argument("--originals", required=True, help="original_images/ directory")
    parser.add_argument("--output",    default=None,  help="Output JPEG path")
    parser.add_argument("--top-n",     type=int, default=3, help="Number of failure rows")
    args = parser.parse_args()
    generate_triage_jpeg(args.run, args.originals, args.output, args.top_n)


if __name__ == "__main__":
    main()
