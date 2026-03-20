"""
Level 2 — Obstruction folders: OCR text extraction
====================================================
Scans every subfolder inside adjusted_images/obstruction/.
For each folder it looks for: D1, U1, DR, UR, DL, UL (any extension).

For each matching image it:
  1. Runs PaddleOCR (angle-aware, handles slanted blackboard text).
  2. Draws numbered bounding boxes on the image, colour-coded by confidence:
       green  >= 80%
       orange  50–79%
       red    <  50%
  3. Stitches a text panel on the right listing each detection + confidence.
  4. Saves the combined image to:
       adjusted_images/obstruction/extracted_text/<folder_name>/<image_base>_annotated.jpg
  5. Extracts MH-XXX via regex and writes a results.json summary to:
       adjusted_images/obstruction/extracted_text/results.json
"""

import json
import os
import re
import textwrap

from PIL import Image, ImageDraw, ImageFont

# ── Paths ────────────────────────────────────────────────────────────────────
OBSTRUCTION_DIR = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\adjusted_images\obstruction"
EXTRACTED_TEXT_DIR = os.path.join(OBSTRUCTION_DIR, "extracted_text")
OUTPUT_JSON = os.path.join(EXTRACTED_TEXT_DIR, "results.json")

# ── Config ───────────────────────────────────────────────────────────────────
TARGET_BASES = {"D1", "U1", "DR", "UR", "DL", "UL"}
VALID_EXTS = {".jpg", ".jpeg", ".png"}
MH_PATTERN = re.compile(r"\bMH-\d+\b", re.IGNORECASE)
PANEL_WIDTH = 440  # pixels for the right-hand text panel

CONFIDENCE_COLORS = {
    "high": (30, 180, 30),    # green  >= 0.80
    "mid":  (220, 140, 0),    # orange  0.50–0.79
    "low":  (210, 30, 30),    # red    < 0.50
}

# ── OCR engine (lazy, model-guarded) ─────────────────────────────────────────
# PaddleOCR calls os._exit() if models are missing and it can't download them.
# We defer initialisation until first use and skip it entirely if the model
# directory is absent (e.g. air-gapped environments).
_ocr_engine = None
_ocr_unavailable = False   # set True once we confirm models are absent


def _models_present() -> bool:
    """Return True if the PaddleOCR model directory exists on disk."""
    paddle_dir = os.path.join(os.path.expanduser("~"), ".paddleocr")
    return os.path.isdir(paddle_dir)


def _get_ocr_engine():
    """Return the PaddleOCR singleton, or None if models are not available."""
    global _ocr_engine, _ocr_unavailable
    if _ocr_unavailable:
        return None
    if _ocr_engine is not None:
        return _ocr_engine
    if not _models_present():
        _ocr_unavailable = True
        print("[stage1_ocr] PaddleOCR models not found — OCR skipped (run with internet access to download models)")
        return None
    try:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except Exception as exc:
        _ocr_unavailable = True
        print(f"[stage1_ocr] PaddleOCR init failed — OCR skipped: {exc}")
    return _ocr_engine


# ── Helpers ──────────────────────────────────────────────────────────────────

def confidence_color(conf: float) -> tuple:
    if conf >= 0.80:
        return CONFIDENCE_COLORS["high"]
    if conf >= 0.50:
        return CONFIDENCE_COLORS["mid"]
    return CONFIDENCE_COLORS["low"]


def load_font(size: int) -> ImageFont.FreeTypeFont:
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


def build_combined_image(image_path: str, ocr_lines: list) -> Image.Image:
    """
    Returns a combined PIL image:
      left  — original image with numbered bounding boxes
      right — text panel listing each detection
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_label = load_font(14)

    # Draw bounding boxes + index labels on the image
    for idx, line in enumerate(ocr_lines, start=1):
        box, (text, conf) = line
        color = confidence_color(conf)
        pts = [(int(p[0]), int(p[1])) for p in box]

        # Draw quadrilateral outline (two polygon calls for a thicker look)
        draw.polygon(pts, outline=color)
        draw.polygon(pts, outline=color)

        # Number badge near the top-left corner of the box
        lx, ly = pts[0]
        draw.rectangle([lx - 1, ly - 17, lx + 16, ly - 1], fill=color)
        draw.text((lx + 1, ly - 17), str(idx), fill=(255, 255, 255), font=font_label)

    # ── Text panel ────────────────────────────────────────────────────────
    panel = Image.new("RGB", (PANEL_WIDTH, img.height), (245, 245, 245))
    pdraw = ImageDraw.Draw(panel)
    font_title = load_font(16)
    font_body = load_font(13)
    font_note = load_font(11)

    y = 12
    pdraw.text((10, y), "PaddleOCR — Extracted Text", fill=(30, 30, 30), font=font_title)
    y += 26
    pdraw.line([(10, y), (PANEL_WIDTH - 10, y)], fill=(180, 180, 180), width=1)
    y += 10

    # Legend
    for label, color in [("≥80% confidence", CONFIDENCE_COLORS["high"]),
                          ("50–79%",          CONFIDENCE_COLORS["mid"]),
                          ("<50%",            CONFIDENCE_COLORS["low"])]:
        pdraw.rectangle([10, y + 2, 20, y + 12], fill=color)
        pdraw.text((26, y), label, fill=(80, 80, 80), font=font_note)
        y += 16
    y += 6
    pdraw.line([(10, y), (PANEL_WIDTH - 10, y)], fill=(200, 200, 200), width=1)
    y += 10

    if not ocr_lines:
        pdraw.text((10, y), "No text detected.", fill=(150, 150, 150), font=font_body)
    else:
        for idx, line in enumerate(ocr_lines, start=1):
            _, (text, conf) = line
            color = confidence_color(conf)
            header = f"{idx}.  [{conf:.0%}]"
            pdraw.text((10, y), header, fill=color, font=font_body)
            y += 17

            for chunk in textwrap.wrap(text, width=46):
                if y > img.height - 18:
                    pdraw.text((10, y), "  … (truncated)", fill=(150, 150, 150), font=font_note)
                    break
                pdraw.text((18, y), chunk, fill=(40, 40, 40), font=font_body)
                y += 16
            y += 5

    # ── Stitch together ───────────────────────────────────────────────────
    combined = Image.new("RGB", (img.width + PANEL_WIDTH, img.height), (180, 180, 180))
    combined.paste(img, (0, 0))
    combined.paste(panel, (img.width, 0))
    return combined


# ── Core processing ──────────────────────────────────────────────────────────

def process_folder(folder_path: str, folder_name: str) -> dict:
    out_dir = os.path.join(EXTRACTED_TEXT_DIR, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    folder_results = {}

    for entry in os.scandir(folder_path):
        base, ext = os.path.splitext(entry.name)
        if base not in TARGET_BASES or ext.lower() not in VALID_EXTS:
            continue

        # Run OCR
        engine = _get_ocr_engine()
        raw = engine.ocr(entry.path, cls=True) if engine else None
        lines = raw[0] if raw and raw[0] else []

        # Build and save combined image
        combined = build_combined_image(entry.path, lines)
        save_path = os.path.join(out_dir, f"{base}_annotated.jpg")
        combined.save(save_path, quality=95)

        # Extract structured fields
        full_text = " ".join(line[1][0] for line in lines)
        mh_matches = MH_PATTERN.findall(full_text)

        folder_results[base] = {
            "manhole_number": mh_matches[0].upper() if mh_matches else None,
            "raw_lines": [
                {"text": line[1][0], "confidence": round(line[1][1], 3)}
                for line in lines
            ],
            "full_text": full_text,
        }

        mh = folder_results[base]["manhole_number"] or "not found"
        print(f"    [{base}] {len(lines)} regions  |  MH: {mh}")

    return folder_results


def run_ocr_on_single_folder(folder_path, folder_name, output_dir, progress_cb=None):
    """
    Run OCR on one obstruction folder and return a structured result dict.

    Parameters
    ----------
    folder_path : str   Path to the job folder (e.g. .../obstruction/146915)
    folder_name : str   Folder name used for output paths and reporting
    output_dir  : str   Root output directory; annotated images go to
                        <output_dir>/extracted_text/<folder_name>/
    progress_cb : callable, optional
                        Called with a status string after each image is processed.

    Returns
    -------
    dict  ``{"status": "ACCEPTED"|"FAILED", "images_processed": int,
             "images_failed": int, "ocr_data": {...}}``
    """
    extracted_dir = os.path.join(output_dir, "extracted_text")
    out_dir = os.path.join(extracted_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    folder_results = {}
    images_processed = 0
    images_failed = 0

    for entry in os.scandir(folder_path):
        base, ext = os.path.splitext(entry.name)
        if base not in TARGET_BASES or ext.lower() not in VALID_EXTS:
            continue
        try:
            engine = _get_ocr_engine()
            raw = engine.ocr(entry.path, cls=True) if engine else None
            lines = raw[0] if raw and raw[0] else []

            combined = build_combined_image(entry.path, lines)
            save_path = os.path.join(out_dir, f"{base}_annotated.jpg")
            combined.save(save_path, quality=95)

            full_text = " ".join(line[1][0] for line in lines)
            mh_matches = MH_PATTERN.findall(full_text)

            folder_results[base] = {
                "manhole_number": mh_matches[0].upper() if mh_matches else None,
                "raw_lines": [
                    {"text": line[1][0], "confidence": round(line[1][1], 3)}
                    for line in lines
                ],
                "full_text": full_text,
            }
            images_processed += 1
            if progress_cb:
                mh = folder_results[base]["manhole_number"] or "not found"
                progress_cb(f"    [{base}] {len(lines)} regions  |  MH: {mh}")
        except Exception as exc:
            images_failed += 1
            if progress_cb:
                progress_cb(f"    [{base}] FAILED -- {exc}")

    status = "ACCEPTED" if images_processed > 0 else "FAILED"
    return {
        "status":           status,
        "images_processed": images_processed,
        "images_failed":    images_failed,
        "ocr_data":         folder_results,
    }


def run_ocr_batch(obstruction_dir, output_dir, progress_cb=None):
    """
    Run OCR on all subfolders of *obstruction_dir*.

    Parameters
    ----------
    obstruction_dir : str   Directory containing obstruction job folders.
    output_dir      : str   Output root; results JSON and annotated images go here.
    progress_cb     : callable(msg), optional

    Returns
    -------
    dict   ``{"folders": {folder_name: result_dict, ...}, "json_path": str}``
    """
    extracted_dir = os.path.join(output_dir, "extracted_text")
    os.makedirs(extracted_dir, exist_ok=True)

    folders = sorted(
        [e for e in os.scandir(obstruction_dir)
         if e.is_dir() and e.name != "extracted_text"],
        key=lambda e: e.name,
    )

    if progress_cb:
        progress_cb(f"Found {len(folders)} obstruction folder(s). Starting OCR...")

    all_results = {}
    for folder in folders:
        if progress_cb:
            progress_cb(f"  OCR: {folder.name}")
        all_results[folder.name] = run_ocr_on_single_folder(
            folder.path, folder.name, output_dir, progress_cb
        )

    json_path = os.path.join(extracted_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return {"folders": all_results, "json_path": json_path}


def main():
    os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)

    folders = sorted(
        [e for e in os.scandir(OBSTRUCTION_DIR) if e.is_dir() and e.name != "extracted_text"],
        key=lambda e: e.name,
    )

    print(f"Found {len(folders)} obstruction folder(s). Starting OCR...\n")
    all_results = {}

    for folder in folders:
        print(f"  {folder.name}")
        all_results[folder.name] = process_folder(folder.path, folder.name)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone.")
    print(f"  Annotated images : {EXTRACTED_TEXT_DIR}")
    print(f"  JSON results     : {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
