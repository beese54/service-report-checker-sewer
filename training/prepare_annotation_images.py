"""
prepare_annotation_images.py
=============================
Copies images from all no_obstruction job folders into a single flat folder
with unique names: <jobfolder>_<filename>.jpg

This ensures labelImg saves each annotation to a unique .txt file instead of
overwriting the same D1.txt / U1.txt repeatedly.

Run once before annotating:
    python prepare_annotation_images.py

Then open dataset/images/to_annotate/ in labelImg and set save dir to
dataset/labels/train/
"""

import os
import shutil

SOURCE_DIR  = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\adjusted_images\no_obstruction"
OUTPUT_DIR  = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\dataset\images\to_annotate"

# Only copy these image types (one before/after pair per direction)
TARGET_NAMES = {"d1", "u1", "d4", "u4"}   # lowercase stems to annotate

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

copied = 0
skipped = 0

for job_folder in sorted(os.listdir(SOURCE_DIR)):
    job_path = os.path.join(SOURCE_DIR, job_folder)
    if not os.path.isdir(job_path):
        continue

    for fname in os.listdir(job_path):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTS:
            continue
        if stem.lower() not in TARGET_NAMES:
            continue

        # Unique name: jobfolder_D1.jpg
        new_name = f"{job_folder}_{fname}"
        src = os.path.join(job_path, fname)
        dst = os.path.join(OUTPUT_DIR, new_name)

        if os.path.exists(dst):
            skipped += 1
            continue

        shutil.copy2(src, dst)
        copied += 1

print(f"Done. Copied: {copied}  |  Already existed (skipped): {skipped}")
print(f"Images ready in: {OUTPUT_DIR}")
print()
print("Next steps:")
print("  1. Open labelImg")
print("  2. Open Dir -> dataset/images/to_annotate/")
print("  3. Change Save Dir -> dataset/labels/train/")
print("  4. Set format to YOLO")
print("  5. Annotate each image and save")
