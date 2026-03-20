"""
Crop pipeline output images for the GitHub examples directory.
Run from wrn_service_report_checker_for_git/:
    python docs/examples/crop_examples.py
Source images must be placed in docs/examples/source/ before running.
"""
from pathlib import Path
from PIL import Image
import numpy as np

SRC = Path(__file__).parent / "source"
OUT = Path(__file__).parent / "160389"
OUT.mkdir(parents=True, exist_ok=True)

def crop_save(src_name, out_name, box=None):
    img = Image.open(SRC / src_name)
    if box:
        img = img.crop(box)
    img.save(OUT / out_name, quality=92)
    print(f"  {out_name}  {img.size}")

# -- Full images --------------------------------------------------------------
crop_save("U_keypoint_matches.jpg",  "U_keypoint_matches.jpg")
crop_save("U2_keypoint_matches.jpg", "U2_keypoint_matches.jpg")
crop_save("U3_keypoint_matches.jpg", "U3_keypoint_matches.jpg")

# -- Composite text sections (text starts at y=1368) --------------------------
crop_save("U_composite.jpg",  "U_composite_text.jpg",  box=(0, 1368, 2400, 2091))
crop_save("U2_composite.jpg", "U2_composite_text.jpg", box=(0, 1368, 2400, 2001))

# -- GF report text section (text starts at y=1422) ---------------------------
crop_save("U3_gf_report.jpg", "U3_gf_report_text.jpg", box=(0, 1422, 2400, 2018))

# -- 160389_gf_report.jpg: U3/U6 text only ------------------------------------
# D3 and U3 reports stacked vertically; each ~2018px tall.
# U3 report starts at 4196-2018=2178; its text section starts at 2178+1422=3600.
crop_save("160389_gf_report.jpg", "160389_gf_report_U3_text.jpg", box=(0, 3600, 2400, 4196))

# -- 160389_report.jpg: U2/U5 visual rows only --------------------------------
# Scan for section bars (rows where almost all pixels are very dark).
# Section bars are ~56px tall bands with R,G,B all < 50.
DARK = 50
img_arr = np.array(Image.open(SRC / "160389_report.jpg"))
h = img_arr.shape[0]

# Find rows that are predominantly dark (section bar rows)
bar_rows = [y for y in range(h) if img_arr[y, :, :3].max(axis=1).mean() < DARK]

# Group contiguous dark rows into bands
bands = []
prev = -10
start = None
for y in bar_rows:
    if y - prev > 5:
        if start is not None:
            bands.append((start, prev))
        start = y
    prev = y
if start is not None:
    bands.append((start, prev))

print(f"  Found {len(bands)} section bars in 160389_report.jpg")

# U2 section is the first section bar past the image midpoint
mid = h // 2
u2_band = next((b for b in bands if b[0] > mid), None)
if u2_band:
    u2_start = u2_band[0]
    u2_end   = u2_start + 56 + 900 + 2001  # bar + keypoint_matches + composite
    result   = Image.fromarray(img_arr[u2_start:u2_end])
    result.save(OUT / "160389_report_U2_visuals.jpg", quality=92)
    print(f"  160389_report_U2_visuals.jpg  {result.size}  (rows {u2_start}-{u2_end})")
else:
    print("  WARNING: Could not locate U2 section bar — check section bar detection")
