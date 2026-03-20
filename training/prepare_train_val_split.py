"""
prepare_train_val_split.py
--------------------------
Distributes annotated images and labels into train/val splits (80/20).

Source images  : dataset/images/to_annotate/
Source labels  : dataset/labels/train/   (all .txt files land here from LabelImg)
Destinations   : dataset/images/train/
                 dataset/images/val/
                 dataset/labels/val/

Labels that already sit in labels/train/ are kept there for the train split,
and moved to labels/val/ for the val split.
Images are copied from to_annotate/ into images/train/ or images/val/.
"""

import random
import shutil
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
SRC_IMAGES    = ROOT / "dataset" / "images" / "to_annotate"
SRC_LABELS    = ROOT / "dataset" / "labels" / "train"
DST_IMG_TRAIN = ROOT / "dataset" / "images" / "train"
DST_IMG_VAL   = ROOT / "dataset" / "images" / "val"
DST_LBL_VAL   = ROOT / "dataset" / "labels" / "val"

# ── Config ───────────────────────────────────────────────────────────────────
VAL_FRACTION  = 0.20
RANDOM_SEED   = 42
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Setup ─────────────────────────────────────────────────────────────────────
for d in (DST_IMG_TRAIN, DST_IMG_VAL, DST_LBL_VAL):
    d.mkdir(parents=True, exist_ok=True)

# ── Step 1: collect all label files ──────────────────────────────────────────
all_labels = sorted(SRC_LABELS.glob("*.txt"))
print(f"Label files found in labels/train/  : {len(all_labels)}")

# ── Step 2: match each label to an image in to_annotate/ ─────────────────────
matched_pairs = []   # list of (label_path, image_path)
skipped_no_image = []

for lbl in all_labels:
    stem = lbl.stem
    img_path = None
    for ext in IMAGE_EXTS:
        candidate = SRC_IMAGES / (stem + ext)
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        skipped_no_image.append(lbl.name)
    else:
        matched_pairs.append((lbl, img_path))

print(f"Matched pairs (label + image found)  : {len(matched_pairs)}")
print(f"Skipped (label but no image)         : {len(skipped_no_image)}")

if not matched_pairs:
    print("\nNo matched pairs found — nothing to do. Check your paths.")
    raise SystemExit(1)

# ── Step 3: shuffle and split ─────────────────────────────────────────────────
random.seed(RANDOM_SEED)
random.shuffle(matched_pairs)

n_val   = max(1, round(len(matched_pairs) * VAL_FRACTION))
n_train = len(matched_pairs) - n_val

val_pairs   = matched_pairs[:n_val]
train_pairs = matched_pairs[n_val:]

print(f"\nSplit (seed={RANDOM_SEED}, val_fraction={VAL_FRACTION})")
print(f"  Train : {n_train}")
print(f"  Val   : {n_val}")

# ── Step 4: copy images + move val labels ─────────────────────────────────────
def copy_image(img_src: Path, img_dst_dir: Path) -> None:
    dst = img_dst_dir / img_src.name
    if not dst.exists():
        shutil.copy2(img_src, dst)

def move_label(lbl_src: Path, lbl_dst_dir: Path) -> None:
    dst = lbl_dst_dir / lbl_src.name
    if not dst.exists():
        shutil.move(str(lbl_src), dst)

print("\nCopying train images ...")
for lbl, img in train_pairs:
    copy_image(img, DST_IMG_TRAIN)
# Train labels already live in SRC_LABELS (labels/train/) — no move needed.

print("Copying val images and moving val labels ...")
for lbl, img in val_pairs:
    copy_image(img, DST_IMG_VAL)
    move_label(lbl, DST_LBL_VAL)

# ── Step 5: summary ───────────────────────────────────────────────────────────
print("\n--- Summary ---")
print(f"images/train/   : {len(list(DST_IMG_TRAIN.iterdir()))} files")
print(f"images/val/     : {len(list(DST_IMG_VAL.iterdir()))} files")
print(f"labels/train/   : {len(list(SRC_LABELS.glob('*.txt')))} .txt files")
print(f"labels/val/     : {len(list(DST_LBL_VAL.glob('*.txt')))} .txt files")
print("\nDone. You can now run: python train_manhole_detector.py")
