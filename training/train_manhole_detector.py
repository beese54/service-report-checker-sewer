"""
Manhole Cover Detector — Training Script
=========================================
Trains a YOLOv8 object detector to identify manhole covers in photos.

Usage:
  python train_manhole_detector.py

Phases:
  Phase 1 (positive-only):  annotate 150-250 manhole images with X-AnyLabeling,
                             place in dataset/images/train + dataset/labels/train,
                             then run this script.
  Phase 5 (with negatives): after monthly Label Studio review cycles have produced
                             verified non-manhole images, add them (with empty .txt
                             labels) to dataset/ and re-run this script.

Outputs:
  manhole_detector/<run_name>/weights/best.pt   ← use this in the pipeline
  manhole_detector/<run_name>/weights/last.pt
  manhole_detector/<run_name>/results.csv       ← training metrics per epoch
"""

import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
DATA_YAML   = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\dataset\data.yaml"
PROJECT_DIR = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\manhole_detector"

# Increment this label each time you retrain (e.g. v1_positive_only, v2_with_negatives)
RUN_NAME    = "v1_positive_only"

# Model size — "n" (nano) trains and runs on CPU; upgrade to "s" (small) if GPU available
MODEL_BASE  = "yolov8n.pt"

EPOCHS      = 100
IMAGE_SIZE  = 640
BATCH       = 8          # reduce to 4 if you get out-of-memory errors on CPU
PATIENCE    = 20         # early stopping: stop if val loss doesn't improve for 20 epochs


def check_dataset():
    """Verify the dataset folder has at least some images before training."""
    train_img_dir = os.path.join(os.path.dirname(DATA_YAML), "images", "train")
    if not os.path.isdir(train_img_dir):
        print(f"ERROR: Training image folder not found: {train_img_dir}")
        print("  Create the folder and add annotated images before training.")
        sys.exit(1)

    images = [f for f in os.listdir(train_img_dir)
              if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}]
    if len(images) < 10:
        print(f"WARNING: Only {len(images)} training image(s) found in {train_img_dir}.")
        print("  Annotate at least 150 images with X-AnyLabeling before training.")
        if len(images) == 0:
            sys.exit(1)
    else:
        print(f"  Training images found: {len(images)}")


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.")
        print("  Install it:  pip install ultralytics")
        sys.exit(1)

    print("── Manhole Detector Training ───────────────────────────────────────")
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Model   : {MODEL_BASE}")
    print(f"  Run     : {RUN_NAME}")
    print(f"  Epochs  : {EPOCHS}  |  Image size: {IMAGE_SIZE}  |  Batch: {BATCH}")
    print()

    check_dataset()

    model = YOLO(MODEL_BASE)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=False,     # set True to overwrite an existing run of the same name
        verbose=True,
    )

    best_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    print()
    print("── Training complete ───────────────────────────────────────────────")
    print(f"  Best weights : {best_weights}")
    print()
    print("  Next steps:")
    print("  1. Check val mAP@0.5 in the results above — target ≥ 0.85")
    print("  2. Update DETECTOR_MODEL in images_with_no_manhole_obstruction.py:")
    print(f'       DETECTOR_MODEL = r"{best_weights}"')
    print("  3. Run the pipeline — low-confidence images will be queued for review")
    print("  4. After 1-2 monthly review cycles, re-run this script as Phase 5")


if __name__ == "__main__":
    main()
