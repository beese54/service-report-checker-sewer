# Pipeline Stages — Detailed Reference

## Overview

Every uploaded job folder goes through up to four stages depending on whether it is classified as **no obstruction** or **obstruction**.

```
Upload ZIP
    │
    ▼
Classify folders (Level 1)
    │
    ├─ obstruction ──────► OCR Stage
    │                          │
    │                          └─ OBSTRUCTION_PROCESSED or REJECTED
    │
    └─ no_obstruction ──► Stage 2N (SIFT alignment)
                              │
                              ├─ FAIL ──────────────────────────────► REJECTED (D1D4_U1U4)
                              │
                              └─ PASS ─► Stage 3N (Washing confidence)
                                             │
                                             ├─ LOW ───────────────► REJECTED (D2D5_U2U5)
                                             │
                                             ├─ MEDIUM ───► Stage 4N rescue
                                             │                   │
                                             │                   ├─ ACCEPTED or REJECTED
                                             │
                                             └─ HIGH ──► Stage 4N (Geometry)
                                                             │
                                                             ├─ ACCEPTED
                                                             ├─ NEEDS_REVIEW
                                                             └─ REJECTED (D3D6_U3U6)
```

---

## Level 1 — Classification

**File:** `pipeline/sort_folders.py` → `classify_only()`

| Condition | Result |
|---|---|
| Folder contains D1 **and** D4 (any extension) | `no_obstruction` |
| Folder contains U1 **and** U4 (any extension) | `no_obstruction` |
| Neither pair present | `obstruction` |

---

## Stage 2N — SIFT Alignment (D1/D4 + U1/U4)

**File:** `pipeline/stage2n_sift.py`
**Images:** D1 (before), D4 (after), U1 (before), U4 (after)
**Purpose:** Confirm the before/after images show the same physical pipe location.

### Algorithm
1. Optional YOLOv8 manhole detector gate (rejects non-pipe images before SIFT)
2. SIFT keypoint detection on both images
3. FLANN nearest-neighbour matching with Lowe's ratio test (`ratio=0.75`)
4. RANSAC homography (`threshold=5.0 px`) to find geometric inliers
5. Spatial coverage check: inlier keypoints must cover ≥ 25% of a 4×4 grid

### Pass Conditions (both required per pair)
| Metric | Threshold | Config variable |
|---|---|---|
| RANSAC inliers | ≥ 10 | `STAGE2N_MIN_INLIERS` |
| Spatial coverage | ≥ 25% | `STAGE2N_MIN_COVERAGE_PCT` |

A folder passes if **at least one pair** (D or U) meets both thresholds.

### Fallback
If Stage 2N fails, the pipeline retries with a more permissive ratio (`RATIO_THRESHOLD_FALLBACK=0.80`). If fallback also fails → `REJECTED (D1D4_U1U4)`.

---

## Stage 3N — Washing Confidence (D2/D5 + U2/U5)

**File:** `pipeline/stage3n_washing.py`
**Images:** D2 (before), D5 (after), U2 (before), U5 (after)
**Purpose:** Confirm the pipe was actually cleaned between recordings.

### Signals (6 features extracted from image pair)
| Signal | Interpretation |
|---|---|
| `kp_ratio` | Keypoint ratio after/before (more features = cleaner surface) |
| `std_increase_pct` | Std dev increase % (more texture variation after) |
| `entropy_increase` | Shannon entropy increase (more detail after) |
| `match_ratio` | FLANN match ratio (lower = more different = cleaned) |
| `edge_increase_pct` | Edge density increase % |
| `lap_increase_pct` | Laplacian variance increase % (sharpness) |

### Confidence Tiers
| Tier | Condition | Pipeline action |
|---|---|---|
| `HIGH` | `washing_confidence >= 0.58` | **PASS** → proceed to Stage 4N |
| `MEDIUM` | `0.30 ≤ washing_confidence < 0.58` | Attempt Stage 4N geometry rescue |
| `LOW` | `washing_confidence < 0.30` | **REJECTED (D2D5_U2U5)** |

---

## Stage 4N — Geometry Analysis (D3/D6 + U3/U6)

**File:** `pipeline/stage4n_geometry.py`
**Images:** D3 (before), D6 (after), U3 (before), U6 (after)
**Purpose:** Multi-signal geometry check to assess pipe condition.

### Signals (5 signals, score 0–5)
| Signal | Passes when |
|---|---|
| S1 — Blur gate | Image is not blurry (Laplacian variance ≥ threshold) |
| S2 — % changed | ≥ 5% of pixels changed between before/after |
| S3 — Texture | Texture confirmed present in pipe area |
| S4 — Water | Flowing water detected (residual moisture signal) |
| S5 — No grease | Grease fraction not flagged (grease_pct ≤ threshold) |

### Verdicts
| Score | Verdict |
|---|---|
| ≥ 3/5 | `ACCEPTED` |
| 2/5 | `NEEDS_REVIEW` |
| ≤ 1/5 | `REJECTED (D3D6_U3U6)` |

---

## OCR Stage — Obstruction Folders

**File:** `pipeline/stage1_ocr.py`
**Images:** D1, U1, DR, UR, DL, UL (blackboard images)
**Engine:** PaddleOCR 2.7.3

### What it extracts
- Manhole number (`MH-XXX` pattern)
- All detected text lines with confidence scores
- Annotated images with bounding boxes

### Output
| File | Location |
|---|---|
| Annotated images | `output/obstruction/extracted_text/<folder>/` |
| JSON results | `output/obstruction/extracted_text/results.json` |

### Verdict
- `OBSTRUCTION_PROCESSED` — at least one image successfully OCR'd
- `REJECTED (OCR_FAILED)` — no images processed

---

## Result Buckets

| Bucket | Meaning |
|---|---|
| `ACCEPTED` | All stages passed — pipe service confirmed |
| `NEEDS_REVIEW` | Stage 4N gave borderline score — human review needed |
| `REJECTED` | Failed at one or more stages — see `failed_stage` field |
| `OBSTRUCTION_PROCESSED` | Obstruction folder with successful OCR |

---

## Configuration

All thresholds can be overridden via environment variables. See `.env.example` for the full list.
