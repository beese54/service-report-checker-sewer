# WRN Service Report Checker

Automated pipeline for sewer service report validation. Uses classical computer vision (SIFT, FLANN, RANSAC), a trained YOLOv8 manhole detector, and PaddleOCR to run staged quality checks on inspection images and generate pass/fail analytics.

---

## Architecture

Architecture diagrams are in [`docs/architecture/`](docs/architecture/).

---

## How It Works

| Stage | Images used | What it checks | Pass condition |
|---|---|---|---|
| **Level 1 — Classification** | All images in folder | Presence of before/after pairs | D1+D4 or U1+U4 present |
| **Stage 2N — SIFT Alignment** | D1/D4, U1/U4 | Same physical pipe location | ≥ 10 inliers AND ≥ 25% spatial coverage |
| **Stage 3N — Washing Confidence** | D2/D5, U2/U5 | Pipe was cleaned | `washing_confidence ≥ 0.58` (HIGH tier) |
| **Stage 4N — Geometry Analysis** | D3/D6, U3/U6 | Grease, water, texture, blur | Score ≥ 3/5 signals |
| **OCR (obstruction only)** | D1, U1, DR, UR, DL, UL | Blackboard text, MH number | Any OCR succeeds |

Obstruction folders (missing D1+D4 and U1+U4) are routed to OCR instead of the SIFT pipeline.

Full stage documentation: [`docs/pipeline-stages.md`](docs/pipeline-stages.md)

---

## Result Buckets

| Bucket | Meaning |
|---|---|
| `ACCEPTED` | All stages passed — pipe service confirmed |
| `NEEDS_REVIEW` | Stage 4N borderline score — human review needed |
| `REJECTED` | Failed at one or more stages |
| `OBSTRUCTION_PROCESSED` | Obstruction folder, OCR succeeded |

The results page shows a summary table with per-stage metrics for each outcome group.

---

## ZIP Format

```
my_service_report.zip
└── (optional root folder)
    ├── 1159097/
    │   ├── D1.jpg      ← downstream before
    │   ├── D2.jpg      ← downstream before (washing)
    │   ├── D3.jpg      ← downstream before (geometry)
    │   ├── D4.jpg      ← downstream after
    │   ├── D5.jpg      ← downstream after (washing)
    │   ├── D6.jpg      ← downstream after (geometry)
    │   ├── U1.jpg      ← upstream before
    │   └── ...
    ├── 1159099/
    └── ...
```

The tool handles ZIPs with or without a single root folder inside. Files can be `.jpg` or `.png`.

### Upload Limit

Maximum **200 MB** per upload. If your batch is larger, split it into two ZIPs and upload them separately. The limit is configurable via `MAX_UPLOAD_MB` (see Configuration).

---

## Quick Start

### Option A — Docker (recommended)

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
# Clone
git clone https://github.com/<your-org>/wrn-service-report-checker.git
cd wrn-service-report-checker

# Start
docker compose up --build

# Open browser
http://localhost:5000
```

Default password: `wrnchecker` (change via `APP_PASSWORD` env var).

### Option B — Local Python

**Prerequisites:** Python 3.11+

```bash
pip install paddleocr==2.7.3 --no-deps
pip install -r requirements-pipeline.txt

cd app
uvicorn main:app --reload --port 5000
```

---

## System Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB+ |
| CPU | 4 cores | 8 cores |
| Disk | 10 GB free | 20 GB+ |
| Docker | 24+ | latest |

### Image and Repository Sizes

| Artefact | Size |
|---|---|
| Git repository (code only) | ~3–5 MB |
| Docker image (after build) | ~3.5–4.5 GB (OpenCV + PaddleOCR + PaddlePaddle + YOLOv8) |
| PaddleOCR models (downloaded on first run) | ~500 MB |
| Manhole detector weights (user-supplied) | ~200–400 MB |

---

## Deployment

See [`docs/deployment-docker.md`](docs/deployment-docker.md) for self-hosted Docker deployment.

---

## Configuration

All values can be overridden by environment variables. Copy `.env.example` to `.env`.

| Variable | Default | Description |
|---|---|---|
| `APP_PASSWORD` | `wrnchecker` | Login password |
| `MAX_UPLOAD_MB` | `200` | Maximum ZIP upload size in MB |
| `STAGE2N_MIN_INLIERS` | `10` | RANSAC inlier threshold for Stage 2N |
| `STAGE2N_MIN_COVERAGE_PCT` | `25.0` | Spatial coverage threshold (%) for Stage 2N |
| `STAGE3N_HIGH_CONFIDENCE` | `0.58` | Washing confidence threshold (HIGH tier) |
| `STAGE3N_PASS_TIER` | `HIGH` | Minimum tier to pass Stage 3N |
| `STAGE4N_REVIEW_IS_ACCEPTED` | `false` | Treat NEEDS_REVIEW as ACCEPTED |
| `DETECTOR_CONF_ACCEPT` | `0.70` | YOLOv8 manhole detector auto-accept confidence |
| `DETECTOR_CONF_REVIEW` | `0.30` | YOLOv8 manhole detector review-queue confidence |
| `OUTPUT_DIR` | `./pipeline_output` | Where pipeline results are written |
| `PORT` | `5000` | Web server port |

---

## Project Structure

```
├── app/
│   ├── checker.py              # Level 1 classification logic
│   ├── contact_sheets.py       # Contact sheet image generator
│   ├── main.py                 # FastAPI web server (all endpoints)
│   └── templates/
│       ├── index.html          # Upload UI
│       ├── pipeline_results.html  # Results dashboard + contact sheets
│       ├── login.html          # Password login page
│       └── verify.html         # Human verification UI
├── pipeline/                   # Pipeline stage modules
│   ├── sort_folders.py         # Level 1 classification
│   ├── stage1_ocr.py           # OCR (obstruction folders)
│   ├── stage2n_sift.py         # SIFT alignment (D1/D4 + U1/U4)
│   ├── stage3n_washing.py      # Washing confidence (D2/D5 + U2/U5)
│   └── stage4n_geometry.py     # Geometry analysis (D3/D6 + U3/U6)
├── analytics/                  # Standalone analytics scripts
├── verification/               # Human verification + threshold optimisation
├── docs/                       # Documentation
│   ├── pipeline-stages.md
│   ├── deployment-docker.md
│   └── architecture/           # Architecture diagrams
├── pipeline.py                 # Main pipeline orchestrator
├── pipeline_config.py          # Centralised thresholds + env-var loading
├── Dockerfile
├── docker-compose.yml
├── requirements.txt            # Web app only
├── requirements-pipeline.txt   # Full pipeline (all stages)
└── requirements-ocr.txt        # OCR standalone
```

> **Note:** `original_images/`, `adjusted_images/`, `dataset/`, `manhole_detector/`, and pipeline output directories are excluded from this repository via `.gitignore`.

---

## Model Weights

The YOLOv8 manhole detector weights are not included in this repository (binary files, ~200–400 MB). You must supply your own trained model:

1. Train using `training/train_manhole_detector.py` with your annotated dataset.
2. Place the resulting weights at: `manhole_detector/v1_positive_only3/weights/best.pt`

If no model is present, the manhole detection gate is skipped and all image pairs are processed through the SIFT pipeline directly.

---

## Human Verification UI

After running pipeline jobs, you can label results to optimise detection thresholds:

1. Open `http://localhost:5000/verify` in your browser.
2. Label at least 30 folders.
3. Run `python verification/optimise_thresholds.py` to sweep thresholds.
4. Review `verification/recommended_thresholds.json` and update `pipeline_config.py`.

---

## OCR Install Notes

PaddleOCR 2.7.3 has an overly strict `opencv-python` pin. Install with `--no-deps`:

```bash
pip install paddleocr==2.7.3 --no-deps
pip install -r requirements-ocr.txt   # includes numpy==1.26.4
```

On first run, PaddleOCR downloads its detection models (~500 MB). Ensure internet access.
