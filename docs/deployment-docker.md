# Deployment — Docker (Local / Self-hosted)

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine (Linux)
- At least 4 GB RAM allocated to Docker (6 GB recommended for full pipeline with OCR)

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/wrn-service-report-checker.git
cd wrn-service-report-checker

# 2. Build
docker compose up --build

# 3. Open browser
http://localhost:5000
```

---

## docker-compose.yml Volume Mounts

```yaml
volumes:
  - ./data/input:/data/input      # Input images (read-only if desired)
  - ./data/output:/data/output    # Pipeline outputs written here
```

Results are written to `./data/output/<job_id>/output/pipeline_run_<timestamp>.json`.

---

## Environment Variable Overrides

Copy `.env.example` to `.env` and edit:

```env
APP_PASSWORD=your_password_here
MAX_UPLOAD_MB=200
STAGE2N_MIN_INLIERS=10
STAGE3N_HIGH_CONFIDENCE=0.58
OUTPUT_DIR=/data/output
PORT=5000
```

Then run:
```bash
docker compose --env-file .env up --build
```

---

## Running Without docker-compose

```bash
docker build -t wrn-checker .

docker run -p 5000:5000 \
  -v ./data/input:/data/input \
  -v ./data/output:/data/output \
  -e APP_PASSWORD=wrnchecker \
  -e OUTPUT_DIR=/data/output \
  wrn-checker
```

---

## Memory Limit

The `docker-compose.yml` sets a 6 GB memory limit (`mem_limit: 6g`) to prevent the PaddleOCR model from consuming too much RAM on shared machines. Adjust in `docker-compose.yml` if needed.

---

## Health Check

```bash
curl http://localhost:5000/health
# {"status":"ok"}
```

---

## Updating

```bash
git pull
docker compose up --build
```
