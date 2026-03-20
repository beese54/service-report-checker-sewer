# WRN Service Report Checker — Full Pipeline Container
# Estimated image size: 3-4 GB (OpenCV, PaddleOCR, scikit-image)
#
# Build:  docker build -t wrn-checker .
# Run:    docker run -p 5000:5000 -v ./data/input:/data/input -v ./data/output:/data/output wrn-checker

FROM python:3.11-slim

# ── System dependencies (OpenCV, PaddleOCR) ───────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgcc-s1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser

# ── Python environment ────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first to leverage Docker layer cache
COPY requirements-pipeline.txt .

# Install PaddleOCR 2.7.3 without its conflicting OpenCV pin, then the rest
RUN pip install --no-cache-dir paddleocr==2.7.3 --no-deps && \
    pip install --no-cache-dir -r requirements-pipeline.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Ensure pipeline output dir exists and is owned by appuser
RUN mkdir -p /data/input /data/output && \
    chown -R appuser:appuser /data /app

USER appuser

# ── Environment variables (override in docker-compose or kubectl) ─────────────
ENV OUTPUT_DIR=/data/output
ENV PORT=5000

# Pipeline thresholds (all have defaults in pipeline_config.py)
# ENV STAGE2N_MIN_INLIERS=10
# ENV STAGE2N_MIN_PCT_CHG=5.0
# ENV STAGE3N_PASS_TIER=HIGH
# ENV STAGE4N_REVIEW_IS_ACCEPTED=false

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-5000}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-5000} --workers 2"]
