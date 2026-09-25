# TactiVision — Render / Docker web dashboard image
# Dashboard + cached analytics only. Full CV/upload needs local GPU + model weights.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TACTIVISION_UPLOADS_ENABLED=0 \
    PORT=8000

WORKDIR /app

# OpenCV headless runtime libs (no GUI / no CUDA)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY web ./web
COPY configs ./configs
COPY scripts ./scripts
COPY deploy ./deploy
COPY data/raw/.gitkeep data/raw/.gitkeep
COPY data/processed/.gitkeep data/processed/.gitkeep
COPY data/interim/.gitkeep data/interim/.gitkeep
COPY data/annotations/.gitkeep data/annotations/.gitkeep

# Prefer headless OpenCV in containers; pyproject pulls opencv-python.
RUN pip install --upgrade pip \
    && pip install -e ".[web]" \
    && pip uninstall -y opencv-python opencv-contrib-python || true \
    && pip install "opencv-python-headless>=4.10"

RUN chmod +x scripts/docker_entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

CMD ["./scripts/docker_entrypoint.sh"]
