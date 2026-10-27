# The API image. Build from the repo root (the context needs src/, params.yaml and uv.lock):
#   docker build -f backend/Dockerfile -t fjd-api .
#   docker run -p 8000:8000 --env-file .env fjd-api
# The model isn't baked in: it is downloaded from the MLflow registry (MODEL_URI) at startup.
FROM python:3.12-slim

# LightGBM needs OpenMP at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8000
WORKDIR /app

# Dependencies first, so code changes don't reinstall them. Only the `backend` group: no dev
# tools or notebooks. XGBoost's NCCL is for multi-GPU training (~240 MB), not CPU inference.
ARG SYNC="uv sync --frozen --no-default-groups --group backend --no-install-package nvidia-nccl-cu13"
COPY pyproject.toml uv.lock README.md ./
RUN $SYNC --no-install-project
COPY params.yaml ./
COPY src ./src
COPY backend ./backend
RUN $SYNC

RUN useradd --create-home --uid 1000 api
USER api
EXPOSE 8000
# Migrate the database, then serve with one worker (Render's free tier has 512 MB of RAM).
# --proxy-headers: the client IP, for rate limiting, comes from Render's X-Forwarded-For.
CMD ["sh", "-c", "alembic -c backend/alembic.ini upgrade head && exec uvicorn backend.main:create_app --factory --host 0.0.0.0 --port $PORT --workers 1 --proxy-headers --forwarded-allow-ips '*'"]
