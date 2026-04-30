# Stage 1: Build frontend
FROM node:22-slim AS frontend
WORKDIR /app/web
ARG VITE_APP_API_KEY=
ENV VITE_APP_API_KEY=${VITE_APP_API_KEY}
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UPLOAD_DIR=/app/data/uploads \
    OUTPUT_DIR=/app/data/output \
    OLLAMA_MODELS=/app/data/ollama/models

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY pyproject.toml README.md ./
COPY lib/ ./lib/
COPY backend/ ./backend/
RUN pip install --upgrade pip \
    && pip install -e "."

# Frontend (built)
COPY --from=frontend /app/web/dist ./web/dist

# Config
COPY .env.example ./.env.example

# Data directories
RUN useradd --system --create-home --shell /usr/sbin/nologin remedy \
    && mkdir -p /app/data/uploads /app/data/output /app/data/ollama/models \
    && chown -R remedy:remedy /app

USER remedy

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
