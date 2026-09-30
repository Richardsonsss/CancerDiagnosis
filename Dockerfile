# Diagnosis service: React web app (PWA) + FastAPI inference API in one image.
#   docker build -t cancer-diagnosis .

# ---- 1. build the web app ------------------------------------------------------------
FROM node:24-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY web/ ./
RUN npm run build

# ---- 2. runtime ------------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/models \
    WEB_DIR=/app/web
WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt
COPY common/ common/
COPY server/ server/
COPY --from=web /web/dist web/
RUN useradd --create-home --uid 1000 app && mkdir -p /models && chown app /models
USER app
WORKDIR /app/server
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
# one worker process (each would hold its own copy of the models); requests run in its thread pool
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
