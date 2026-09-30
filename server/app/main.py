"""Cancer image diagnosis API (FastAPI).

    GET  /api/health     liveness / readiness (for the load balancer)
    GET  /api/tasks      cancer types with an installed model
    POST /api/diagnose   multipart form: task_id + image  ->  diagnosis text (no numbers)
    GET  /api/docs       OpenAPI documentation
    GET  /               the React web app (PWA), when WEB_DIR is set

Images are processed in memory only: never written to disk and never logged.
"""
import hmac
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "common"))
from .config import settings  # noqa: E402
from .diagnosis import load_engines  # noqa: E402
from .packages import install_zip, sync_from_s3  # noqa: E402

try:  # iPhone photos may arrive as HEIC
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
Image.MAX_IMAGE_PIXELS = 60_000_000  # refuse decompression bombs
MIN_SIDE = 64

engines = {}


@asynccontextmanager
async def lifespan(app):
    if settings.models_s3_uri:
        sync_from_s3(settings.models_s3_uri, settings.models_dir)
    # packages copied by hand (e.g. with WinSCP) into MODELS_DIR as *_package.zip are installed too
    for f in sorted(os.listdir(settings.models_dir)) if os.path.isdir(settings.models_dir) else []:
        if f.endswith("_package.zip"):
            path = os.path.join(settings.models_dir, f)
            log.info("installing %s", f)
            install_zip(path, settings.models_dir)
            os.remove(path)
    engines.update(load_engines(settings.models_dir))
    log.info("ready with %d model package(s): %s", len(engines), ", ".join(engines) or "-")
    yield


app = FastAPI(title="Cancer Image Diagnosis API", version="1.0.0", lifespan=lifespan,
              docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=500)  # compresses the web app's JS / CSS / HTML
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["Authorization", "Content-Type"])


# ---------------------------------------------------------------- schemas
class TaskInfo(BaseModel):
    id: str
    name: str
    modality: str
    input_hint: str


class DiagnosisResponse(BaseModel):
    level: str  # "high" | "low" | "uncertain"
    verdict: str
    advice: str


def require_token(authorization: str = Header(default="")):
    if settings.access_token:
        token = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token, settings.access_token):
            raise HTTPException(401, "Invalid access code")


# ---------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"status": "ok", "models": len(engines)}


@app.get("/api/tasks", response_model=list[TaskInfo], dependencies=[Depends(require_token)])
def tasks():
    return [e.info for e in engines.values()]


@app.post("/api/diagnose", response_model=DiagnosisResponse, dependencies=[Depends(require_token)])
def diagnose(task_id: str = Form(...), image: UploadFile = File(...)):
    # a sync endpoint: FastAPI runs it in its thread pool, so ONNX inference does not block the event loop
    engine = engines.get(task_id)
    if engine is None:
        raise HTTPException(404, "No model is installed for this cancer type")
    limit = int(settings.max_upload_mb * 1024 * 1024)
    data = image.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, "The image file is too large. Please retake or compress the photo.")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")  # honour the iPhone's orientation flag
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(415, "Unsupported image format. Please upload a JPEG, PNG or HEIC photo.")
    if min(img.size) < MIN_SIDE:
        raise HTTPException(422, "The image resolution is too low. Please retake the photo.")
    t0 = time.perf_counter()
    result = engine.diagnose(img)
    ms = (time.perf_counter() - t0) * 1000
    log.info("diagnose task=%s level=%s latency_ms=%.0f", task_id, result.level, ms)
    # timing for browser developer tools / monitoring only; the web app never displays it
    return JSONResponse(DiagnosisResponse(**result.__dict__).model_dump(),
                        headers={"Server-Timing": f"inference;dur={ms:.0f}", "Cache-Control": "no-store"})


# ---------------------------------------------------------------- LAN mode: local certificate authority
if os.getenv("LAN_CA_CERT"):
    @app.get("/lan/ca.crt", include_in_schema=False)
    def lan_ca():
        """The local CA certificate (public part only), for the phone to trust the LAN HTTPS certificate."""
        return FileResponse(os.environ["LAN_CA_CERT"], media_type="application/x-x509-ca-cert",
                            filename="CancerDiagnosisLocalCA.crt")


# ---------------------------------------------------------------- web app
if settings.web_dir and os.path.isdir(settings.web_dir):
    class ImmutableAssets(StaticFiles):
        """Vite puts a content hash in every asset file name, so browsers may cache them for a year."""
        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return response

    app.mount("/assets", ImmutableAssets(directory=os.path.join(settings.web_dir, "assets")), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def web(path: str):
        """Static files of the React build; unknown paths fall back to index.html (SPA)."""
        if path.startswith("api/"):
            raise HTTPException(404, "Not Found")
        root = os.path.realpath(settings.web_dir)
        full = os.path.realpath(os.path.join(root, path))
        if not (path and os.path.commonpath([root, full]) == root and os.path.isfile(full)):
            full = os.path.join(root, "index.html")
        # index.html, sw.js, manifest: always revalidate so a new release is picked up at once
        return FileResponse(full, headers={"Cache-Control": "no-cache"})
