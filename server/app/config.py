"""Service settings, read from environment variables (12-factor style)."""
import os
from dataclasses import dataclass, field


def _list(name):
    return [v.strip() for v in os.getenv(name, "").split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    # folder with installed model packages: <MODELS_DIR>/<task_id>/package.json
    models_dir: str = os.getenv("MODELS_DIR", "/models")
    # optional S3 prefix holding <task>_package.zip files, synced at start-up, e.g. s3://my-bucket/cancer-models/
    models_s3_uri: str = os.getenv("MODELS_S3_URI", "")
    # optional shared access token; when set, requests need "Authorization: Bearer <token>"
    access_token: str = os.getenv("ACCESS_TOKEN", "")
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "15"))
    # built React app served at "/" (empty = API only, e.g. when the web app is hosted on S3 + CloudFront)
    web_dir: str = os.getenv("WEB_DIR", "")
    # extra origins allowed to call the API from a browser (only needed when the web app is hosted elsewhere)
    cors_origins: list = field(default_factory=lambda: _list("CORS_ORIGINS"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
