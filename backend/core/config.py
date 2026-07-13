import logging
import secrets
from pathlib import Path
from typing import Any
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _find_best_model(base_dir: Path) -> Path | None:
    runs_dir = base_dir / "runs"
    canonical_names = ("best_model.pt", "model.onnx")

    if not runs_dir.exists():
        return None

    experiment_dir = runs_dir / "experiment"
    if experiment_dir.exists():
        run_folders = sorted(
            [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda x: x.name,
            reverse=True
        )
        for run_folder in run_folders:
            for model_name in canonical_names:
                model_path = run_folder / model_name
                if model_path.exists():
                    return model_path

        for model_name in canonical_names:
            direct_model = experiment_dir / model_name
            if direct_model.exists():
                return direct_model

    model_files = [*runs_dir.rglob("*.pt"), *runs_dir.rglob("*.onnx")]
    if model_files:
        canonical_models = [f for f in model_files if f.name in canonical_names]
        if canonical_models:
            return max(canonical_models, key=lambda x: x.stat().st_mtime)
        return max(model_files, key=lambda x: x.stat().st_mtime)

    return None


def _is_placeholder(value: str) -> bool:
    v = (value or "").strip().upper()
    return (
        v.startswith(("CHANGE-THIS", "CHANGE-ME", "CHANGEME"))
        or v in {"", "DEFAULT", "SECRET", "PASSWORD"}
    )


class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parents[2]

    ENVIRONMENT: str = "development"

    CREATE_DIRS: bool = True
    DETECT_DEVICE: bool = True
    AUTO_DISCOVER_MODEL: bool = True

    UPLOAD_DIR: Path | None = None
    TEMP_DIR: Path | None = None
    METRICS_DIR: Path | None = None
    HEATMAPS_DIR: Path | None = None

    MODEL_PATH: Path | None = None

    DATABASE_URL: str = "sqlite:///aipr_storage.db"
    AUTO_CREATE_TABLES: bool = True

    DEVICE: str = ""

    SECRET_KEY: str = "CHANGE-THIS-IN-PRODUCTION-USE-SECRETS"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24h
    ADMIN_BOOTSTRAP_TOKEN: str = "CHANGE-THIS-BOOTSTRAP-TOKEN"

    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    ALLOWED_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".bmp", ".tiff"}

    MAX_UPLOAD_SIZE_MB: int = 20
    MAX_BATCH_TOTAL_SIZE_MB: int = 200
    MAX_MODEL_UPLOAD_SIZE_MB: int = 1024
    MAX_IMAGE_PIXELS: int = 25_000_000
    MAX_IMAGE_DIMENSION: int = 12_000
    UPLOAD_CHUNK_SIZE_KB: int = 1024
    BATCH_INFERENCE_SIZE: int = 16

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def model_post_init(self, __context: Any) -> None:
        if not (self.BASE_DIR / "backend").exists():
            logger.warning("BASE_DIR might be incorrect. Could not find 'backend' folder at %s", self.BASE_DIR)

        if _is_placeholder(self.SECRET_KEY):
            if self.ENVIRONMENT.strip().lower() in {"production", "prod"}:
                raise ValueError(
                    "SECRET_KEY must be configured before starting in production."
                )
            self.SECRET_KEY = secrets.token_urlsafe(48)
            logger.warning(
                "SECRET_KEY was not configured; generated an ephemeral development key. "
                "Sessions will be invalidated after restart."
            )

        if _is_placeholder(self.ADMIN_BOOTSTRAP_TOKEN):
            logger.warning("INSECURE: ADMIN_BOOTSTRAP_TOKEN looks like a placeholder.")

        if self.DETECT_DEVICE and not self.DEVICE:
            self.DEVICE = _detect_device()

        if self.UPLOAD_DIR is None:
            self.UPLOAD_DIR = self.BASE_DIR / "uploads"
        elif not self.UPLOAD_DIR.is_absolute():
            self.UPLOAD_DIR = self.BASE_DIR / self.UPLOAD_DIR

        if self.TEMP_DIR is None:
            self.TEMP_DIR = self.BASE_DIR / "temp"
        elif not self.TEMP_DIR.is_absolute():
            self.TEMP_DIR = self.BASE_DIR / self.TEMP_DIR

        if self.METRICS_DIR is None:
            self.METRICS_DIR = self.BASE_DIR / "backend" / "custom_metrics"
        elif not self.METRICS_DIR.is_absolute():
            self.METRICS_DIR = self.BASE_DIR / self.METRICS_DIR

        if self.HEATMAPS_DIR is None:
            self.HEATMAPS_DIR = self.BASE_DIR / "heatmaps"
        elif not self.HEATMAPS_DIR.is_absolute():
            self.HEATMAPS_DIR = self.BASE_DIR / self.HEATMAPS_DIR

        if self.AUTO_DISCOVER_MODEL and self.MODEL_PATH is None:
            discovered = _find_best_model(self.BASE_DIR)
            if discovered:
                self.MODEL_PATH = discovered
                logger.info("Auto-discovered model: %s", discovered)
            else:
                self.MODEL_PATH = self.BASE_DIR / "runs" / "experiment" / "best_model.pt"
                logger.warning("No model found, defaulting to: %s", self.MODEL_PATH)
        elif self.MODEL_PATH and not self.MODEL_PATH.is_absolute():
            self.MODEL_PATH = self.BASE_DIR / self.MODEL_PATH

        if self.MODEL_PATH is None and not self.AUTO_DISCOVER_MODEL:
            logger.warning("MODEL_PATH is not set and auto-discovery is disabled.")

        if self.DATABASE_URL.startswith("sqlite:///"):
            rel_path = self.DATABASE_URL.removeprefix("sqlite:///")
            p = Path(rel_path)
            if not p.is_absolute():
                resolved_path = (self.BASE_DIR / p).resolve()
                self.DATABASE_URL = f"sqlite:///{resolved_path.as_posix()}"

        if self.CREATE_DIRS:
            for dir_path in [self.UPLOAD_DIR, self.TEMP_DIR, self.METRICS_DIR, self.HEATMAPS_DIR]:
                if dir_path:
                    dir_path.mkdir(parents=True, exist_ok=True)

    def validate_file_extension(self, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in self.ALLOWED_EXTENSIONS


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
