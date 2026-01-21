import logging
from pathlib import Path
from typing import Any
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Detect best available device"""
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
    """Auto-discover the best model file in runs/ directory."""
    runs_dir = base_dir / "runs"

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
            model_path = run_folder / "best_model.pt"
            if model_path.exists():
                return model_path

        direct_model = experiment_dir / "best_model.pt"
        if direct_model.exists():
            return direct_model

    pt_files = list(runs_dir.rglob("*.pt"))
    if pt_files:
        best_models = [f for f in pt_files if "best_model" in f.name]
        if best_models:
            return sorted(best_models, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        return sorted(pt_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]

    return None


def _is_placeholder(value: str) -> bool:
    """Check if a config value looks like a placeholder."""
    v = (value or "").strip().upper()
    return v.startswith("CHANGE-THIS") or v in {"", "DEFAULT", "SECRET", "PASSWORD"}


class Settings(BaseSettings):
    """Application settings with validation."""

    BASE_DIR: Path = Path(__file__).resolve().parents[2]

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

    DEVICE: str = ""  # Will be auto-detected if DETECT_DEVICE is True

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

    ALLOWED_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".bmp", ".tiff", ".gif"}

    BATCH_MAX_WORKERS: int = 4
    BATCH_INFERENCE_SIZE: int = 16
    BATCH_JOB_TTL_HOURS: int = 24

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize paths, device, and security checks after loading."""

        if not (self.BASE_DIR / "backend").exists():
            logger.warning("BASE_DIR might be incorrect. Could not find 'backend' folder at %s", self.BASE_DIR)

        if _is_placeholder(self.SECRET_KEY):
            logger.warning("INSECURE: SECRET_KEY looks like a placeholder.")

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