"""
AI Photo Recognizer - Batch Processing Manager
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from PIL import Image

from backend.core.config import settings
from backend.core.database import SessionLocal
from backend.custom_metrics import MetricContext
from backend.models.analysis import Analysis
from backend.services.ml_engine import ml_engine
from backend.services.metrics_loader import metrics_engine

logger = logging.getLogger(__name__)

# 16-32 is optimal for 224x224 images on RTX3060 (I think)
DEFAULT_BATCH_SIZE = 16


@dataclass
class BatchJob:
    """Represents a batch processing job."""
    job_id: str
    owner_id: int
    status: str = "queued"
    total: int = 0
    processed: int = 0
    current_file: str | None = None
    errors: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def progress_percent(self) -> float:
        if self.total == 0:
            return 0.0
        return round(self.processed / self.total * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "total": self.total,
            "processed": self.processed,
            "progress_percent": self.progress_percent,
            "current_file": self.current_file,
            "errors": self.errors[-10:],  # Last 10 errors
            "results": self.results,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class LoadedImage:
    filepath: Path
    image: Image.Image
    file_hash: str


class BatchProcessor:
    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        self._jobs: dict[str, BatchJob] = {}
        self._lock = Lock()
        self._progress_callbacks: dict[str, list[Callable]] = {}
        self.batch_size = batch_size

    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _collect_files(self, folder_path: Path, recursive: bool) -> list[Path]:
        pattern = "**/*" if recursive else "*"
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff"}

        files = []
        for p in folder_path.glob(pattern):
            if p.is_file() and p.suffix.lower() in extensions:
                files.append(p)

        return sorted(files)

    def _load_single_image(self, filepath: Path) -> LoadedImage | None:
        try:
            image = Image.open(filepath).convert("RGB")
            file_hash = self.compute_file_hash(filepath)
            return LoadedImage(filepath=filepath, image=image, file_hash=file_hash)
        except Exception as e:
            logger.warning(f"Failed to load image {filepath}: {e}")
            return None

    def _load_images_parallel(
        self,
        filepaths: list[Path],
        max_workers: int = 4
    ) -> tuple[list[LoadedImage], dict[Path, str]]:
        loaded = []
        errors = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self._load_single_image, fp): fp
                for fp in filepaths
            }

            for future in future_to_path:
                filepath = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        loaded.append(result)
                    else:
                        errors[filepath] = "Failed to load image"
                except Exception as e:
                    errors[filepath] = str(e)

        return loaded, errors

    def _check_cache_batch(
        self,
        loaded_images: list[LoadedImage],
        owner_id: int,
        db
    ) -> tuple[list[LoadedImage], list[dict[str, Any]]]:
        to_process = []
        cached_results = []

        hashes = [img.file_hash for img in loaded_images]

        cached = db.query(Analysis).filter(
            Analysis.file_hash.in_(hashes),
            Analysis.owner_id == owner_id,
            Analysis.backbone_name == ml_engine.backbone_name
        ).all()

        cached_map = {c.file_hash: c for c in cached}

        for loaded_img in loaded_images:
            if loaded_img.file_hash in cached_map:
                cached_analysis = cached_map[loaded_img.file_hash]
                cached_results.append({
                    "id": cached_analysis.id,
                    "filename": loaded_img.filepath.name,
                    "file_path": str(loaded_img.filepath),
                    "is_ai": cached_analysis.is_ai,
                    "score": cached_analysis.score,
                    "confidence": cached_analysis.confidence,
                    "inference_time_ms": cached_analysis.inference_time_ms,
                    "model_type": cached_analysis.model_type,
                    "backbone_name": cached_analysis.backbone_name,
                    "threshold_used": cached_analysis.threshold_used,
                    "custom_metrics": cached_analysis.custom_metrics,
                    "has_heatmap": cached_analysis.has_heatmap,
                    "created_at": cached_analysis.created_at.isoformat(),
                    "from_cache": True,
                })
            else:
                to_process.append(loaded_img)

        return to_process, cached_results

    def _create_metric_context(self) -> MetricContext:
        return MetricContext(
            threshold=ml_engine.threshold,
            model_backbone=ml_engine.backbone_name,
        )

    def _process_batch(
        self,
        loaded_images: list[LoadedImage],
        owner_id: int,
        job: BatchJob
    ) -> list[dict[str, Any]]:
        if not loaded_images:
            return []

        results = []
        context = self._create_metric_context()

        with SessionLocal() as db:
            to_process, cached_results = self._check_cache_batch(loaded_images, owner_id, db)
            results.extend(cached_results)

            if not to_process:
                return results

            pil_images = [img.image for img in to_process]

            predictions = ml_engine.predict_batch(pil_images, return_errors=True)

            for loaded_img, prediction in zip(to_process, predictions, strict=True):
                if prediction is None or prediction.error:
                    error_msg = prediction.error if prediction else "Unknown error"
                    job.errors.append(f"{loaded_img.filepath.name}: {error_msg}")
                    continue

                metrics = metrics_engine.compute_all(
                    loaded_img.image,
                    prediction.score,
                    context=context,
                    parallel=False
                )

                analysis = Analysis(
                    file_hash=loaded_img.file_hash,
                    filename=loaded_img.filepath.name,
                    file_path=str(loaded_img.filepath),
                    is_ai=prediction.is_ai,
                    score=prediction.score,
                    confidence=prediction.confidence,
                    threshold_used=prediction.threshold_used,
                    inference_time_ms=prediction.inference_time_ms,
                    model_type=prediction.model_type,
                    backbone_name=prediction.backbone_name,
                    custom_metrics=metrics,
                    owner_id=owner_id,
                )

                db.add(analysis)

                results.append({
                    "id": None,
                    "filename": loaded_img.filepath.name,
                    "file_path": str(loaded_img.filepath),
                    "is_ai": analysis.is_ai,
                    "score": analysis.score,
                    "confidence": analysis.confidence,
                    "inference_time_ms": analysis.inference_time_ms,
                    "model_type": analysis.model_type,
                    "backbone_name": analysis.backbone_name,
                    "threshold_used": analysis.threshold_used,
                    "custom_metrics": analysis.custom_metrics,
                    "has_heatmap": False,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "from_cache": False,
                })

            db.commit()

        return results

    def register_progress_callback(
        self,
        job_id: str,
        callback: Callable[[dict[str, Any]], Any]
    ) -> None:
        with self._lock:
            if job_id not in self._progress_callbacks:
                self._progress_callbacks[job_id] = []
            self._progress_callbacks[job_id].append(callback)

    def unregister_progress_callback(
        self,
        job_id: str,
        callback: Callable[[dict[str, Any]], Any]
    ) -> None:
        with self._lock:
            if job_id in self._progress_callbacks:
                try:
                    self._progress_callbacks[job_id].remove(callback)
                except ValueError:
                    pass

    async def start_batch_job(
        self,
        folder_path: str,
        owner_id: int,
        recursive: bool = False
    ) -> str:
        path = Path(folder_path)

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {folder_path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

        files = self._collect_files(path, recursive)

        if not files:
            raise ValueError(f"No image files found in: {folder_path}")

        job_id = str(uuid.uuid4())
        job = BatchJob(
            job_id=job_id,
            owner_id=owner_id,
            status="queued",
            total=len(files),
            started_at=datetime.now(timezone.utc)
        )

        with self._lock:
            self._jobs[job_id] = job

        asyncio.create_task(self._run_batch_worker(job, files, owner_id))

        return job_id

    async def _run_batch_worker(
        self,
        job: BatchJob,
        files: list[Path],
        owner_id: int
    ):
        job.status = "processing"
        loop = asyncio.get_running_loop()

        try:
            for batch_start in range(0, len(files), self.batch_size):
                batch_files = files[batch_start:batch_start + self.batch_size]

                batch_num = batch_start // self.batch_size + 1
                total_batches = (len(files) + self.batch_size - 1) // self.batch_size
                job.current_file = f"Batch {batch_num}/{total_batches}"

                loaded_images, load_errors = await loop.run_in_executor(
                    None,
                    self._load_images_parallel,
                    batch_files,
                    min(settings.BATCH_MAX_WORKERS, len(batch_files))
                )

                for filepath, error in load_errors.items():
                    job.errors.append(f"{filepath.name}: {error}")
                    job.processed += 1

                if loaded_images:
                    batch_results = await loop.run_in_executor(
                        None,
                        self._process_batch,
                        loaded_images,
                        owner_id,
                        job
                    )

                    job.results.extend(batch_results)
                    job.processed += len(loaded_images)

                await self._notify_progress(job)

            job.status = "completed"

        except Exception as e:
            job.status = "failed"
            job.errors.append(f"Job failed: {str(e)}")
            logger.exception(f"Batch job {job.job_id} failed")

        finally:
            job.completed_at = datetime.now(timezone.utc)
            job.current_file = None
            await self._notify_progress(job)

    async def _notify_progress(self, job: BatchJob):
        callbacks = self._progress_callbacks.get(job.job_id, [])
        for callback in callbacks:
            try:
                result = callback(job.to_dict())
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        cleaned = 0

        with self._lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.completed_at and job.completed_at < cutoff:
                    to_remove.append(job_id)

            for job_id in to_remove:
                del self._jobs[job_id]
                self._progress_callbacks.pop(job_id, None)
                cleaned += 1

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old batch jobs")

        return cleaned

    def get_active_jobs(self, owner_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [
                job.to_dict()
                for job in self._jobs.values()
                if job.owner_id == owner_id and job.status in ("queued", "processing")
            ]

batch_processor = BatchProcessor(batch_size=settings.BATCH_INFERENCE_SIZE)