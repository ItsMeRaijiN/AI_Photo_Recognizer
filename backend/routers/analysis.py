from __future__ import annotations

import hashlib
import io
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from backend.core.config import settings
from backend.core.database import get_db
from backend.core.uploads import read_upload_limited
from backend.custom_metrics import MetricContext
from backend.models.analysis import Analysis
from backend.models.user import User
from backend.routers.deps import get_current_user, get_current_user_or_none
from backend.schemas.analysis import AnalysisResponse, BatchUploadResponse
from backend.services.metrics_loader import metrics_engine
from backend.services.ml_engine import ml_engine

router = APIRouter(prefix="/analysis", tags=["Analysis"])
logger = logging.getLogger(__name__)

MAX_BATCH_FILES = 100
METRICS_VERSION_KEY = "__metrics_version"


@dataclass(frozen=True)
class BatchItem:
    source_index: int
    filename: str
    file_hash: str
    file_bytes: bytes


def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_image_bytes(data: bytes, filename: str) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise HTTPException(400, f"Invalid image dimensions for '{filename}'")
            if max(width, height) > settings.MAX_IMAGE_DIMENSION:
                raise HTTPException(
                    413,
                    f"Image '{filename}' exceeds the maximum dimension of "
                    f"{settings.MAX_IMAGE_DIMENSION}px",
                )
            if width * height > settings.MAX_IMAGE_PIXELS:
                raise HTTPException(
                    413,
                    f"Image '{filename}' exceeds the "
                    f"{settings.MAX_IMAGE_PIXELS:,} pixel limit",
                )
            image.verify()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise HTTPException(413, f"Image '{filename}' is too large: {e}")
    except Exception as e:
        raise HTTPException(400, f"Invalid image '{filename}': {e}")


def load_image_from_bytes(data: bytes, filename: str) -> Image.Image:
    validate_image_bytes(data, filename)
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image '{filename}': {e}")


def create_metric_context() -> MetricContext:
    return MetricContext(
        threshold=ml_engine.threshold,
        model_backbone=ml_engine.backbone_name,
    )


def build_guest_response(
    filename: str,
    result: Any,
    metrics: dict[str, Any] | None,
) -> AnalysisResponse:
    return AnalysisResponse(
        id=-1,
        filename=filename,
        file_path="memory",
        is_ai=result.is_ai,
        score=result.score,
        confidence=result.confidence,
        threshold_used=result.threshold_used,
        inference_time_ms=result.inference_time_ms,
        model_type=result.model_type,
        backbone_name=result.backbone_name,
        model_version=ml_engine.model_version,
        custom_metrics=metrics,
        heatmap_path=None,
        created_at=datetime.now(timezone.utc),
    )


def save_analysis_to_db(
    db: Session,
    filename: str,
    file_hash: str,
    file_path: str,
    result: Any,
    metrics: dict[str, Any] | None,
    owner_id: int,
) -> Analysis:
    analysis = Analysis(
        file_hash=file_hash,
        filename=filename,
        file_path=file_path,
        is_ai=result.is_ai,
        score=result.score,
        confidence=result.confidence,
        threshold_used=result.threshold_used,
        inference_time_ms=result.inference_time_ms,
        model_type=result.model_type,
        backbone_name=result.backbone_name,
        model_version=ml_engine.model_version,
        custom_metrics={
            METRICS_VERSION_KEY: metrics_engine.version,
            **(metrics or {}),
        },
        owner_id=owner_id,
    )
    db.add(analysis)
    return analysis


def get_cached_analyses(
    db: Session,
    hashes: list[str],
    owner_id: int,
) -> dict[str, Analysis]:
    cached = db.query(Analysis).filter(
        Analysis.file_hash.in_(hashes),
        Analysis.owner_id == owner_id,
        Analysis.model_version == ml_engine.model_version,
    ).all()
    return {
        str(c.file_hash): c
        for c in cached
        if (c.custom_metrics or {}).get(METRICS_VERSION_KEY) == metrics_engine.version
    }


def analysis_to_response(
    analysis: Analysis,
    *,
    source_index: int | None = None,
    filename: str | None = None,
) -> AnalysisResponse:
    response = AnalysisResponse.model_validate(analysis)
    public_metrics = dict(response.custom_metrics or {})
    public_metrics.pop(METRICS_VERSION_KEY, None)
    return response.model_copy(update={
        "filename": filename or response.filename,
        "file_path": "stored",
        "heatmap_path": "available" if response.heatmap_path else None,
        "custom_metrics": public_metrics or None,
        "source_index": source_index,
    })


def ensure_model_loaded() -> None:
    if not ml_engine.is_loaded or not ml_engine.model_version:
        raise HTTPException(503, "Analysis model is not available")


def resolve_managed_file(path: str | None, *managed_dirs: Path | None) -> Path | None:
    if not path:
        return None

    target = Path(path).resolve()
    for managed in managed_dirs:
        if managed and target.is_relative_to(managed.resolve()):
            return target
    return None


def try_delete_managed_file(path: str | None) -> None:
    if not path:
        return

    target = resolve_managed_file(path, settings.UPLOAD_DIR, settings.HEATMAPS_DIR)
    if target:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/predict", response_model=AnalysisResponse)
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_or_none),
) -> AnalysisResponse:
    ensure_model_loaded()
    filename = file.filename or "unknown"
    safe_filename = filename.replace("\\", "/").split("/")[-1]

    if not settings.validate_file_extension(safe_filename):
        raise HTTPException(400, f"Unsupported format: {safe_filename}")

    file_bytes = await read_upload_limited(
        file,
        max_bytes=settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024,
        label=f"File '{safe_filename}'",
    )
    file_hash = compute_file_hash(file_bytes)

    if current_user:
        cached = get_cached_analyses(db, [file_hash], current_user.id).get(file_hash)
        if cached:
            return analysis_to_response(cached)

    image = load_image_from_bytes(file_bytes, safe_filename)
    created_path: Path | None = None

    try:
        result = await run_in_threadpool(ml_engine.predict, image)
        if result.error:
            raise HTTPException(500, "Prediction could not be completed")

        context = create_metric_context()
        metrics = await run_in_threadpool(
            metrics_engine.compute_all,
            image,
            result.score,
            context,
            True,
        )

        if current_user:
            temp_filename = f"{uuid.uuid4()}_{safe_filename}"
            created_path = settings.UPLOAD_DIR / temp_filename
            await run_in_threadpool(created_path.write_bytes, file_bytes)

            analysis = save_analysis_to_db(
                db, safe_filename, file_hash, str(created_path),
                result, metrics, current_user.id
            )
            try:
                db.commit()
                db.refresh(analysis)
            except Exception:
                db.rollback()
                try_delete_managed_file(str(created_path))
                raise
            return analysis_to_response(analysis)

        return build_guest_response(safe_filename, result, metrics)
    except HTTPException:
        raise
    except Exception:
        if created_path:
            try_delete_managed_file(str(created_path))
        raise
    finally:
        image.close()


@router.post("/predict/batch", response_model=BatchUploadResponse)
async def predict_batch_upload(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_or_none),
) -> BatchUploadResponse:
    ensure_model_loaded()
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(400, f"Maximum {MAX_BATCH_FILES} files")
    if not files:
        raise HTTPException(400, "No files provided")

    start_time = time.perf_counter()
    context = create_metric_context()
    max_file_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    max_batch_bytes = settings.MAX_BATCH_TOTAL_SIZE_MB * 1024 * 1024

    items: list[BatchItem] = []
    errors: list[str] = []
    total_bytes = 0

    for source_index, file in enumerate(files):
        filename = (file.filename or "unknown").replace("\\", "/").split("/")[-1]

        if not settings.validate_file_extension(filename):
            errors.append(f"{filename}: Unsupported format")
            continue

        try:
            file_bytes = await read_upload_limited(
                file,
                max_bytes=max_file_bytes,
                label=f"File '{filename}'",
            )
            total_bytes += len(file_bytes)
            if total_bytes > max_batch_bytes:
                raise HTTPException(
                    413,
                    f"Batch exceeds the {settings.MAX_BATCH_TOTAL_SIZE_MB} MB limit",
                )
            validate_image_bytes(file_bytes, filename)
            file_hash = compute_file_hash(file_bytes)
            items.append(BatchItem(source_index, filename, file_hash, file_bytes))
        except HTTPException as e:
            if e.status_code == 413 and "Batch exceeds" in str(e.detail):
                raise
            errors.append(f"{filename}: {e.detail}")
        except Exception as e:
            errors.append(f"{filename}: {e}")

    if not items:
        raise HTTPException(400, f"No valid images. Errors: {errors}")

    cached_map: dict[str, Analysis] = {}
    if current_user:
        hashes = [item.file_hash for item in items]
        cached_map = get_cached_analyses(db, hashes, current_user.id)

    result_slots: dict[int, AnalysisResponse] = {}
    to_process: list[BatchItem] = []
    for item in items:
        if item.file_hash in cached_map:
            result_slots[item.source_index] = analysis_to_response(
                cached_map[item.file_hash],
                source_index=item.source_index,
                filename=item.filename,
            )
        else:
            to_process.append(item)

    saved_paths: list[Path] = []
    if to_process:
        chunk_size = max(1, min(settings.BATCH_INFERENCE_SIZE, 8))
        try:
            for start in range(0, len(to_process), chunk_size):
                chunk = to_process[start:start + chunk_size]
                decoded: list[tuple[BatchItem, Image.Image]] = []

                try:
                    for item in chunk:
                        try:
                            decoded.append((
                                item,
                                load_image_from_bytes(item.file_bytes, item.filename),
                            ))
                        except HTTPException as e:
                            errors.append(f"{item.filename}: {e.detail}")

                    if not decoded:
                        continue

                    predictions = await run_in_threadpool(
                        ml_engine.predict_batch,
                        [image for _, image in decoded],
                        True,
                    )

                    for (item, image), prediction in zip(decoded, predictions, strict=True):
                        if prediction is None or prediction.error:
                            errors.append(f"{item.filename}: prediction failed")
                            continue

                        metrics = await run_in_threadpool(
                            metrics_engine.compute_all,
                            image,
                            prediction.score,
                            context,
                            False,
                        )

                        if current_user:
                            file_path = settings.UPLOAD_DIR / f"{uuid.uuid4()}_{item.filename}"
                            saved_paths.append(file_path)
                            await run_in_threadpool(file_path.write_bytes, item.file_bytes)

                            analysis = save_analysis_to_db(
                                db, item.filename, item.file_hash, str(file_path),
                                prediction, metrics, current_user.id
                            )
                            db.flush()
                            result_slots[item.source_index] = analysis_to_response(
                                analysis,
                                source_index=item.source_index,
                                filename=item.filename,
                            )
                        else:
                            result_slots[item.source_index] = build_guest_response(
                                item.filename, prediction, metrics
                            ).model_copy(update={"source_index": item.source_index})
                finally:
                    for _, image in decoded:
                        image.close()

            if current_user:
                db.commit()
        except Exception:
            db.rollback()
            for path in saved_paths:
                try_delete_managed_file(str(path))
            raise

    results = [
        result_slots[index]
        for index in sorted(result_slots)
    ]

    total_time = (time.perf_counter() - start_time) * 1000

    return BatchUploadResponse(
        total=len(files),
        processed=len(results),
        failed=len(files) - len(results),
        results=results,
        errors=errors[:20],
        total_inference_time_ms=round(total_time, 2),
    )


@router.get("/model/info")
def get_model_info() -> dict[str, Any]:
    return {
        "type": ml_engine.model_type,
        "backbone": ml_engine.backbone_name,
        "threshold": ml_engine.threshold,
        "version": ml_engine.model_version,
        "image_size": ml_engine.image_size,
        "device": str(ml_engine.device),
        "loaded": ml_engine.is_loaded,
    }


@router.get("/metrics/available")
def get_available_metrics() -> dict[str, Any]:
    return metrics_engine.get_available_metrics()


@router.get("/history", response_model=list[AnalysisResponse])
def get_history(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[AnalysisResponse]:
    analyses = db.query(Analysis).filter(
        Analysis.owner_id == current_user.id
    ).order_by(Analysis.created_at.desc()).offset(skip).limit(limit).all()

    return [analysis_to_response(a) for a in analyses]


@router.get("/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisResponse:
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id,
        Analysis.owner_id == current_user.id,
    ).first()

    if not analysis:
        raise HTTPException(404, "Analysis not found")

    return analysis_to_response(analysis)


@router.delete("/{analysis_id}")
def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id,
        Analysis.owner_id == current_user.id,
    ).first()

    if not analysis:
        raise HTTPException(404, "Analysis not found")

    file_path = str(analysis.file_path) if analysis.file_path else None
    heatmap_path = str(analysis.heatmap_path) if analysis.heatmap_path else None
    db.delete(analysis)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    try_delete_managed_file(file_path)
    try_delete_managed_file(heatmap_path)

    return {"status": "deleted"}


@router.get("/{analysis_id}/image")
def get_analysis_image(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id,
        Analysis.owner_id == current_user.id,
    ).first()

    if not analysis:
        raise HTTPException(404, "Analysis not found")

    file_path = resolve_managed_file(str(analysis.file_path), settings.UPLOAD_DIR)
    if not file_path or not file_path.exists():
        raise HTTPException(404, "Image file not found")

    return FileResponse(file_path)


@router.get("/{analysis_id}/heatmap")
def get_heatmap(
    analysis_id: int,
    download: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id,
        Analysis.owner_id == current_user.id,
    ).first()

    if not analysis:
        raise HTTPException(404, "Analysis not found")

    ensure_model_loaded()
    if not analysis.model_version or analysis.model_version != ml_engine.model_version:
        raise HTTPException(
            409,
            "This analysis was created with a different model version. "
            "Run the analysis again to generate a matching heatmap.",
        )

    if analysis.heatmap_path:
        heatmap_file = resolve_managed_file(
            str(analysis.heatmap_path), settings.HEATMAPS_DIR
        )
        if heatmap_file and heatmap_file.exists():
            return FileResponse(
                heatmap_file,
                media_type="image/jpeg",
                filename=f"heatmap_{analysis.filename}" if download else None,
            )

    file_path = resolve_managed_file(str(analysis.file_path), settings.UPLOAD_DIR)
    if not file_path or not file_path.exists():
        raise HTTPException(404, "Original image not found")

    try:
        image = load_image_from_bytes(file_path.read_bytes(), analysis.filename)
        try:
            heatmap_image = ml_engine.generate_heatmap(image)
        finally:
            image.close()

        if heatmap_image is None:
            raise HTTPException(400, "Heatmap not available for this model")

        heatmap_path = settings.HEATMAPS_DIR / f"heatmap_{analysis_id}.jpg"
        try:
            heatmap_image.save(heatmap_path, "JPEG", quality=90)
        finally:
            heatmap_image.close()
        analysis.heatmap_path = str(heatmap_path)
        try:
            db.commit()
        except Exception:
            db.rollback()
            try_delete_managed_file(str(heatmap_path))
            raise

        return FileResponse(
            heatmap_path,
            media_type="image/jpeg",
            filename=f"heatmap_{analysis.filename}" if download else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Heatmap generation failed for analysis %s", analysis_id)
        raise HTTPException(500, "Heatmap generation failed") from e
