from __future__ import annotations

import hashlib
import io
import time
import uuid
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
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.database import get_db
from backend.custom_metrics import MetricContext
from backend.models.analysis import Analysis
from backend.models.user import User
from backend.routers.deps import get_current_user, get_current_user_or_none
from backend.schemas.analysis import (
    AnalysisResponse,
    BatchRequest,
    BatchUploadResponse,
    FolderAnalysisRequest,
    JobStatusResponse,
)
from backend.services.batch_manager import batch_processor
from backend.services.metrics_loader import metrics_engine
from backend.services.ml_engine import ml_engine

router = APIRouter(prefix="/analysis", tags=["Analysis"])

# Constants
MAX_BATCH_FILES = 100
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic"}


def compute_file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def load_image_from_bytes(data: bytes, filename: str) -> Image.Image:
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
        custom_metrics=metrics or None,
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
        Analysis.backbone_name == ml_engine.backbone_name,
    ).all()
    return {str(c.file_hash): c for c in cached}


def try_delete_file(path: str | None) -> None:
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/predict", response_model=AnalysisResponse)
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_or_none),
) -> AnalysisResponse:
    filename = file.filename or "unknown"
    safe_filename = filename.replace("\\", "/").split("/")[-1]

    if not settings.validate_file_extension(safe_filename):
        raise HTTPException(400, f"Unsupported format: {safe_filename}")

    file_bytes = await file.read()
    file_hash = compute_file_hash(file_bytes)

    if current_user:
        cached = db.query(Analysis).filter(
            Analysis.file_hash == file_hash,
            Analysis.owner_id == current_user.id,
            Analysis.backbone_name == ml_engine.backbone_name,
        ).first()
        if cached:
            return AnalysisResponse.model_validate(cached)

    image = load_image_from_bytes(file_bytes, safe_filename)

    result = ml_engine.predict(image)
    if result.error:
        raise HTTPException(500, f"Prediction error: {result.error}")

    context = create_metric_context()
    metrics = metrics_engine.compute_all(image, result.score, context, parallel=True)

    if current_user:
        temp_filename = f"{uuid.uuid4()}_{safe_filename}"
        file_path = settings.UPLOAD_DIR / temp_filename
        file_path.write_bytes(file_bytes)

        analysis = save_analysis_to_db(
            db, safe_filename, file_hash, str(file_path),
            result, metrics, current_user.id
        )
        db.commit()
        db.refresh(analysis)
        return AnalysisResponse.model_validate(analysis)

    return build_guest_response(safe_filename, result, metrics)


@router.post("/predict/batch", response_model=BatchUploadResponse)
async def predict_batch_upload(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_or_none),
) -> BatchUploadResponse:
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(400, f"Maximum {MAX_BATCH_FILES} files")
    if not files:
        raise HTTPException(400, "No files provided")

    start_time = time.perf_counter()
    context = create_metric_context()

    images_data: list[tuple[str, str, Image.Image, bytes]] = []
    errors: list[str] = []

    for file in files:
        filename = (file.filename or "unknown").replace("\\", "/").split("/")[-1]

        if not settings.validate_file_extension(filename):
            errors.append(f"{filename}: Unsupported format")
            continue

        try:
            file_bytes = await file.read()
            file_hash = compute_file_hash(file_bytes)
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            images_data.append((filename, file_hash, image, file_bytes))
        except Exception as e:
            errors.append(f"{filename}: {e}")

    if not images_data:
        raise HTTPException(400, f"No valid images. Errors: {errors}")

    results: list[AnalysisResponse] = []

    cached_map: dict[str, Analysis] = {}
    if current_user:
        hashes = [d[1] for d in images_data]
        cached_map = get_cached_analyses(db, hashes, current_user.id)

    to_process: list[tuple[str, str, Image.Image, bytes]] = []
    for filename, file_hash, image, file_bytes in images_data:
        if file_hash in cached_map:
            results.append(AnalysisResponse.model_validate(cached_map[file_hash]))
        else:
            to_process.append((filename, file_hash, image, file_bytes))

    if to_process:
        images_list = [item[2] for item in to_process]
        predictions = ml_engine.predict_batch(images_list, return_errors=True)

        for (filename, file_hash, image, file_bytes), prediction in zip(
            to_process, predictions, strict=True
        ):
            if prediction is None or prediction.error:
                errors.append(f"{filename}: {prediction.error if prediction else 'Error'}")
                continue

            metrics = metrics_engine.compute_all(image, prediction.score, context, parallel=False)

            if current_user:
                temp_filename = f"{uuid.uuid4()}_{filename}"
                file_path = settings.UPLOAD_DIR / temp_filename
                file_path.write_bytes(file_bytes)

                analysis = save_analysis_to_db(
                    db, filename, file_hash, str(file_path),
                    prediction, metrics, current_user.id
                )
                db.flush()
                results.append(AnalysisResponse.model_validate(analysis))
            else:
                results.append(build_guest_response(filename, prediction, metrics))

        if current_user:
            db.commit()

    total_time = (time.perf_counter() - start_time) * 1000

    return BatchUploadResponse(
        total=len(files),
        processed=len(results),
        failed=len(errors),
        results=results,
        errors=errors[:20],
        total_inference_time_ms=round(total_time, 2),
    )


@router.post("/folder", response_model=BatchUploadResponse)
async def analyze_folder(
    req: FolderAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> BatchUploadResponse:
    target_path = Path(req.path)

    if not target_path.exists():
        raise HTTPException(404, f"Path not found: {req.path}")

    start_time = time.perf_counter()
    context = create_metric_context()

    files: list[Path] = []

    if target_path.is_file():
        if target_path.suffix.lower() in IMAGE_EXTENSIONS:
            files = [target_path]
        else:
            raise HTTPException(400, f"Unsupported format: {target_path.suffix}")
    else:
        pattern = "**/*" if req.recursive else "*"
        for p in target_path.glob(pattern):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(p)
                if len(files) >= req.max_images:
                    break

    if not files:
        raise HTTPException(400, f"No images found in: {req.path}")

    results: list[AnalysisResponse] = []
    errors: list[str] = []

    images_data: list[tuple[str, str, Image.Image, str]] = []
    for filepath in files:
        try:
            file_bytes = filepath.read_bytes()
            file_hash = compute_file_hash(file_bytes)
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            images_data.append((filepath.name, file_hash, image, str(filepath)))
        except Exception as e:
            errors.append(f"{filepath.name}: {e}")

    if not images_data:
        raise HTTPException(400, f"No valid images. Errors: {errors}")

    hashes = [d[1] for d in images_data]
    cached_map = get_cached_analyses(db, hashes, current_user.id)

    to_process: list[tuple[str, str, Image.Image, str]] = []
    for item in images_data:
        if item[1] in cached_map:
            results.append(AnalysisResponse.model_validate(cached_map[item[1]]))
        else:
            to_process.append(item)

    if to_process:
        images_list = [item[2] for item in to_process]
        predictions = ml_engine.predict_batch(images_list, return_errors=True)

        for (filename, file_hash, image, filepath), prediction in zip(
            to_process, predictions, strict=True
        ):
            if prediction is None or prediction.error:
                errors.append(f"{filename}: {prediction.error if prediction else 'Error'}")
                continue

            metrics = metrics_engine.compute_all(image, prediction.score, context, parallel=False)

            analysis = save_analysis_to_db(
                db, filename, file_hash, filepath,
                prediction, metrics, current_user.id
            )
            db.flush()
            results.append(AnalysisResponse.model_validate(analysis))

        db.commit()

    total_time = (time.perf_counter() - start_time) * 1000

    return BatchUploadResponse(
        total=len(files),
        processed=len(results),
        failed=len(errors),
        results=results,
        errors=errors[:20],
        total_inference_time_ms=round(total_time, 2),
    )


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

    return [AnalysisResponse.model_validate(a) for a in analyses]


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

    return AnalysisResponse.model_validate(analysis)


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

    try_delete_file(str(analysis.file_path) if analysis.file_path else None)
    try_delete_file(str(analysis.heatmap_path) if analysis.heatmap_path else None)

    db.delete(analysis)
    db.commit()

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

    file_path = Path(str(analysis.file_path))
    if not file_path.exists():
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

    if analysis.heatmap_path:
        heatmap_file = Path(str(analysis.heatmap_path))
        if heatmap_file.exists():
            return FileResponse(
                heatmap_file,
                media_type="image/jpeg",
                filename=f"heatmap_{analysis.filename}" if download else None,
            )

    file_path = Path(str(analysis.file_path))
    if not file_path.exists():
        raise HTTPException(404, "Original image not found")

    try:
        image = Image.open(file_path).convert("RGB")
        heatmap_image = ml_engine.generate_heatmap(image)

        if heatmap_image is None:
            raise HTTPException(400, "Heatmap not available for this model")

        heatmap_filename = f"heatmap_{analysis_id}_{uuid.uuid4().hex[:8]}.jpg"
        heatmap_path = settings.HEATMAPS_DIR / heatmap_filename
        heatmap_image.save(heatmap_path, "JPEG", quality=90)

        return FileResponse(
            heatmap_path,
            media_type="image/jpeg",
            filename=f"heatmap_{analysis.filename}" if download else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Heatmap error: {e}")


@router.post("/{analysis_id}/heatmap/save")
def save_heatmap(
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

    if analysis.heatmap_path and Path(str(analysis.heatmap_path)).exists():
        return {"status": "already_saved", "path": str(analysis.heatmap_path)}

    file_path = Path(str(analysis.file_path))
    if not file_path.exists():
        raise HTTPException(404, "Original image not found")

    try:
        image = Image.open(file_path).convert("RGB")
        heatmap_image = ml_engine.generate_heatmap(image)

        if heatmap_image is None:
            raise HTTPException(400, "Heatmap not available")

        heatmap_filename = f"heatmap_{analysis_id}.jpg"
        heatmap_path = settings.HEATMAPS_DIR / heatmap_filename
        heatmap_image.save(heatmap_path, "JPEG", quality=90)

        analysis.heatmap_path = str(heatmap_path)
        db.commit()

        return {"status": "saved", "path": str(heatmap_path)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")


@router.post("/batch/start", response_model=JobStatusResponse)
async def start_batch_job(
    req: BatchRequest,
    current_user: User = Depends(get_current_user),
) -> JobStatusResponse:
    try:
        job_id = await batch_processor.start_batch_job(
            req.path, current_user.id, req.recursive
        )
        job_status = batch_processor.get_job_status(job_id)

        return JobStatusResponse(
            job_id=job_id,
            status=job_status["status"],
            total=job_status["total"],
            processed=job_status["processed"],
            progress_percent=job_status["progress_percent"],
            current_file=job_status.get("current_file"),
            errors=job_status.get("errors", []),
            results=[],
        )
    except FileNotFoundError:
        raise HTTPException(404, "Folder not found")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/batch/{job_id}", response_model=JobStatusResponse)
def get_batch_status(
    job_id: str,
    include_results: bool = True,
    _current_user: User = Depends(get_current_user),  # Auth only
) -> JobStatusResponse:
    job_status = batch_processor.get_job_status(job_id)
    if not job_status:
        raise HTTPException(404, "Job not found")

    return JobStatusResponse(
        job_id=job_id,
        status=job_status["status"],
        total=job_status["total"],
        processed=job_status["processed"],
        progress_percent=job_status["progress_percent"],
        current_file=job_status.get("current_file"),
        errors=job_status.get("errors", [])[-10:],
        results=job_status.get("results", []) if include_results else [],
    )


@router.get("/batch/{job_id}/stream")
async def batch_progress_stream(
    job_id: str,
    _current_user: User = Depends(get_current_user),  # Auth only
) -> StreamingResponse:
    import asyncio
    import json

    async def event_generator():
        while True:
            job_status = batch_processor.get_job_status(job_id)
            if not job_status:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            yield f"data: {json.dumps(job_status)}\n\n"

            if job_status["status"] in ("completed", "failed"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/model/info")
def get_model_info() -> dict[str, Any]:
    return {
        "type": ml_engine.model_type,
        "backbone": ml_engine.backbone_name,
        "threshold": ml_engine.threshold,
        "device": str(ml_engine.device),
        "loaded": ml_engine.model_type != "unknown",
    }


@router.get("/metrics/available")
def get_available_metrics() -> dict[str, Any]:
    return metrics_engine.get_available_metrics()