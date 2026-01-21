from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.database import get_db
from backend.core.security import get_password_hash
from backend.models.analysis import Analysis
from backend.models.user import User
from backend.routers.deps import get_current_superuser
from backend.schemas.auth import UserResponse
from backend.services.batch_manager import batch_processor
from backend.services.metrics_loader import metrics_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

class BootstrapRequest(BaseModel):
    """Request to create initial admin."""
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=100)
    secret_token: str | None = Field(
        default=None,
        description="Required if ADMIN_BOOTSTRAP_TOKEN is set in config",
    )

class SystemStats(BaseModel):
    total_users: int
    active_users: int
    total_analyses: int
    ai_detections: int
    human_detections: int
    ai_ratio: float
    analyses_today: int
    storage_used_mb: float
    model_info: dict[str, Any]
    metrics_count: int


class CleanupResult(BaseModel):
    deleted_temp_files: int
    deleted_old_jobs: int
    deleted_orphan_analyses: int


@router.post("/bootstrap", status_code=status.HTTP_201_CREATED)
def create_initial_admin(
    req: BootstrapRequest,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    existing_admin = db.query(User).filter(User.is_superuser == True).first()

    if existing_admin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin already exists. Use normal login.",
        )

    configured_token = getattr(settings, "ADMIN_BOOTSTRAP_TOKEN", None)
    if configured_token and configured_token not in ("", "change-me-in-production"):
        if req.secret_token != configured_token:
            logger.warning(f"Bootstrap attempt with invalid token for user: {req.username}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid bootstrap token",
            )

    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken",
        )

    new_admin = User(
        username=req.username,
        hashed_password=get_password_hash(req.password),
        is_superuser=True,
        is_active=True,
    )

    db.add(new_admin)
    db.commit()

    logger.info(f"Initial admin created: {req.username}")

    return {"message": "Admin created successfully", "username": req.username}


@router.get("/users", response_model=list[UserResponse])
def list_users(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_superuser),
) -> list[User]:
    return db.query(User).offset(skip).limit(limit).all()


@router.get("/users/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_superuser),
) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_superuser),
) -> dict[str, str]:
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    username = user.username
    db.delete(user)
    db.commit()

    return {"message": f"User {username} deleted"}


@router.patch("/users/{user_id}/toggle-active")
def toggle_user_active(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_superuser),
) -> dict[str, str]:
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")

    user.is_active = not user.is_active
    db.commit()

    status_str = "activated" if user.is_active else "deactivated"
    return {"message": f"User {user.username} {status_str}"}


@router.patch("/users/{user_id}/toggle-admin")
def toggle_user_admin(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_superuser),
) -> dict[str, str]:
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot remove your own admin rights")

    if user.is_superuser:
        admin_count = db.query(User).filter(User.is_superuser == True).count()  # noqa: E712
        if admin_count <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot remove last admin",
            )

    user.is_superuser = not user.is_superuser
    db.commit()

    status_str = "granted admin" if user.is_superuser else "revoked admin"
    return {"message": f"User {user.username}: {status_str}"}


@router.get("/stats", response_model=SystemStats)
def get_system_stats(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_superuser),
) -> SystemStats:
    from backend.services.ml_engine import ml_engine

    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()

    total_analyses = db.query(Analysis).count()
    ai_detections = db.query(Analysis).filter(Analysis.is_ai == True).count()
    human_detections = total_analyses - ai_detections

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    analyses_today = db.query(Analysis).filter(Analysis.created_at >= today_start).count()

    storage_mb = 0.0
    for dir_path in [settings.UPLOAD_DIR, settings.HEATMAPS_DIR]:
        if dir_path and dir_path.exists():
            for f in dir_path.iterdir():
                if f.is_file():
                    storage_mb += f.stat().st_size / (1024 * 1024)

    return SystemStats(
        total_users=total_users,
        active_users=active_users,
        total_analyses=total_analyses,
        ai_detections=ai_detections,
        human_detections=human_detections,
        ai_ratio=round(ai_detections / max(1, total_analyses), 3),
        analyses_today=analyses_today,
        storage_used_mb=round(storage_mb, 2),
        model_info=ml_engine.model_info,
        metrics_count=metrics_engine.metric_count,
    )


@router.post("/cleanup", response_model=CleanupResult)
def cleanup_system(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_superuser),
) -> CleanupResult:
    deleted_temp = 0
    deleted_orphans = 0

    if settings.TEMP_DIR and settings.TEMP_DIR.exists():
        cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()
        for f in settings.TEMP_DIR.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff_time:
                try:
                    f.unlink()
                    deleted_temp += 1
                except OSError:
                    pass

    deleted_jobs = batch_processor.cleanup_old_jobs(max_age_hours=settings.BATCH_JOB_TTL_HOURS)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    orphans = db.query(Analysis).filter(
        Analysis.owner_id == None,
        Analysis.created_at < cutoff,
    ).all()

    for orphan in orphans:
        db.delete(orphan)
        deleted_orphans += 1

    db.commit()

    return CleanupResult(
        deleted_temp_files=deleted_temp,
        deleted_old_jobs=deleted_jobs,
        deleted_orphan_analyses=deleted_orphans,
    )


@router.post("/metrics/reload")
def reload_metrics(
    _admin: User = Depends(get_current_superuser),
) -> dict[str, Any]:
    from backend.services.metrics_loader import metrics_engine

    metrics_engine.reload_metrics()

    return {
        "message": "Metrics reloaded",
        "count": metrics_engine.metric_count,
        "available": list(metrics_engine.get_available_metrics().keys()),
    }


@router.post("/optimize-db")
def optimize_database(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_superuser),
) -> dict[str, str]:
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")

    if not Path(db_path).exists():
        raise HTTPException(status_code=400, detail="Database file not found")

    size_before = Path(db_path).stat().st_size

    try:
        db.close()

        conn = sqlite3.connect(db_path)
        conn.execute("VACUUM")
        conn.execute("REINDEX")
        conn.close()

        size_after = Path(db_path).stat().st_size

        return {
            "message": "Database optimized",
            "size_before": f"{size_before / 1024:.1f} KB",
            "size_after": f"{size_after / 1024:.1f} KB",
            "saved": f"{(size_before - size_after) / 1024:.1f} KB",
        }
    except Exception as e:
        logger.error(f"Database optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {e}")


@router.post("/upload-model")
async def upload_model(
    model: UploadFile = File(...),
    _admin: User = Depends(get_current_superuser),  # Auth only
) -> dict[str, str]:
    filename = model.filename or "uploaded_model"

    if not (filename.endswith(".pt") or filename.endswith(".onnx")):
        raise HTTPException(status_code=400, detail="Model must be .pt or .onnx file")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest_dir = settings.BASE_DIR / "runs" / "experiment" / f"run_{timestamp}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_filename = "best_model.pt" if filename.endswith(".pt") else "model.onnx"
    dest_path = dest_dir / dest_filename

    try:
        content = await model.read()
        dest_path.write_bytes(content)

        logger.info(f"Model uploaded: {dest_path}")

        return {
            "message": "Model uploaded successfully",
            "path": str(dest_path),
            "note": "Restart server to load new model",
        }
    except Exception as e:
        logger.error(f"Model upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {e}")
    finally:
        await model.close()
