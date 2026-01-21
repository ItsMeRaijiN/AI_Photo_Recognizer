from __future__ import annotations

import base64
from typing import Generator
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.core.database import Base, get_db
from backend.core.security import create_access_token, get_password_hash
from backend.models.analysis import Analysis
from backend.models.user import User

@pytest.fixture(scope="function")
def test_db() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

def _create_mock_ml_engine():
    mock = MagicMock()
    mock.is_loaded = True
    mock.model_type = "torch"
    mock.backbone_name = "convnext"
    mock.threshold = 0.5
    mock.model_info = {
        "type": "torch",
        "backbone": "convnext",
        "threshold": 0.5,
        "device": "cpu",
        "loaded": True,
    }

    result = MagicMock()
    result.score = 0.95
    result.is_ai = True
    result.confidence = 0.9
    result.error = None
    result.inference_time_ms = 50.0
    result.model_type = "torch"
    result.backbone_name = "convnext"
    result.threshold_used = 0.5

    mock.predict.return_value = result
    mock.predict_batch.side_effect = lambda images, **kwargs: [result] * len(images)

    from PIL import Image
    mock.generate_heatmap.return_value = Image.new("RGB", (10, 10), color="red")

    return mock


def _create_mock_metrics_engine():
    mock = MagicMock()
    mock.metric_count = 7
    mock.compute_all.return_value = {"blur_score": 0.1, "noise_level": 0.05}
    mock.get_available_metrics.return_value = {"blur": "Blur detection", "noise": "Noise analysis"}
    return mock


def _create_mock_batch_processor():
    mock = MagicMock()
    mock.start_batch_job = AsyncMock(return_value="job-123")
    mock.get_job_status.return_value = {
        "status": "completed",
        "total": 1,
        "processed": 1,
        "progress_percent": 100,
        "results": [],
        "errors": [],
    }
    mock.cleanup_old_jobs.return_value = 5
    return mock


@pytest.fixture(scope="function")
def mock_ml_services():
    with (
        patch("backend.routers.analysis.ml_engine") as mock_ml,
        patch("backend.routers.analysis.metrics_engine") as mock_metrics,
        patch("backend.routers.analysis.batch_processor") as mock_batch,
    ):
        ml = _create_mock_ml_engine()
        metrics = _create_mock_metrics_engine()
        batch = _create_mock_batch_processor()

        for attr in ["is_loaded", "model_type", "backbone_name", "threshold", "model_info",
                     "predict", "predict_batch", "generate_heatmap"]:
            setattr(mock_ml, attr, getattr(ml, attr))

        for attr in ["metric_count", "compute_all", "get_available_metrics"]:
            setattr(mock_metrics, attr, getattr(metrics, attr))

        for attr in ["start_batch_job", "get_job_status", "cleanup_old_jobs"]:
            setattr(mock_batch, attr, getattr(batch, attr))

        yield {
            "ml": mock_ml,
            "metrics": mock_metrics,
            "batch": mock_batch,
        }

@pytest.fixture(scope="function")
def client(test_db: Session, mock_ml_services) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_db] = lambda: test_db

    with patch.object(app.state, "ml_engine", mock_ml_services["ml"], create=True):
        with TestClient(app) as test_client:
            yield test_client

    app.dependency_overrides.clear()

@pytest.fixture
def sample_image_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def test_user(test_db: Session) -> User:
    user = User(
        username="testuser",
        hashed_password=get_password_hash("testpass123"),
        is_active=True,
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def test_admin(test_db: Session) -> User:
    admin = User(
        username="admin",
        hashed_password=get_password_hash("adminpass123"),
        is_active=True,
        is_superuser=True,
    )
    test_db.add(admin)
    test_db.commit()
    test_db.refresh(admin)
    return admin


@pytest.fixture
def auth_headers(test_user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(subject=test_user.username)}"}


@pytest.fixture
def admin_headers(test_admin: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(subject=test_admin.username)}"}


@pytest.fixture
def test_analysis(test_db: Session, test_user: User) -> Analysis:
    analysis = Analysis(
        filename="test.jpg",
        file_hash="abc123hash",
        file_path="/tmp/test.jpg",
        is_ai=True,
        score=0.95,
        confidence=0.9,
        inference_time_ms=50.0,
        model_type="torch",
        backbone_name="convnext",
        owner_id=test_user.id,
    )
    test_db.add(analysis)
    test_db.commit()
    test_db.refresh(analysis)
    return analysis