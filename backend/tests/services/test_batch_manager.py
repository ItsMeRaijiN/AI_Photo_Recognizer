from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from backend.services.batch_manager import BatchJob, BatchProcessor, LoadedImage
from backend.services.ml_engine import PredictionResult


@pytest.fixture
def processor() -> BatchProcessor:
    return BatchProcessor(batch_size=2)


class TestFileOperations:
    def test_collect_files_filters_by_extension(self, processor: BatchProcessor, tmp_path: Path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"x")
        (tmp_path / "c.txt").write_text("x")

        files = processor._collect_files(tmp_path, recursive=False)

        assert sorted(p.name for p in files) == ["a.jpg", "b.png"]

    def test_load_single_image_returns_none_on_failure(self, processor: BatchProcessor, tmp_path: Path):
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"not an image")

        with patch("backend.services.batch_manager.Image.open", side_effect=OSError("bad")):
            assert processor._load_single_image(p) is None


class TestCaching:
    def test_splits_cached_and_new(self, processor: BatchProcessor, tmp_path: Path):
        img1 = LoadedImage(filepath=tmp_path / "a.jpg", image=Image.new("RGB", (8, 8)), file_hash="h1")
        img2 = LoadedImage(filepath=tmp_path / "b.jpg", image=Image.new("RGB", (8, 8)), file_hash="h2")

        cached_analysis = MagicMock()
        cached_analysis.id = 10
        cached_analysis.file_hash = "h1"
        cached_analysis.is_ai = True
        cached_analysis.score = 0.9
        cached_analysis.confidence = 0.9
        cached_analysis.inference_time_ms = 1.0
        cached_analysis.model_type = "torch"
        cached_analysis.backbone_name = "x"
        cached_analysis.threshold_used = 0.5
        cached_analysis.custom_metrics = {"m": 1}
        cached_analysis.has_heatmap = False
        cached_analysis.created_at = datetime.now(timezone.utc)

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [cached_analysis]

        to_process, cached_results = processor._check_cache_batch([img1, img2], owner_id=1, db=db)

        assert [i.file_hash for i in to_process] == ["h2"]
        assert cached_results[0]["from_cache"] is True


class TestBatchProcessing:
    def test_records_prediction_errors(self, processor: BatchProcessor, tmp_path: Path):
        job = BatchJob(job_id="j", owner_id=1, total=1)
        li = LoadedImage(filepath=tmp_path / "a.jpg", image=Image.new("RGB", (8, 8)), file_hash="h")

        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.all.return_value = []

        class _Ctx:
            def __enter__(self):
                return fake_db
            def __exit__(self, *args):
                return False

        pred_err = PredictionResult(
            score=0.5,
            is_ai=False,
            confidence=0.5,
            inference_time_ms=0,
            model_type="torch",
            backbone_name="x",
            threshold_used=0.5,
            error="boom",
        )

        with (
            patch("backend.services.batch_manager.SessionLocal", return_value=_Ctx()),
            patch("backend.services.batch_manager.ml_engine.predict_batch", return_value=[pred_err]),
            patch("backend.services.batch_manager.metrics_engine.compute_all") as mock_metrics,
        ):
            out = processor._process_batch([li], owner_id=1, job=job)

        assert out == []
        assert job.errors and "boom" in job.errors[0]
        mock_metrics.assert_not_called()


class TestJobLifecycle:
    @pytest.mark.asyncio
    async def test_start_rejects_nonexistent_path(self, processor: BatchProcessor):
        with pytest.raises(FileNotFoundError):
            await processor.start_batch_job("/non/existent/path", owner_id=1)

    @pytest.mark.asyncio
    async def test_worker_completes_job(self, processor: BatchProcessor, tmp_path: Path):
        files = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
        job = BatchJob(job_id="j", owner_id=1, total=2, started_at=datetime.now(timezone.utc))

        loaded = [LoadedImage(filepath=files[0], image=Image.new("RGB", (8, 8)), file_hash="h1")]
        load_errors = {files[1]: "Failed to load image"}
        batch_results = [{"filename": "a.jpg", "from_cache": False}]

        async def fake_executor(_executor, func, *args):
            return func(*args)

        class DummyLoop:
            run_in_executor = staticmethod(fake_executor)

        with (
            patch("asyncio.get_running_loop", return_value=DummyLoop()),
            patch.object(processor, "_load_images_parallel", return_value=(loaded, load_errors)),
            patch.object(processor, "_process_batch", return_value=batch_results),
            patch.object(processor, "_notify_progress", new=AsyncMock()),
        ):
            await processor._run_batch_worker(job, files, owner_id=1)

        assert job.status == "completed"
        assert job.processed == 2
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_worker_marks_failed_on_exception(self, processor: BatchProcessor, tmp_path: Path):
        files = [tmp_path / "a.jpg"]
        job = BatchJob(job_id="j", owner_id=1, total=1, started_at=datetime.now(timezone.utc))

        async def crash_executor(_executor, func, *args):
            raise RuntimeError("crash")

        class DummyLoop:
            run_in_executor = staticmethod(crash_executor)

        with (
            patch("asyncio.get_running_loop", return_value=DummyLoop()),
            patch.object(processor, "_notify_progress", new=AsyncMock()),
        ):
            await processor._run_batch_worker(job, files, owner_id=1)

        assert job.status == "failed"
        assert any("Job failed" in e for e in job.errors)


class TestJobCleanup:
    def test_cleanup_removes_old_jobs(self, processor: BatchProcessor):
        old = BatchJob(
            job_id="old",
            owner_id=1,
            status="completed",
            total=1,
            processed=1,
            completed_at=datetime.now(timezone.utc) - timedelta(hours=30),
        )
        new = BatchJob(
            job_id="new",
            owner_id=1,
            status="completed",
            total=1,
            processed=1,
            completed_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        processor._jobs = {"old": old, "new": new}
        processor._progress_callbacks["old"] = [MagicMock()]

        cleaned = processor.cleanup_old_jobs(max_age_hours=24)

        assert cleaned == 1
        assert "old" not in processor._jobs
        assert "old" not in processor._progress_callbacks
        assert "new" in processor._jobs