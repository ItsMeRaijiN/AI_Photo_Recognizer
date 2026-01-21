import os
from unittest.mock import patch, MagicMock

from backend.core.config import Settings, _detect_device, _find_best_model, _is_placeholder


class TestDeviceDetection:
    def test_detects_cuda_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("torch.cuda.is_available", return_value=True):
                assert _detect_device() == "cuda"

    def test_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False, create=True):
                assert _detect_device() == "cpu"


class TestSettings:
    def test_default_paths_are_set(self, tmp_path):
        (tmp_path / "backend").mkdir()

        settings = Settings(
            BASE_DIR=tmp_path,
            SECRET_KEY="test-secret-key",
            ADMIN_BOOTSTRAP_TOKEN="test-bootstrap",
            DETECT_DEVICE=False,
            AUTO_DISCOVER_MODEL=False,
            CREATE_DIRS=False,
        )

        assert settings.UPLOAD_DIR == tmp_path / "uploads"
        assert settings.TEMP_DIR == tmp_path / "temp"
        assert settings.METRICS_DIR == tmp_path / "backend" / "custom_metrics"

    def test_sqlite_path_is_resolved(self, tmp_path):
        (tmp_path / "backend").mkdir()

        settings = Settings(
            BASE_DIR=tmp_path,
            DATABASE_URL="sqlite:///data.db",
            SECRET_KEY="s",
            ADMIN_BOOTSTRAP_TOKEN="t",
            DETECT_DEVICE=False,
            AUTO_DISCOVER_MODEL=False,
            CREATE_DIRS=False,
        )

        expected = f"sqlite:///{(tmp_path / 'data.db').as_posix()}"
        assert settings.DATABASE_URL == expected

    def test_validate_allowed_extensions(self, tmp_path):
        (tmp_path / "backend").mkdir()
        settings = Settings(
            BASE_DIR=tmp_path,
            SECRET_KEY="s",
            ADMIN_BOOTSTRAP_TOKEN="t",
            CREATE_DIRS=False,
        )

        assert settings.validate_file_extension("photo.jpg") is True
        assert settings.validate_file_extension("IMAGE.PNG") is True
        assert settings.validate_file_extension("test.webp") is True
        assert settings.validate_file_extension("script.py") is False
        assert settings.validate_file_extension("archive.zip") is False
        assert settings.validate_file_extension("document.pdf") is False

    def test_default_batch_settings(self, tmp_path):
        (tmp_path / "backend").mkdir()
        settings = Settings(
            BASE_DIR=tmp_path,
            SECRET_KEY="s",
            ADMIN_BOOTSTRAP_TOKEN="t",
            CREATE_DIRS=False,
        )

        assert settings.BATCH_MAX_WORKERS == 4
        assert settings.BATCH_INFERENCE_SIZE == 16
        assert settings.BATCH_JOB_TTL_HOURS == 24