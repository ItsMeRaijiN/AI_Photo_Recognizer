from unittest.mock import patch, MagicMock

import pytest

from backend.core.config import Settings, _detect_device, _find_best_model


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


class TestModelDiscovery:
    def test_discovers_onnx_from_latest_run(self, tmp_path):
        older_run = tmp_path / "runs" / "experiment" / "run_20260101_000000"
        newer_run = tmp_path / "runs" / "experiment" / "run_20260102_000000"
        older_run.mkdir(parents=True)
        newer_run.mkdir(parents=True)
        (older_run / "best_model.pt").write_bytes(b"pytorch")
        onnx_model = newer_run / "model.onnx"
        onnx_model.write_bytes(b"onnx")

        assert _find_best_model(tmp_path) == onnx_model


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
        assert settings.validate_file_extension("image.avif") is False
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

        assert settings.BATCH_INFERENCE_SIZE == 16
        assert settings.MAX_UPLOAD_SIZE_MB == 20

    def test_development_placeholder_secret_is_replaced(self, tmp_path):
        (tmp_path / "backend").mkdir()
        settings = Settings(
            BASE_DIR=tmp_path,
            ENVIRONMENT="development",
            SECRET_KEY="change-me-in-production",
            ADMIN_BOOTSTRAP_TOKEN="t",
            DETECT_DEVICE=False,
            AUTO_DISCOVER_MODEL=False,
            CREATE_DIRS=False,
        )
        assert settings.SECRET_KEY != "change-me-in-production"
        assert len(settings.SECRET_KEY) >= 48

    def test_production_rejects_placeholder_secret(self, tmp_path):
        (tmp_path / "backend").mkdir()
        with pytest.raises(ValueError, match="SECRET_KEY"):
            Settings(
                BASE_DIR=tmp_path,
                ENVIRONMENT="production",
                SECRET_KEY="change-me-in-production",
                ADMIN_BOOTSTRAP_TOKEN="t",
                DETECT_DEVICE=False,
                AUTO_DISCOVER_MODEL=False,
                CREATE_DIRS=False,
            )
