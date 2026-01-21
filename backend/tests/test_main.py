from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


class TestRootEndpoint:
    def test_returns_system_info(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["system"] == "AI Photo Recognizer"
        assert data["status"] == "online"
        assert "version" in data

    def test_shows_model_status(self, client: TestClient):
        response = client.get("/")
        data = response.json()

        assert "model" in data
        assert data["model"]["loaded"] is True
        assert data["model"]["backbone"] in ("convnext", "effnetv2")


class TestHealthEndpoint:
    def test_healthy_when_all_services_up(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert data["model_loaded"] is True

    def test_degraded_when_database_fails(self, client: TestClient):
        with patch("backend.main.SessionLocal") as mock_session_cls:
            mock_session = mock_session_cls.return_value
            mock_session.execute.side_effect = Exception("DB Connection Error")

            response = client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["database"] == "error"
            assert data["status"] == "degraded"


class TestApplicationStartup:
    def test_app_starts_when_metrics_fail(self):
        from backend.main import app

        with patch("backend.services.metrics_loader.metrics_engine") as mock_metrics:
            mock_metrics.metric_count = MagicMock(side_effect=Exception("Metrics fail"))

            with patch("backend.services.ml_engine.ml_engine") as mock_ml:
                mock_ml.is_loaded = True
                mock_ml.model_type = "torch"
                mock_ml.backbone_name = "convnext"
                mock_ml.threshold = 0.5

                with TestClient(app) as test_client:
                    response = test_client.get("/")
                    assert response.status_code == 200

    def test_app_shows_model_not_loaded(self):
        from backend.main import app

        with patch("backend.services.ml_engine.ml_engine") as mock_ml:
            mock_ml.is_loaded = False
            mock_ml.model_type = "unknown"
            mock_ml.backbone_name = "unknown"
            mock_ml.threshold = 0.5

            with TestClient(app) as test_client:
                response = test_client.get("/")
                data = response.json()
                assert data["model"]["loaded"] is False
                health = test_client.get("/health").json()
                assert health["status"] == "degraded"