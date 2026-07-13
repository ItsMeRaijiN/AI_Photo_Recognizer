from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from backend.models.analysis import Analysis


class TestAnalysisMetadata:
    def test_model_info_static_route_is_reachable(self, client: TestClient):
        resp = client.get("/analysis/model/info")
        assert resp.status_code == 200
        assert resp.json()["version"] == "test-fingerprint"

    def test_metrics_static_route_is_reachable(self, client: TestClient):
        resp = client.get("/analysis/metrics/available")
        assert resp.status_code == 200
        assert "blur" in resp.json()


class TestPredictSingle:
    def test_predict_guest_user(self, client: TestClient, sample_image_bytes: bytes):
        files = {"file": ("test.png", sample_image_bytes, "image/png")}
        resp = client.post("/analysis/predict", files=files)
        assert resp.status_code == 200
        body = resp.json()
        assert "score" in body
        assert "is_ai" in body

    def test_predict_invalid_image_returns_400(self, client: TestClient, auth_headers: dict):
        files = {"file": ("bad.png", b"not-an-image", "image/png")}
        resp = client.post("/analysis/predict", headers=auth_headers, files=files)
        assert resp.status_code == 400

    def test_predict_rejects_invalid_optional_token(self, client: TestClient, sample_image_bytes: bytes):
        files = {"file": ("test.png", sample_image_bytes, "image/png")}
        resp = client.post(
            "/analysis/predict",
            headers={"Authorization": "Bearer invalid-token"},
            files=files,
        )
        assert resp.status_code == 401

    def test_predict_cache_hit_same_model(self, client: TestClient, auth_headers: dict, test_db, test_user):
        img = Image.new("RGB", (16, 16), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        payload = buf.getvalue()

        from backend.routers.analysis import compute_file_hash

        file_hash = compute_file_hash(payload)

        analysis = Analysis(
            filename="cached.png",
            file_hash=file_hash,
            file_path="/tmp/cached.png",
            is_ai=True,
            score=0.99,
            confidence=0.99,
            inference_time_ms=12.0,
            model_type="torch",
            backbone_name="convnext",
            model_version="test-fingerprint",
            threshold_used=0.5,
            custom_metrics={"__metrics_version": "test-metrics-fingerprint"},
            created_at=datetime.now(timezone.utc),
            owner_id=test_user.id,
        )
        test_db.add(analysis)
        test_db.commit()

        files = {"file": ("cached.png", payload, "image/png")}
        resp = client.post("/analysis/predict", headers=auth_headers, files=files)
        assert resp.status_code == 200
        assert resp.json()["score"] == 0.99

    def test_predict_cache_miss_for_different_model(
        self, client: TestClient, auth_headers: dict, test_db, test_user
    ):
        img = Image.new("RGB", (16, 16), color=(40, 50, 60))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        payload = buf.getvalue()

        from backend.routers.analysis import compute_file_hash

        analysis = Analysis(
            filename="stale.png",
            file_hash=compute_file_hash(payload),
            file_path="/tmp/stale.png",
            is_ai=True,
            score=0.99,
            confidence=0.99,
            inference_time_ms=12.0,
            model_type="torch",
            backbone_name="convnext",
            model_version="old-model-weights",
            threshold_used=0.5,
            created_at=datetime.now(timezone.utc),
            owner_id=test_user.id,
        )
        test_db.add(analysis)
        test_db.commit()

        files = {"file": ("stale.png", payload, "image/png")}
        resp = client.post("/analysis/predict", headers=auth_headers, files=files)
        assert resp.status_code == 200
        assert resp.json()["score"] == 0.95

    def test_predict_cache_miss_for_changed_metrics(
        self, client: TestClient, auth_headers: dict, test_db, test_user
    ):
        img = Image.new("RGB", (16, 16), color=(80, 90, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        payload = buf.getvalue()
        from backend.routers.analysis import compute_file_hash

        test_db.add(Analysis(
            filename="old-metrics.png", file_hash=compute_file_hash(payload),
            file_path="/tmp/old-metrics.png", is_ai=True, score=0.99, confidence=0.99,
            inference_time_ms=12.0, model_type="torch", backbone_name="convnext",
            model_version="test-fingerprint", threshold_used=0.5,
            custom_metrics={"__metrics_version": "old-metrics"},
            created_at=datetime.now(timezone.utc), owner_id=test_user.id,
        ))
        test_db.commit()

        resp = client.post(
            "/analysis/predict", headers=auth_headers,
            files={"file": ("old-metrics.png", payload, "image/png")},
        )

        assert resp.status_code == 200
        assert resp.json()["score"] == 0.95

    def test_predict_rejects_oversized_upload(self, client: TestClient):
        from backend.core.config import settings

        oversized = b"x" * (settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
        files = {"file": ("big.png", oversized, "image/png")}
        resp = client.post("/analysis/predict", files=files)
        assert resp.status_code == 413

    def test_predict_rejects_excessive_pixel_count(self, client: TestClient, monkeypatch):
        from backend.core.config import settings

        img = Image.new("RGB", (11, 11), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 100)

        resp = client.post("/analysis/predict", files={"file": ("pixels.png", buf.getvalue(), "image/png")})
        assert resp.status_code == 413


class TestPredictBatch:
    def test_batch_max_files_limit(self, client: TestClient, auth_headers: dict, sample_image_bytes: bytes):
        files = [("files", (f"img{i}.png", sample_image_bytes, "image/png")) for i in range(101)]
        resp = client.post("/analysis/predict/batch", headers=auth_headers, files=files)
        assert resp.status_code in (400, 422)

    def test_batch_mixed_valid_invalid(self, client: TestClient, auth_headers: dict, sample_image_bytes: bytes):
        files = [
            ("files", ("ok.png", sample_image_bytes, "image/png")),
            ("files", ("bad.png", b"xxx", "image/png")),
        ]
        resp = client.post("/analysis/predict/batch", headers=auth_headers, files=files)
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert "errors" in body

    def test_batch_preserves_source_indexes_for_duplicate_names(
        self, client: TestClient, auth_headers: dict, sample_image_bytes: bytes
    ):
        files = [
            ("files", ("same.png", sample_image_bytes, "image/png")),
            ("files", ("same.png", sample_image_bytes, "image/png")),
        ]
        resp = client.post("/analysis/predict/batch", headers=auth_headers, files=files)

        assert resp.status_code == 200
        assert [item["source_index"] for item in resp.json()["results"]] == [0, 1]


class TestHistory:
    def test_history_pagination(self, client: TestClient, auth_headers: dict, test_db, test_user):
        for i in range(5):
            a = Analysis(
                filename=f"a{i}.png",
                file_hash=f"h{i}",
                file_path=f"/tmp/a{i}.png",
                is_ai=False,
                score=0.1,
                confidence=0.9,
                inference_time_ms=1.0,
                model_type="torch",
                backbone_name="convnext",
                owner_id=test_user.id,
                created_at=datetime.now(timezone.utc),
            )
            test_db.add(a)
        test_db.commit()

        resp = client.get("/analysis/history?skip=1&limit=2", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 2


class TestAnalysisCRUD:
    def test_get_analysis_success(self, client: TestClient, auth_headers: dict, test_analysis: Analysis):
        resp = client.get(f"/analysis/{test_analysis.id}", headers=auth_headers)
        assert resp.status_code == 200

    def test_get_analysis_not_found(self, client: TestClient, auth_headers: dict):
        resp = client.get("/analysis/999999", headers=auth_headers)
        assert resp.status_code == 404

    def test_delete_analysis(
        self, client: TestClient, auth_headers: dict, test_db, test_user, tmp_path: Path
    ):
        img = tmp_path / "x.png"
        img.write_bytes(b"x")

        a = Analysis(
            filename="x.png",
            file_hash="hx",
            file_path=str(img),
            is_ai=False,
            score=0.2,
            confidence=0.8,
            inference_time_ms=2.0,
            model_type="torch",
            backbone_name="convnext",
            owner_id=test_user.id,
            created_at=datetime.now(timezone.utc),
        )
        test_db.add(a)
        test_db.commit()
        test_db.refresh(a)

        resp = client.delete(f"/analysis/{a.id}", headers=auth_headers)
        assert resp.status_code == 200
        assert test_db.query(Analysis).filter(Analysis.id == a.id).first() is None
        assert img.exists()

    def test_delete_analysis_removes_managed_upload(
        self, client: TestClient, auth_headers: dict, test_db, test_user, tmp_path: Path
    ):
        from backend.core.config import settings

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        img = upload_dir / "managed.png"
        img.write_bytes(b"x")

        a = Analysis(
            filename="managed.png",
            file_hash="hm",
            file_path=str(img),
            is_ai=False,
            score=0.2,
            confidence=0.8,
            inference_time_ms=2.0,
            model_type="torch",
            backbone_name="convnext",
            owner_id=test_user.id,
            created_at=datetime.now(timezone.utc),
        )
        test_db.add(a)
        test_db.commit()
        test_db.refresh(a)

        with patch.object(settings, "UPLOAD_DIR", upload_dir):
            resp = client.delete(f"/analysis/{a.id}", headers=auth_headers)

        assert resp.status_code == 200
        assert not img.exists()


class TestAnalysisImage:
    def test_get_image_success(
        self, client: TestClient, auth_headers: dict, test_db, test_user, tmp_path: Path
    ):
        from backend.core.config import settings

        img = tmp_path / "x.png"
        img.write_bytes(b"image_data")

        a = Analysis(
            filename="x.png",
            file_hash="hx",
            file_path=str(img),
            is_ai=False,
            score=0.2,
            confidence=0.8,
            inference_time_ms=2.0,
            model_type="torch",
            backbone_name="convnext",
            owner_id=test_user.id,
            created_at=datetime.now(timezone.utc),
        )
        test_db.add(a)
        test_db.commit()
        test_db.refresh(a)

        with patch.object(settings, "UPLOAD_DIR", tmp_path):
            resp = client.get(f"/analysis/{a.id}/image", headers=auth_headers)
        assert resp.status_code == 200

    def test_get_image_file_missing(
        self, client: TestClient, auth_headers: dict, test_db, test_user, tmp_path: Path
    ):
        img = tmp_path / "deleted.png"
        img.write_bytes(b"x")

        a = Analysis(
            filename="deleted.png",
            file_hash="hx",
            file_path=str(img),
            is_ai=False,
            score=0.2,
            confidence=0.8,
            inference_time_ms=2.0,
            model_type="torch",
            backbone_name="convnext",
            owner_id=test_user.id,
            created_at=datetime.now(timezone.utc),
        )
        test_db.add(a)
        test_db.commit()
        test_db.refresh(a)

        img.unlink()

        resp = client.get(f"/analysis/{a.id}/image", headers=auth_headers)
        assert resp.status_code == 404


class TestHeatmap:
    def test_get_cached_heatmap(
        self, client: TestClient, auth_headers: dict, test_db, test_user, tmp_path: Path
    ):
        hm = tmp_path / "hm.jpg"
        hm.write_bytes(b"heatmap_data")

        a = Analysis(
            filename="x.png",
            file_hash="hx",
            file_path=str(tmp_path / "x.png"),
            is_ai=True,
            score=0.9,
            confidence=0.9,
            inference_time_ms=5.0,
            model_type="torch",
            backbone_name="effnet",
            model_version="test-fingerprint",
            owner_id=test_user.id,
            heatmap_path=str(hm),
            created_at=datetime.now(timezone.utc),
        )
        test_db.add(a)
        test_db.commit()
        test_db.refresh(a)

        from backend.core.config import settings

        with patch.object(settings, "HEATMAPS_DIR", tmp_path):
            resp = client.get(f"/analysis/{a.id}/heatmap", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/")

    def test_rejects_heatmap_from_different_model(
        self, client: TestClient, auth_headers: dict, test_db, test_user
    ):
        a = Analysis(
            filename="old.png", file_hash="old", file_path="/tmp/old.png",
            is_ai=True, score=0.9, confidence=0.9, inference_time_ms=5.0,
            model_type="torch", backbone_name="effnet", model_version="old-model",
            owner_id=test_user.id, created_at=datetime.now(timezone.utc),
        )
        test_db.add(a)
        test_db.commit()
        test_db.refresh(a)

        resp = client.get(f"/analysis/{a.id}/heatmap", headers=auth_headers)
        assert resp.status_code == 409


    def test_generated_heatmap_is_persisted(
        self, client: TestClient, auth_headers: dict, test_db, test_user, tmp_path: Path
    ):
        img_path = tmp_path / "x.png"
        img = Image.new("RGB", (64, 64), color=(20, 20, 20))
        img.save(img_path)

        a = Analysis(
            filename="x.png",
            file_hash="hx",
            file_path=str(img_path),
            is_ai=True,
            score=0.9,
            confidence=0.9,
            inference_time_ms=5.0,
            model_type="torch",
            backbone_name="effnet",
            model_version="test-fingerprint",
            owner_id=test_user.id,
            heatmap_path=None,
            created_at=datetime.now(timezone.utc),
        )
        test_db.add(a)
        test_db.commit()
        test_db.refresh(a)

        from backend.core.config import settings

        with (
            patch.object(settings, "UPLOAD_DIR", tmp_path),
            patch.object(settings, "HEATMAPS_DIR", tmp_path),
        ):
            resp = client.get(f"/analysis/{a.id}/heatmap", headers=auth_headers)

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/")
        test_db.refresh(a)
        assert a.heatmap_path is not None
        assert Path(a.heatmap_path).exists()
