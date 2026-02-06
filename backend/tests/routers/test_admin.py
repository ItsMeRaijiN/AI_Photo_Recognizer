from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.core.config import settings
from backend.models.analysis import Analysis


class TestAdminBootstrap:
    def test_bootstrap_success(self, client: TestClient, test_db):
        payload = {
            "username": "superadmin",
            "password": "verysecretpassword123",
            "secret_token": getattr(settings, "ADMIN_BOOTSTRAP_TOKEN", None),
        }
        response = client.post("/admin/bootstrap", json=payload)
        assert response.status_code == 201
        assert response.json()["username"] == "superadmin"

    def test_bootstrap_fails_when_admin_exists(self, client: TestClient, test_admin):
        payload = {"username": "hack", "password": "password123", "secret_token": "any"}
        response = client.post("/admin/bootstrap", json=payload)
        assert response.status_code == 400

    def test_bootstrap_invalid_token_rejected(self, client: TestClient, test_db, monkeypatch):
        monkeypatch.setattr(settings, "ADMIN_BOOTSTRAP_TOKEN", "my-secret-token", raising=False)
        payload = {
            "username": "superadmin2",
            "password": "verysecretpassword123",
            "secret_token": "WRONG",
        }
        response = client.post("/admin/bootstrap", json=payload)
        assert response.status_code == 403
        assert "Invalid bootstrap token" in response.json()["detail"]


class TestUserManagement:
    def test_list_users(self, client: TestClient, admin_headers: dict, test_user):
        response = client.get("/admin/users", headers=admin_headers)
        assert response.status_code == 200
        assert len(response.json()) >= 2  # admin + test_user

    def test_get_user_not_found(self, client: TestClient, admin_headers: dict):
        response = client.get("/admin/users/999999", headers=admin_headers)
        assert response.status_code == 404

    def test_delete_user_prevents_self_delete(self, client: TestClient, admin_headers: dict, test_admin):
        response = client.delete(f"/admin/users/{test_admin.id}", headers=admin_headers)
        assert response.status_code == 400
        assert "Cannot delete yourself" in response.json()["detail"]

    def test_toggle_user_active(self, client: TestClient, admin_headers: dict, test_user):
        response = client.patch(f"/admin/users/{test_user.id}/toggle-active", headers=admin_headers)
        assert response.status_code == 200
        assert "User" in response.json()["message"]

    def test_toggle_active_prevents_self_deactivate(self, client: TestClient, admin_headers: dict, test_admin):
        response = client.patch(f"/admin/users/{test_admin.id}/toggle-active", headers=admin_headers)
        assert response.status_code == 400
        assert "Cannot deactivate yourself" in response.json()["detail"]

    def test_toggle_user_admin(self, client: TestClient, admin_headers: dict, test_user):
        response = client.patch(f"/admin/users/{test_user.id}/toggle-admin", headers=admin_headers)
        assert response.status_code == 200
        assert "granted admin" in response.json()["message"]

    def test_toggle_admin_prevents_self_removal(self, client: TestClient, admin_headers: dict, test_admin):
        response = client.patch(f"/admin/users/{test_admin.id}/toggle-admin", headers=admin_headers)
        assert response.status_code == 400
        assert "Cannot remove your own admin rights" in response.json()["detail"]


class TestSystemStats:
    def test_get_stats(self, client: TestClient, admin_headers: dict):
        response = client.get("/admin/stats", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total_users"] >= 1
        assert "model_info" in data


class TestSystemCleanup:
    def test_cleanup_deletes_old_temp_and_orphans(
        self,
        client: TestClient,
        admin_headers: dict,
        test_db,
        tmp_path: Path,
        monkeypatch,
    ):
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir(parents=True)

        old_file = temp_dir / "old.tmp"
        old_file.write_bytes(b"x")
        cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()
        os.utime(old_file, (cutoff_time, cutoff_time))

        new_file = temp_dir / "new.tmp"
        new_file.write_bytes(b"x")

        monkeypatch.setattr(settings, "TEMP_DIR", temp_dir, raising=False)

        orphan = Analysis(
            file_hash="orphan",
            filename="x.jpg",
            file_path="/tmp/x.jpg",
            is_ai=False,
            score=0.1,
            confidence=0.9,
            threshold_used=0.5,
            inference_time_ms=1.0,
            model_type="torch",
            backbone_name="test",
            owner_id=None,
            created_at=datetime.now(timezone.utc) - timedelta(days=2),
        )
        test_db.add(orphan)
        test_db.commit()

        with patch("backend.routers.admin.batch_processor.cleanup_old_jobs", return_value=2):
            response = client.post("/admin/cleanup", headers=admin_headers)

        assert response.status_code == 200
        payload = response.json()
        assert payload["deleted_temp_files"] == 1
        assert payload["deleted_old_jobs"] == 2
        assert payload["deleted_orphan_analyses"] == 1


class TestModelUpload:
    def test_upload_invalid_extension_rejected(self, client: TestClient, admin_headers: dict):
        files = {"model": ("model.txt", b"content", "text/plain")}
        response = client.post("/admin/upload-model", headers=admin_headers, files=files)
        assert response.status_code == 400
        assert "must be .pt or .onnx" in response.json()["detail"]

    def test_upload_success(self, client: TestClient, admin_headers: dict, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(settings, "BASE_DIR", tmp_path, raising=False)

        files = {"model": ("best_model.pt", b"model_content", "application/octet-stream")}
        response = client.post("/admin/upload-model", headers=admin_headers, files=files)

        assert response.status_code == 200
        out_path = Path(response.json()["path"])
        assert out_path.exists()
        assert out_path.read_bytes() == b"model_content"