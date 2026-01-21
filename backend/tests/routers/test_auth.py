from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models.user import User


class TestRegistration:
    def test_register_success(self, client: TestClient, test_db: Session):
        response = client.post(
            "/auth/register",
            json={"username": "newuser", "password": "securepassword123"},
        )
        assert response.status_code == 201
        assert "hashed_password" not in response.json()

    def test_register_duplicate_username_fails(self, client: TestClient, test_user: User):
        response = client.post(
            "/auth/register",
            json={"username": test_user.username, "password": "somepassword"},
        )
        assert response.status_code == 400


class TestLogin:
    def test_login_success(self, client: TestClient, test_user: User):
        response = client.post(
            "/auth/token",
            data={"username": test_user.username, "password": "testpass123"},
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_login_wrong_password_fails(self, client: TestClient, test_user: User):
        response = client.post(
            "/auth/token",
            data={"username": test_user.username, "password": "wrongpassword"},
        )
        assert response.status_code == 401


class TestCurrentUser:
    def test_get_me_returns_user_info(self, client: TestClient, auth_headers: dict):
        response = client.get("/auth/me", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["username"] == "testuser"

    def test_get_me_without_auth_fails(self, client: TestClient):
        response = client.get("/auth/me")
        assert response.status_code == 401