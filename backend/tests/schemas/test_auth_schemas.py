import pytest
from pydantic import ValidationError

from backend.schemas.auth import UserCreate, UserUpdate


class TestUserCreate:
    def test_password_too_short_rejected(self):
        with pytest.raises(ValidationError):
            UserCreate(username="validuser", password="123")

    def test_username_too_short_rejected(self):
        with pytest.raises(ValidationError):
            UserCreate(username="ab", password="securepassword123")


class TestUserUpdate:
    def test_empty_update_allowed(self):
        update = UserUpdate()
        assert update.password is None