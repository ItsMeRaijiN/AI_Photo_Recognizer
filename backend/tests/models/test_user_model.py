import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.models.user import User


class TestUserModel:
    def test_create_user(self, test_db: Session):
        user = User(
            username="newuser",
            hashed_password="hashed_content",
            is_active=True,
        )
        test_db.add(user)
        test_db.commit()

        assert user.id is not None
        assert user.username == "newuser"
        assert user.is_superuser is False
        assert user.created_at is not None

    def test_username_must_be_unique(self, test_db: Session):
        user1 = User(username="duplicate", hashed_password="p1")
        test_db.add(user1)
        test_db.commit()

        user2 = User(username="duplicate", hashed_password="p2")
        test_db.add(user2)

        with pytest.raises(IntegrityError):
            test_db.commit()