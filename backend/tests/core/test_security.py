from datetime import timedelta
from jose import jwt

from backend.core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_token,
    decode_token_payload
)
from backend.core.config import settings


class TestPasswordHashing:
    def test_hash_is_different_from_plaintext(self):
        password = "secret_password"
        hashed = get_password_hash(password)
        assert hashed != password

    def test_correct_password_verifies(self):
        password = "secret_password"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed) is True

    def test_wrong_password_fails(self):
        hashed = get_password_hash("correct_password")
        assert verify_password("wrong_password", hashed) is False


class TestJWTTokens:
    def test_token_contains_required_claims(self):
        token = create_access_token(subject="testuser")
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

        assert payload["sub"] == "testuser"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

    def test_custom_expiration_delta(self):
        token = create_access_token(subject="user", expires_delta=timedelta(minutes=5))
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

        assert payload["exp"] - payload["iat"] == 300

    def test_decode_valid_token(self):
        token = create_access_token(subject="valid_user")
        assert decode_token(token) == "valid_user"

    def test_decode_invalid_token_returns_none(self):
        assert decode_token("invalid.token.garbage") is None

    def test_decode_tampered_token_returns_none(self):
        token = create_access_token(subject="user")
        tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
        assert decode_token(tampered) is None

    def test_decode_token_payload_returns_full_payload(self):
        token = create_access_token(subject="user")
        payload = decode_token_payload(token)

        assert payload is not None
        assert payload["sub"] == "user"
        assert "exp" in payload
        assert "jti" in payload