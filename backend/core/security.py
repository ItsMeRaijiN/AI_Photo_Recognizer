import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Dict

from jose import jwt, JWTError
from passlib.context import CryptContext

from .config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(
    subject: str | int,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.
    """
    now = datetime.now(timezone.utc)

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "exp": int(expire.timestamp()),
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "jti": secrets.token_urlsafe(16)
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def decode_token_payload(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode JWT and return full payload.
    """
    try:
        return jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
    except JWTError:
        return None


def decode_token(token: str) -> Optional[str]:
    """
    Decode and validate JWT token.
    """
    payload = decode_token_payload(token)
    return payload.get("sub") if payload else None