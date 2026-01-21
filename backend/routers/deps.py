from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.security import decode_token
from backend.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)


async def get_current_user_or_none(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    db: Session = Depends(get_db),
) -> User | None:
    if not token:
        return None

    username = decode_token(token)
    if not username:
        return None

    user = db.query(User).filter(User.username == username).first()

    if user and not user.is_active:
        return None

    return user


async def get_current_user(
    user: User | None = Depends(get_current_user_or_none),
) -> User:
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_active_user(
    user: User = Depends(get_current_user),
) -> User:
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )
    return user


async def get_current_superuser(
    user: User = Depends(get_current_active_user),
) -> User:
    if not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user
