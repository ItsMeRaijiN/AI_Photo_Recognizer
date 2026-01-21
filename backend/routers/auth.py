from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.database import get_db
from backend.core.security import create_access_token, get_password_hash, verify_password
from backend.models.analysis import Analysis
from backend.models.user import User
from backend.routers.deps import get_current_user
from backend.schemas.auth import Token, UserCreate, UserResponse, UserStats

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)) -> User:
    if db.query(User).filter(User.username == user_in.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    new_user = User(
        username=user_in.username,
        hashed_password=get_password_hash(user_in.password),
        is_active=True,
        is_superuser=False,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


@router.post("/token", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> Token:
    user = db.query(User).filter(User.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    access_token = create_access_token(subject=user.username)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.get("/me/stats", response_model=UserStats)
def get_current_user_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserStats:
    total = db.query(Analysis).filter(Analysis.owner_id == current_user.id).count()
    ai_count = db.query(Analysis).filter(
        Analysis.owner_id == current_user.id,
        Analysis.is_ai == True,
    ).count()

    return UserStats(
        total_analyses=total,
        ai_detections=ai_count,
        human_detections=total - ai_count,
        ai_ratio=round(ai_count / max(1, total), 3),
    )
