from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token expiration in seconds")


class TokenData(BaseModel):
    username: str | None = None


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=100)


class UserUpdate(BaseModel):
    password: str | None = Field(default=None, min_length=6, max_length=100)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    is_active: bool
    is_superuser: bool
    created_at: datetime


class UserStats(BaseModel):
    total_analyses: int = Field(ge=0)
    ai_detections: int = Field(ge=0)
    human_detections: int = Field(ge=0)
    ai_ratio: float = Field(ge=0.0, le=1.0)
