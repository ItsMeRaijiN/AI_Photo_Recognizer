"""
AI Photo Recognizer - Pydantic Schemas
"""

from .analysis import AnalysisResponse, BatchUploadResponse
from .auth import (
    Token,
    TokenData,
    UserCreate,
    UserResponse,
    UserStats,
    UserUpdate,
)

__all__ = [
    "AnalysisResponse",
    "BatchUploadResponse",
    "Token",
    "TokenData",
    "UserCreate",
    "UserResponse",
    "UserStats",
    "UserUpdate",
]
