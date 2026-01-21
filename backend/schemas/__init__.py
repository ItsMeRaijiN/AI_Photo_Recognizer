"""
AI Photo Recognizer - Pydantic Schemas
"""

from .analysis import (
    AnalysisResponse,
    BatchRequest,
    BatchUploadResponse,
    FolderAnalysisRequest,
    JobStatusResponse,
)
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
    "BatchRequest",
    "BatchUploadResponse",
    "FolderAnalysisRequest",
    "JobStatusResponse",
    "Token",
    "TokenData",
    "UserCreate",
    "UserResponse",
    "UserStats",
    "UserUpdate",
]
