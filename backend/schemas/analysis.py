"""
AI Photo Recognizer - Analysis Schemas
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    filename: str
    file_path: str

    is_ai: bool
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    threshold_used: float = Field(ge=0.0, le=1.0)
    inference_time_ms: float = Field(ge=0.0)

    model_type: str
    backbone_name: str
    model_version: str | None = None

    heatmap_path: str | None = None

    custom_metrics: dict[str, Any] | None = None

    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @computed_field
    @property
    def has_heatmap(self) -> bool:
        return bool(self.heatmap_path)

    @computed_field
    @property
    def model_used(self) -> str:
        return f"{self.model_type}_{self.backbone_name}"


class BatchUploadResponse(BaseModel):
    total: int = Field(ge=0)
    processed: int = Field(ge=0)
    failed: int = Field(ge=0)
    results: list[AnalysisResponse]
    errors: list[str]
    total_inference_time_ms: float = Field(ge=0.0)


