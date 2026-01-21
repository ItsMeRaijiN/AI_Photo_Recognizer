"""
AI Photo Recognizer - Analysis Schemas
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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

    @computed_field
    @property
    def has_heatmap(self) -> bool:
        return bool(self.heatmap_path)

    @computed_field
    @property
    def model_used(self) -> str:
        return f"{self.model_type}_{self.backbone_name}"


class BatchRequest(BaseModel):
    path: str = Field(description="Folder path to scan")
    recursive: bool = Field(default=False, description="Scan subfolders")


class FolderAnalysisRequest(BaseModel):
    path: str = Field(description="Absolute path to folder or file")
    recursive: bool = Field(default=False, description="Scan subfolders")
    max_images: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum images to process",
    )


class BatchUploadResponse(BaseModel):
    total: int = Field(ge=0)
    processed: int = Field(ge=0)
    failed: int = Field(ge=0)
    results: list[AnalysisResponse]
    errors: list[str]
    total_inference_time_ms: float = Field(ge=0.0)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    total: int = Field(ge=0)
    processed: int = Field(ge=0)
    progress_percent: float = Field(ge=0.0, le=100.0)
    current_file: str | None = None
    errors: list[str] = Field(default_factory=list)
    results: list[AnalysisResponse] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
