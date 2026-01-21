from datetime import datetime

import pytest
from pydantic import ValidationError

from backend.schemas.analysis import AnalysisResponse, FolderAnalysisRequest


class TestAnalysisResponse:
    def test_score_validation_rejects_out_of_range(self):
        base = {
            "id": 1,
            "filename": "f",
            "file_path": "p",
            "is_ai": True,
            "threshold_used": 0.5,
            "inference_time_ms": 10,
            "model_type": "t",
            "backbone_name": "b",
            "created_at": datetime.now(),
        }

        with pytest.raises(ValidationError):
            AnalysisResponse(**{**base, "score": 1.5, "confidence": 0.5})

    def test_confidence_validation_rejects_negative(self):
        base = {
            "id": 1,
            "filename": "f",
            "file_path": "p",
            "is_ai": True,
            "threshold_used": 0.5,
            "inference_time_ms": 10,
            "model_type": "t",
            "backbone_name": "b",
            "created_at": datetime.now(),
        }

        with pytest.raises(ValidationError):
            AnalysisResponse(**{**base, "score": 0.5, "confidence": -0.1})