from __future__ import annotations

from typing import Any, TYPE_CHECKING

from backend.custom_metrics._common import (
    DEFAULT_THRESHOLD,
    MetricContext,
    categorize_confidence,
    get_confidence,
    is_classified_as_ai,
)

if TYPE_CHECKING:
    from PIL import Image


METRIC_INFO: dict[str, str] = {
    "name": "confidence",
    "description": "Model prediction certainty relative to decision threshold",
    "category": "prediction",
}


def calculate(
    image: "Image.Image",  # Not used but required for consistent interface
    score: float,
    context: MetricContext | None = None,
) -> dict[str, Any]:
    threshold = context.threshold if context else DEFAULT_THRESHOLD

    confidence = get_confidence(score, threshold)
    distance = abs(score - threshold)
    is_ai = is_classified_as_ai(score, threshold)
    level = categorize_confidence(confidence)

    if is_ai:
        max_distance = 1.0 - threshold
        prediction_strength = distance / max_distance if max_distance > 0 else 0
    else:
        max_distance = threshold
        prediction_strength = distance / max_distance if max_distance > 0 else 0

    return {
        "confidence": round(confidence, 4),
        "confidence_level": level,
        "is_ai": is_ai,
        "score_distance": round(distance, 4),
        "raw_score": round(score, 4),
        "threshold_used": round(threshold, 4),
        "prediction_strength": round(prediction_strength, 4),
    }