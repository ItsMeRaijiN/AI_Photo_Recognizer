from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from backend.custom_metrics._common import (
    DEFAULT_THRESHOLD,
    MAX_PROCESS_SIZE,
    MetricContext,
    is_classified_as_ai,
    pil_to_grayscale,
    resize_for_processing,
)


METRIC_INFO: dict[str, str] = {
    "name": "texture_analysis",
    "description": "Analyzes texture patterns using Local Binary Patterns and entropy",
    "category": "structure",
}

LOW_ENTROPY_THRESHOLD: float = 5.0
HIGH_UNIFORMITY_THRESHOLD: float = 0.05
VERY_LOW_ENTROPY: float = 4.0
VERY_HIGH_ENTROPY: float = 7.5
SUSPICIOUS_UNIFORMITY: float = 0.1


def calculate(
    image: Image.Image,
    score: float,
    context: MetricContext | None = None,
) -> dict[str, Any]:
    threshold = context.threshold if context else DEFAULT_THRESHOLD

    if image.size[0] < 3 or image.size[1] < 3:
        return {
            "entropy": 0.0,
            "uniformity": 1.0,
            "complexity": 0.0,
            "has_repetitive_patterns": False,
            "texture_category": "smooth",
            "is_suspicious": False
        }

    image = resize_for_processing(image, MAX_PROCESS_SIZE)
    gray = pil_to_grayscale(image).astype(np.int16)

    lbp = _calculate_lbp_vectorized(gray)
    entropy, uniformity, complexity = _calculate_texture_metrics(lbp)

    has_repetitive = entropy < LOW_ENTROPY_THRESHOLD and uniformity > HIGH_UNIFORMITY_THRESHOLD
    texture_category = _categorize_texture(entropy, uniformity)

    is_ai = is_classified_as_ai(score, threshold)
    is_suspicious = _detect_suspicious_pattern(entropy, uniformity, is_ai)

    return {
        "entropy": round(entropy, 3),
        "uniformity": round(uniformity, 4),
        "complexity": round(complexity, 2),
        "has_repetitive_patterns": has_repetitive,
        "texture_category": texture_category,
        "is_suspicious": is_suspicious,
    }


def _calculate_lbp_vectorized(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]

    neighbors = [
        gray[0:-2, 0:-2],  # top-left
        gray[0:-2, 1:-1],  # top
        gray[0:-2, 2:],    # top-right
        gray[1:-1, 2:],    # right
        gray[2:, 2:],      # bottom-right
        gray[2:, 1:-1],    # bottom
        gray[2:, 0:-2],    # bottom-left
        gray[1:-1, 0:-2],  # left
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for i, neighbor in enumerate(neighbors):
        lbp |= (neighbor >= center).astype(np.uint8) << (7 - i)

    return lbp


def _calculate_texture_metrics(lbp: np.ndarray) -> tuple[float, float, float]:
    hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)

    h_sum = hist.sum()
    if h_sum > 0:
        hist /= h_sum
    else:
        return 0.0, 1.0, 0.0

    non_zero = hist[hist > 0]
    entropy = float(-np.sum(non_zero * np.log2(non_zero)))
    uniformity = float(np.sum(hist**2))
    complexity = float(np.std(lbp))

    return entropy, uniformity, complexity


def _categorize_texture(entropy: float, uniformity: float) -> str:
    if entropy < 3.5:
        return "smooth"
    elif entropy < 5.0 and uniformity > 0.03:
        return "regular"
    elif entropy < 6.5:
        return "natural"
    elif entropy < 7.5:
        return "complex"
    else:
        return "chaotic"


def _detect_suspicious_pattern(
    entropy: float,
    uniformity: float,
    is_ai: bool,
) -> bool:
    return (
        entropy < VERY_LOW_ENTROPY
        or entropy > VERY_HIGH_ENTROPY
        or (uniformity > SUSPICIOUS_UNIFORMITY and is_ai)
    )