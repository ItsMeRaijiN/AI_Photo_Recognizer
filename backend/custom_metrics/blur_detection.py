from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from backend.custom_metrics._common import (
    DEFAULT_THRESHOLD,
    MetricContext,
    is_classified_as_ai,
    pil_to_grayscale,
    resize_for_processing,
)


METRIC_INFO: dict[str, str] = {
    "name": "blur_detection",
    "description": "Analyzes image sharpness using Laplacian variance and Sobel gradients",
    "category": "image_quality",
}

BLUR_THRESHOLDS: dict[str, float] = {
    "very_blurry": 50,
    "blurry": 200,
    "normal": 500,
    "sharp": 1500,
}

SUSPICIOUS_LOW_SHARPNESS: float = 80
SUSPICIOUS_HIGH_SHARPNESS: float = 1200
EXTREMELY_BLURRY: float = 50


def calculate(
    image: Image.Image,
    score: float,
    context: MetricContext | None = None,
) -> dict[str, Any]:
    """Detect blur level using multiple methods."""
    threshold = context.threshold if context else DEFAULT_THRESHOLD
    image = resize_for_processing(image)
    gray = pil_to_grayscale(image)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = float(laplacian.var())

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_sharpness = float(sobel_mag.mean())

    blur_level = _categorize_blur(laplacian_var)
    sharpness_percentile = _estimate_percentile(laplacian_var)

    is_ai = is_classified_as_ai(score, threshold)
    is_suspicious = _detect_suspicious_pattern(laplacian_var, is_ai)

    return {
        "laplacian_variance": round(laplacian_var, 2),
        "sobel_sharpness": round(sobel_sharpness, 2),
        "blur_level": blur_level,
        "sharpness_percentile": sharpness_percentile,
        "is_suspicious": is_suspicious,
    }


def _categorize_blur(laplacian_var: float) -> str:
    if laplacian_var < BLUR_THRESHOLDS["very_blurry"]:
        return "very_blurry"
    elif laplacian_var < BLUR_THRESHOLDS["blurry"]:
        return "blurry"
    elif laplacian_var < BLUR_THRESHOLDS["normal"]:
        return "normal"
    elif laplacian_var < BLUR_THRESHOLDS["sharp"]:
        return "sharp"
    else:
        return "very_sharp"


def _estimate_percentile(laplacian_var: float) -> int:
    if laplacian_var <= 0:
        return 0
    log_val = np.log10(laplacian_var + 1)
    percentile = int(np.clip((log_val - 1.0) / 2.5 * 100, 0, 100))
    return percentile


def _detect_suspicious_pattern(laplacian_var: float, is_ai: bool) -> bool:
    return (
        (laplacian_var < SUSPICIOUS_LOW_SHARPNESS and is_ai)
        or (laplacian_var > SUSPICIOUS_HIGH_SHARPNESS and is_ai)
        or (laplacian_var < EXTREMELY_BLURRY)
    )
