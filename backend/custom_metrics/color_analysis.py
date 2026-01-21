from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from backend.custom_metrics._common import (
    DEFAULT_THRESHOLD,
    MetricContext,
    is_classified_as_ai,
    pil_to_hsv,
    pil_to_rgb_array,
    resize_for_processing,
)


METRIC_INFO: dict[str, str] = {
    "name": "color_analysis",
    "description": "Analyzes color distribution, saturation patterns, and gradient smoothness",
    "category": "color",
}

SATURATION_THRESHOLDS: dict[str, float] = {
    "very_low": 40,
    "low": 80,
    "normal": 130,
    "high": 180,
}

OVERSATURATION_MEAN: float = 150
OVERSATURATION_MAX: float = 250
SMOOTH_GRADIENT_THRESHOLD: float = 0.1
MIN_COLOR_CLUSTERS: int = 3
LOW_SATURATION_STD: float = 30


def calculate(
    image: Image.Image,
    score: float,
    context: MetricContext | None = None,
) -> dict[str, Any]:
    """Analyze color characteristics."""
    threshold = context.threshold if context else DEFAULT_THRESHOLD
    image = resize_for_processing(image)

    rgb_array = pil_to_rgb_array(image)
    hsv = pil_to_hsv(image)

    saturation = hsv[:, :, 1].astype(np.float64)
    sat_mean = float(np.mean(saturation))
    sat_std = float(np.std(saturation))
    sat_max = float(np.percentile(saturation, 99))

    saturation_level = _categorize_saturation(sat_mean)
    color_clusters, dominant_colors = _analyze_color_diversity(rgb_array)
    gradient_smoothness = _calculate_gradient_smoothness(hsv)

    is_oversaturated = sat_mean > OVERSATURATION_MEAN or sat_max > OVERSATURATION_MAX
    is_ai = is_classified_as_ai(score, threshold)

    is_suspicious = _detect_suspicious_pattern(
        is_oversaturated, gradient_smoothness, color_clusters, sat_std, is_ai
    )

    return {
        "saturation_mean": round(sat_mean, 1),
        "saturation_std": round(sat_std, 1),
        "saturation_level": saturation_level,
        "color_clusters": color_clusters,
        "dominant_colors": dominant_colors,
        "gradient_smoothness": round(gradient_smoothness, 4),
        "is_oversaturated": is_oversaturated,
        "is_suspicious": is_suspicious,
    }


def _categorize_saturation(sat_mean: float) -> str:
    if sat_mean < SATURATION_THRESHOLDS["very_low"]:
        return "very_low"
    elif sat_mean < SATURATION_THRESHOLDS["low"]:
        return "low"
    elif sat_mean < SATURATION_THRESHOLDS["normal"]:
        return "normal"
    elif sat_mean < SATURATION_THRESHOLDS["high"]:
        return "high"
    else:
        return "very_high"


def _analyze_color_diversity(
    rgb_array: np.ndarray,
    n_clusters: int = 5,
    sample_size: int = 10000,
) -> tuple[int, list[dict[str, Any]]]:
    pixels = rgb_array.reshape(-1, 3).astype(np.float32)

    if len(pixels) < n_clusters:
        return 1, [{"hex": "#000000", "percentage": 100.0}]

    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[indices]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    _, counts = np.unique(labels, return_counts=True)
    significant_clusters = sum(c > len(labels) * 0.05 for c in counts)

    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors = []
    for idx in sorted_indices[:3]:
        color = centers[idx].astype(int)
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        dominant_colors.append({
            "hex": hex_color,
            "percentage": round(counts[idx] / len(labels) * 100, 1),
        })

    return int(significant_clusters), dominant_colors


def _calculate_gradient_smoothness(hsv: np.ndarray) -> float:
    hue = hsv[:, :, 0]
    gradient_x = cv2.Sobel(hue, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(hue, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
    return float(1.0 / (np.std(gradient_mag) + 1))


def _detect_suspicious_pattern(
    is_oversaturated: bool,
    gradient_smoothness: float,
    color_clusters: int,
    sat_std: float,
    is_ai: bool,
) -> bool:
    return (
        (is_oversaturated and is_ai)
        or (gradient_smoothness > SMOOTH_GRADIENT_THRESHOLD and is_ai)
        or (color_clusters < MIN_COLOR_CLUSTERS and sat_std < LOW_SATURATION_STD)
    )
