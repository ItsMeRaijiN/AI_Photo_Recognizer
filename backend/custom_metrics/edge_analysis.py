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
    "name": "edge_analysis",
    "description": "Analyzes edge patterns, density, and haloing artifacts",
    "category": "structure",
}

EDGE_DENSITY_THRESHOLDS: dict[str, float] = {
    "very_low": 1,
    "low": 3,
    "normal": 8,
    "high": 15,
}

HIGH_CONSISTENCY_THRESHOLD: float = 0.85
HALOING_THRESHOLD: float = 0.3
HIGH_SHARPNESS_THRESHOLD: float = 2.0
COMBINED_SHARPNESS_CONSISTENCY: float = 0.8


def calculate(
    image: Image.Image,
    score: float,
    context: MetricContext | None = None,
) -> dict[str, Any]:
    threshold = context.threshold if context else DEFAULT_THRESHOLD

    image = resize_for_processing(image)
    gray = pil_to_grayscale(image)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / edges.size * 100)
    density_level = _categorize_edge_density(edge_density)

    edge_sharpness = _calculate_edge_sharpness(gray)
    edge_consistency = _calculate_edge_consistency(edges)

    haloing_score = _detect_haloing(gray, edges)
    has_haloing = haloing_score > HALOING_THRESHOLD

    is_ai = is_classified_as_ai(score, threshold)
    is_suspicious = _detect_suspicious_pattern(
        edge_consistency, has_haloing, edge_sharpness, is_ai
    )

    return {
        "edge_density": round(edge_density, 2),
        "edge_density_level": density_level,
        "edge_sharpness": round(edge_sharpness, 3),
        "edge_consistency": round(edge_consistency, 3),
        "haloing_score": round(haloing_score, 3),
        "has_haloing": has_haloing,
        "is_suspicious": is_suspicious,
    }


def _categorize_edge_density(edge_density: float) -> str:
    if edge_density < EDGE_DENSITY_THRESHOLDS["very_low"]:
        return "very_low"
    elif edge_density < EDGE_DENSITY_THRESHOLDS["low"]:
        return "low"
    elif edge_density < EDGE_DENSITY_THRESHOLDS["normal"]:
        return "normal"
    elif edge_density < EDGE_DENSITY_THRESHOLDS["high"]:
        return "high"
    else:
        return "very_high"


def _calculate_edge_sharpness(gray: np.ndarray) -> float:
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    p90 = np.percentile(gradient_mag, 90)
    p50 = np.percentile(gradient_mag, 50)

    strong_edges = np.sum(gradient_mag > p90)
    weak_edges = np.sum((gradient_mag > p50) & (gradient_mag <= p90))

    if weak_edges > 0:
        return float(strong_edges / weak_edges)
    return 0.0


def _calculate_edge_consistency(edges: np.ndarray) -> float:
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    edge_coords = np.where(edges > 0)
    if len(edge_coords[0]) < 100:
        return 0.5

    n_samples = min(1000, len(edge_coords[0]))
    sampled_idx = np.random.choice(len(edge_coords[0]), n_samples, replace=False)

    edge_widths = []
    h, w = edges.shape

    for i in sampled_idx:
        y, x = edge_coords[0][i], edge_coords[1][i]
        y1, y2 = max(0, y - 2), min(h, y + 3)
        x1, x2 = max(0, x - 2), min(w, x + 3)
        local_width = np.sum(dilated[y1:y2, x1:x2] > 0)
        edge_widths.append(local_width)

    edge_widths_array = np.array(edge_widths)
    mean_width = np.mean(edge_widths_array)
    if mean_width > 0:
        consistency = 1.0 - min(1.0, np.std(edge_widths_array) / (mean_width + 1e-6))
    else:
        consistency = 0.5

    return float(consistency)


def _detect_haloing(gray: np.ndarray, edges: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_laplacian = laplacian * (edges > 0)

    pos_response = np.sum(edge_laplacian > 50)
    neg_response = np.sum(edge_laplacian < -50)

    edge_count = np.sum(edges > 0)
    if edge_count > 0:
        return min(pos_response, neg_response) / (edge_count / 255 + 1)
    return 0.0


def _detect_suspicious_pattern(
    edge_consistency: float,
    has_haloing: bool,
    edge_sharpness: float,
    is_ai: bool,
) -> bool:
    return (
        (edge_consistency > HIGH_CONSISTENCY_THRESHOLD and is_ai)
        or (has_haloing and is_ai)
        or (edge_sharpness > HIGH_SHARPNESS_THRESHOLD
            and edge_consistency > COMBINED_SHARPNESS_CONSISTENCY)
    )
