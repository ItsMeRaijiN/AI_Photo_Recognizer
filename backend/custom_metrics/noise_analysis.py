from __future__ import annotations

from typing import Any

import cv2
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
    "name": "noise_analysis",
    "description": "Analyzes noise patterns and compression artifacts",
    "category": "image_quality",
}

BLOCK_SIZE: int = 32
SMOOTH_BLOCK_THRESHOLD: float = 30
MAX_BLOCKS_FOR_NOISE: int = 50
DARK_PIXEL_THRESHOLD: int = 50
MIN_DARK_PIXELS: int = 100
SENSOR_NOISE_THRESHOLD: float = 3.0
COMPRESSION_ARTIFACT_THRESHOLD: float = 0.5
HIGH_UNIFORMITY_THRESHOLD: float = 0.9
LOW_NOISE_THRESHOLD: float = 5


def calculate(
    image: Image.Image,
    score: float,
    context: MetricContext | None = None,
) -> dict[str, Any]:
    threshold = context.threshold if context else DEFAULT_THRESHOLD

    image = resize_for_processing(image, MAX_PROCESS_SIZE)
    gray = pil_to_grayscale(image).astype(np.float64)

    noise_std = _estimate_noise_level(gray)
    noise_uniformity = _calculate_noise_uniformity(gray)
    has_sensor_noise = _detect_sensor_noise(gray)

    compression_score = _detect_compression_artifacts(gray)
    has_jpeg_artifacts = compression_score > COMPRESSION_ARTIFACT_THRESHOLD

    is_ai = is_classified_as_ai(score, threshold)
    is_suspicious = _detect_suspicious_pattern(
        noise_uniformity, noise_std, has_sensor_noise, is_ai
    )

    return {
        "noise_level": round(noise_std, 2),
        "noise_uniformity": round(noise_uniformity, 3),
        "has_sensor_noise": has_sensor_noise,
        "has_jpeg_artifacts": has_jpeg_artifacts,
        "compression_artifact_score": round(compression_score, 3),
        "is_suspicious": is_suspicious,
    }


def _estimate_noise_level(gray: np.ndarray) -> float:
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filtered = cv2.filter2D(gray, -1, kernel)
    return float(np.std(filtered))


def _calculate_noise_uniformity(gray: np.ndarray) -> float:
    h, w = gray.shape
    n_blocks_y = h // BLOCK_SIZE
    n_blocks_x = w // BLOCK_SIZE

    if n_blocks_y == 0 or n_blocks_x == 0:
        return 0.5

    cropped = gray[: n_blocks_y * BLOCK_SIZE, : n_blocks_x * BLOCK_SIZE]
    blocks = cropped.reshape(n_blocks_y, BLOCK_SIZE, n_blocks_x, BLOCK_SIZE)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)

    block_stds = np.std(blocks, axis=(1, 2))
    smooth_mask = block_stds < SMOOTH_BLOCK_THRESHOLD

    if np.sum(smooth_mask) == 0:
        return 0.5

    smooth_blocks = blocks[smooth_mask][:MAX_BLOCKS_FOR_NOISE]

    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    noise_vars = []
    for block in smooth_blocks:
        local_noise = cv2.filter2D(block, -1, kernel)
        noise_vars.append(np.var(local_noise))

    if not noise_vars:
        return 0.5

    noise_vars_array = np.array(noise_vars)
    mean_var = np.mean(noise_vars_array)
    if mean_var > 0:
        uniformity = 1.0 - min(1.0, np.std(noise_vars_array) / (mean_var + 1e-6))
    else:
        uniformity = 0.5

    return float(uniformity)


def _detect_sensor_noise(gray: np.ndarray) -> bool | None:
    dark_mask = gray < DARK_PIXEL_THRESHOLD
    dark_count = np.sum(dark_mask)

    if dark_count < MIN_DARK_PIXELS:
        return None

    dark_pixels = gray[dark_mask]
    dark_noise = float(np.std(dark_pixels))

    return dark_noise > SENSOR_NOISE_THRESHOLD


def _detect_compression_artifacts(gray: np.ndarray) -> float:
    h, w = gray.shape
    sample_size = 256
    sample_h = min(h, sample_size)
    sample_w = min(w, sample_size)

    n_blocks_y = sample_h // 8
    n_blocks_x = sample_w // 8

    if n_blocks_y == 0 or n_blocks_x == 0:
        return 0.0

    cropped = gray[: n_blocks_y * 8, : n_blocks_x * 8]
    blocks = cropped.reshape(n_blocks_y, 8, n_blocks_x, 8)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, 8, 8)

    sample_count = min(100, len(blocks))
    sample_indices = np.linspace(0, len(blocks) - 1, sample_count, dtype=int)
    sample_blocks = blocks[sample_indices]

    hf_components = []
    for block in sample_blocks:
        dct = cv2.dct(block.astype(np.float32))
        hf_components.append(np.abs(dct[7, 7]))

    return float(np.mean(hf_components))


def _detect_suspicious_pattern(
    noise_uniformity: float,
    noise_std: float,
    has_sensor_noise: bool | None,
    is_ai: bool,
) -> bool:
    return (
        (noise_uniformity > HIGH_UNIFORMITY_THRESHOLD and is_ai)
        or (noise_std < LOW_NOISE_THRESHOLD and is_ai)
        or (has_sensor_noise is False and not is_ai)
    )
