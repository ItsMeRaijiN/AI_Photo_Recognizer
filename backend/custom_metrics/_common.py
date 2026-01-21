from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image


# CONSTANTS
MAX_PROCESS_SIZE: int = 512
DEFAULT_THRESHOLD: float = 0.5
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# METRIC CONTEXT
@dataclass
class MetricContext:
    threshold: float = DEFAULT_THRESHOLD
    is_ai_ground_truth: bool | None = None
    model_backbone: str | None = None
    batch_labels: list[int] | None = None
    batch_scores: list[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_ground_truth(self) -> bool:
        return self.is_ai_ground_truth is not None

    @property
    def has_batch_data(self) -> bool:
        return self.batch_labels is not None and self.batch_scores is not None


# IMAGE CONVERSION HELPERS
def resize_for_processing(
    image: Image.Image,
    max_size: int = MAX_PROCESS_SIZE
) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def pil_to_rgb_array(image: Image.Image) -> NDArray[np.uint8]:
    return np.array(image.convert("RGB"))


def pil_to_grayscale(image: Image.Image) -> NDArray[np.uint8]:
    rgb = pil_to_rgb_array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def pil_to_hsv(image: Image.Image) -> NDArray[np.uint8]:
    rgb = pil_to_rgb_array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)



# CLASSIFICATION HELPERS
def is_classified_as_ai(score: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    return score >= threshold


def get_confidence(score: float, threshold: float = DEFAULT_THRESHOLD) -> float:
    if score >= threshold:
        return 0.5 + (score - threshold) / (2 * (1 - threshold + 1e-10))
    else:
        return 0.5 + (threshold - score) / (2 * (threshold + 1e-10))


def categorize_confidence(confidence: float) -> str:
    if confidence >= 0.95:
        return "very_high"
    elif confidence >= 0.85:
        return "high"
    elif confidence >= 0.70:
        return "medium"
    else:
        return "low"


# NUMPY TYPE CONVERSION
def to_python_types(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {to_python_types(k): to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [to_python_types(item) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj
