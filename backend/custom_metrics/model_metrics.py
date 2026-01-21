from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from backend.custom_metrics._common import (
    DEFAULT_THRESHOLD,
    MetricContext,
    is_classified_as_ai,
)

if TYPE_CHECKING:
    from PIL import Image


METRIC_INFO: dict[str, str] = {
    "name": "model_metrics",
    "description": "ML classification metrics",
    "category": "ml_evaluation",
}


def calculate(
    image: "Image.Image",  # Not used, but required for consistent interface
    score: float,
    context: MetricContext | None = None,
) -> dict[str, Any]:
    threshold = context.threshold if context else DEFAULT_THRESHOLD

    is_ai_predicted = is_classified_as_ai(score, threshold)
    prediction = "ai" if is_ai_predicted else "nature"

    result: dict[str, Any] = {
        "prediction": prediction,
        "threshold_used": round(threshold, 4),
        "raw_score": round(score, 4),
    }

    if context and context.has_ground_truth:
        is_ai_actual = context.is_ai_ground_truth
        is_correct = is_ai_predicted == is_ai_actual

        error_type = None
        if not is_correct:
            if is_ai_predicted and not is_ai_actual:
                error_type = "false_positive"
            else:
                error_type = "false_negative"

        result.update({
            "ground_truth": "ai" if is_ai_actual else "nature",
            "is_correct": is_correct,
            "error_type": error_type,
            "is_true_positive": is_ai_predicted and is_ai_actual,
            "is_true_negative": not is_ai_predicted and not is_ai_actual,
            "is_false_positive": is_ai_predicted and not is_ai_actual,
            "is_false_negative": not is_ai_predicted and is_ai_actual,
        })
    else:
        result.update({
            "ground_truth": None,
            "is_correct": None,
            "error_type": None,
        })

    if context and context.has_batch_data:
        batch_metrics = _calculate_batch_metrics(
            context.batch_labels,
            context.batch_scores,
            threshold,
        )
        result["batch_metrics"] = batch_metrics
    else:
        result["batch_metrics"] = None

    return result


def _calculate_batch_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float,
) -> dict[str, Any]:
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    predictions = (scores_arr >= threshold).astype(int)

    n_samples = len(labels)
    n_ai = int(labels_arr.sum())
    n_nature = n_samples - n_ai

    tp = int(np.sum((predictions == 1) & (labels_arr == 1)))
    tn = int(np.sum((predictions == 0) & (labels_arr == 0)))
    fp = int(np.sum((predictions == 1) & (labels_arr == 0)))
    fn = int(np.sum((predictions == 0) & (labels_arr == 1)))

    accuracy = (tp + tn) / n_samples if n_samples > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "n_samples": n_samples,
        "n_ai": n_ai,
        "n_nature": n_nature,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
    }