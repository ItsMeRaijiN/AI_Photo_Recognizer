from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import calculate_metrics, find_best_threshold

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    grad_accum_steps: int = 1,
    grad_clip_norm: float = 1.0,
    freeze_bn: bool = False,
) -> float:
    """
    Trains model for one epoch.
    """
    model.train()

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    running_loss = 0.0
    total_samples = 0
    accumulated_steps = 0
    use_amp = scaler is not None

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels, _ in pbar:
        if images.size(0) == 0:
            continue

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_steps += 1

        if accumulated_steps % grad_accum_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        batch_loss = loss.item() * grad_accum_steps
        running_loss += batch_loss * images.size(0)
        total_samples += images.size(0)

        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

    if accumulated_steps % grad_accum_steps != 0:
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / total_samples if total_samples > 0 else 0.0


def _run_inference(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    desc: str = "Inference",
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    """
    Common inference logic for validation/evaluation.
    """
    model.eval()

    running_loss = 0.0
    total_samples = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_paths: list[str] = []

    with torch.inference_mode():
        for images, labels, paths in tqdm(loader, desc=desc, leave=False):
            if images.size(0) == 0:
                continue

            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_preds.append(probs)
            all_labels.append(labels.numpy())
            all_paths.extend(paths)

            if criterion is not None:
                labels_t = labels.to(device, non_blocking=True).unsqueeze(1)
                loss = criterion(outputs, labels_t)
                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

    if not all_preds:
        raise RuntimeError(f"No valid samples in {desc}!")

    y_true = np.concatenate(all_labels).ravel()
    y_pred = np.concatenate(all_preds).ravel()
    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0

    return avg_loss, y_true, y_pred, all_paths


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold_method: str = "f1",
) -> tuple[float, float, dict, float, np.ndarray, np.ndarray]:
    """
    Validates model and finds optimal threshold.
    """
    avg_loss, y_true, y_pred, _ = _run_inference(
        model, loader, criterion, device, desc="Validation"
    )

    best_thresh, _ = find_best_threshold(y_true, y_pred, method=threshold_method)
    metrics = calculate_metrics(y_true, y_pred, best_thresh)

    return avg_loss, metrics["f1"], metrics, best_thresh, y_true, y_pred


def evaluate_with_threshold(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> tuple[float, dict, np.ndarray, np.ndarray]:
    """
    Evaluates model with fixed threshold.
    """
    avg_loss, y_true, y_pred, _ = _run_inference(
        model, loader, criterion, device, desc="Evaluating"
    )

    metrics = calculate_metrics(y_true, y_pred, threshold)

    return avg_loss, metrics, y_true, y_pred


def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Gets predictions for all samples.
    """
    _, y_true, y_pred, paths = _run_inference(
        model, loader, None, device, desc="Predicting"
    )

    return y_true, y_pred, paths