from __future__ import annotations

import hashlib
import os
import pickle
import random
from functools import partial
from typing import Any, Callable

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    FILE_HASH_CHUNK_SIZE,
    MAX_HAMMING_DISTANCE,
    PHASH_SIZE,
    PLOT_DPI,
)

def calculate_file_hash(path: str) -> str | None:
    """Calculates MD5 hash of a file"""
    try:
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(FILE_HASH_CHUNK_SIZE), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except (OSError, ValueError):
        return None


def calculate_perceptual_hash(path: str) -> str | None:
    """Calculates perceptual hash using average hash algorithm"""
    try:
        with Image.open(path) as img:
            img = img.convert('L').resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS)
            pixels = np.array(img).flatten()
            avg = pixels.mean()
            bits = ''.join('1' if p > avg else '0' for p in pixels)
            return hex(int(bits, 2))[2:].zfill(16)
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculates Hamming distance between two hex hashes"""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return MAX_HAMMING_DISTANCE

    try:
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        xor = int1 ^ int2
        return bin(xor).count('1')
    except ValueError:
        return MAX_HAMMING_DISTANCE


def load_image_rgb(path: str) -> np.ndarray | None:
    """Loads image as RGB numpy array. Returns None if corrupted"""
    try:
        with Image.open(path) as img:
            img.load()
            return np.array(img.convert('RGB'))
    except (OSError, UnidentifiedImageError, ValueError):
        return None


class HashCache:
    """
    Persistent hash cache for faster leak detection
    """
    def __init__(self, cache_path: str = "runs/hash_cache.pkl"):
        self.cache_path = cache_path
        self.cache: dict[str, dict[str, str]] = {'md5': {}, 'phash': {}}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        if 'md5' not in data:
                            self.cache = {'md5': data, 'phash': {}}
                        else:
                            self.cache = data
            except (OSError, pickle.UnpicklingError, EOFError):
                pass

    def save(self) -> None:
        """
        Saves cache to disk atomically.
        """
        if not self._dirty:
            return

        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp_path = self.cache_path + '.tmp'

        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(self.cache, f)
            os.replace(tmp_path, self.cache_path)
            self._dirty = False
        except OSError:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise

    def get_md5(self, path: str) -> str | None:
        abs_path = os.path.abspath(path)
        if abs_path in self.cache['md5']:
            return self.cache['md5'][abs_path]

        h = calculate_file_hash(abs_path)
        if h:
            self.cache['md5'][abs_path] = h
            self._dirty = True
        return h

    def get_phash(self, path: str) -> str | None:
        abs_path = os.path.abspath(path)
        if abs_path in self.cache['phash']:
            return self.cache['phash'][abs_path]

        h = calculate_perceptual_hash(abs_path)
        if h:
            self.cache['phash'][abs_path] = h
            self._dirty = True
        return h


def get_dataloader_kwargs(
    batch_size: int,
    num_workers: int,
    image_size: int,
    collate_fn_base: Callable,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    worker_init_fn: Callable | None = None,
) -> dict[str, Any]:
    """
    Creates standardized DataLoader kwargs
    """
    if pin_memory is None:
        if device is not None:
            use_pin = device.type in ('cuda', 'mps')
        else:
            use_pin = torch.cuda.is_available() or torch.backends.mps.is_available()
    else:
        use_pin = pin_memory

    if persistent_workers is None:
        use_persistent = num_workers > 0
    else:
        use_persistent = persistent_workers

    kwargs: dict[str, Any] = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': use_pin,
        'collate_fn': partial(collate_fn_base, image_size=image_size),
    }

    if use_persistent:
        kwargs['persistent_workers'] = True

    if num_workers > 0:
        kwargs['prefetch_factor'] = 2

    if worker_init_fn is not None:
        kwargs['worker_init_fn'] = worker_init_fn

    return kwargs


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class BCEWithLogitsLossSmoothed(nn.Module):
    """BCE Loss with optional label smoothing."""
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(inputs, targets)


def get_loss_function(
    use_focal: bool = True,
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.0
) -> nn.Module:
    """Factory for loss functions."""
    if use_focal:
        return FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
    return BCEWithLogitsLossSmoothed(label_smoothing=label_smoothing)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> dict[str, Any]:
    """Calculates comprehensive classification metrics."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel()
    y_pred_binary = (y_pred >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])

    sensitivity: float | None = None
    specificity: float | None = None
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc: float | None = None
    if len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            pass

    return {
        "cm": cm,
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": threshold,
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "f1"
) -> tuple[float, float]:
    """Finds optimal classification threshold."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel()

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        j_scores = tpr - fpr
        idx = np.argmax(j_scores)
        return float(thresholds[idx]), float(j_scores[idx])

    else:
        prec, rec, thresholds = precision_recall_curve(y_true, y_pred)

        with np.errstate(divide='ignore', invalid='ignore'):
            denom = prec[:-1] + rec[:-1]
            f1s = np.where(
                denom > 0,
                2 * (prec[:-1] * rec[:-1]) / denom,
                0.0
            )

        valid = (thresholds >= 0.01) & (thresholds <= 0.99)
        if not valid.any():
            return 0.5, 0.0

        f1s_valid = np.where(valid, f1s, -1)
        idx = int(np.argmax(f1s_valid))
        return float(thresholds[idx]), float(f1s[idx])


def validate_threshold(threshold: float | None) -> float:
    """Validates and clamps threshold to [0, 1] range."""
    if threshold is None:
        return 0.5
    return max(0.0, min(1.0, float(threshold)))


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = 0.0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self) -> None:
        self.counter = 0
        self.best_score = 0.0
        self.should_stop = False


def save_training_curves(history: dict[str, list], output_dir: str) -> None:
    """Saves training curves plot."""
    if not history.get('train_loss'):
        return

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train', linewidth=2)
    ax.plot(history['val_loss'], label='Validation', linewidth=2)
    ax.set_title('Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history['val_f1'], label='F1', color='green', linewidth=2)
    ax.set_title('F1 Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1')
    ax.set_ylim((0, 1.05))
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if history.get('val_auc'):
        auc_clean = [v if v is not None else np.nan for v in history['val_auc']]
        ax.plot(auc_clean, label='AUC', color='purple', linewidth=2)
    ax.set_title('AUC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_ylim((0, 1.05))
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if history.get('lr'):
        ax.plot(history['lr'], color='orange', linewidth=2)
        ax.set_yscale('log')
    ax.set_title('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


def save_confusion_matrix(
    cm: np.ndarray,
    output_dir: str,
    threshold: float,
    class_names: list[str] | None = None
) -> None:
    """Saves confusion matrix plot."""
    os.makedirs(output_dir, exist_ok=True)
    labels = class_names if class_names else ['Nature (0)', 'AI (1)']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels, cbar=False)
    ax.set_title(f'Confusion Matrix (counts)\nThreshold: {threshold:.3f}', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    ax = axes[1]
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels, cbar=False)
    ax.set_title('Confusion Matrix (normalized)', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


def save_roc_pr_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    threshold: float
) -> None:
    """Saves ROC and Precision-Recall curves."""
    os.makedirs(output_dir, exist_ok=True)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(np.unique(y_true)) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')

    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tpr_point = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_point = fp / (fp + tn) if (fp + tn) > 0 else 0
        ax.scatter([fpr_point], [tpr_point], color='red', s=100, zorder=5,
                   label=f'Threshold={threshold:.3f}')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    ax = axes[1]
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    ax.plot(rec, prec, linewidth=2, label='PR Curve')
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve', fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_pr_curves.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


def save_threshold_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    best_threshold: float
) -> None:
    """Saves threshold analysis plot."""
    os.makedirs(output_dir, exist_ok=True)
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel()

    if len(np.unique(y_true)) < 2:
        return

    thresholds = np.linspace(0.01, 0.99, 99)
    metrics_data: dict[str, list[float]] = {
        'f1': [], 'accuracy': [], 'sensitivity': [], 'specificity': []
    }

    for t in thresholds:
        m = calculate_metrics(y_true, y_pred, t)
        for key in metrics_data:
            val = m.get(key)
            metrics_data[key].append(val if val is not None else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, metrics_data['f1'], label='F1', linewidth=2)
    ax.plot(thresholds, metrics_data['accuracy'], label='Accuracy', linewidth=2)
    ax.plot(thresholds, metrics_data['sensitivity'], label='Sensitivity', linewidth=2)
    ax.plot(thresholds, metrics_data['specificity'], label='Specificity', linewidth=2)

    ax.axvline(x=best_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Selected ({best_threshold:.3f})')

    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Threshold Analysis', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1.05))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


def save_all_plots(
    history: dict[str, list],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
    threshold: float,
    output_dir: str,
    class_names: list[str] | None = None
) -> None:
    """Saves all visualization plots."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    save_training_curves(history, plots_dir)
    save_confusion_matrix(cm, plots_dir, threshold, class_names)
    save_roc_pr_curves(y_true, y_pred, plots_dir, threshold)
    save_threshold_analysis(y_true, y_pred, plots_dir, threshold)
    print(f"   Plots saved to {plots_dir}")


def export_model(
    model: nn.Module,
    input_size: int,
    output_dir: str,
    device: torch.device,
    export_onnx: bool = True,
    onnx_opset: int = 18
) -> None:
    """Exports model to PyTorch and optionally ONNX formats."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, "weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"   Saved PyTorch weights: {weights_path}")

    if export_onnx:
        try:
            onnx_path = os.path.join(output_dir, "model.onnx")
            model_cpu = model.cpu()
            dummy = torch.randn(1, 3, input_size, input_size)

            torch.onnx.export(
                model_cpu,
                dummy,
                onnx_path,
                opset_version=onnx_opset,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            )
            print(f"   Saved ONNX model: {onnx_path}")
            model.to(device)
        except Exception as e:
            print(f"   ONNX export failed: {e}")
            model.to(device)


class TensorBoardWriter:
    """Wrapper for TensorBoard"""
    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                print("   TensorBoard not available, logging disabled")
                self.enabled = False

    def add_scalar(self, tag: str, value: float | None, step: int) -> None:
        if self.enabled and self.writer and value is not None:
            self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float | None], step: int) -> None:
        if self.enabled and self.writer:
            filtered = {k: v for k, v in tag_scalar_dict.items() if v is not None}
            if filtered:
                self.writer.add_scalars(main_tag, filtered, step)

    def close(self) -> None:
        if self.writer:
            self.writer.close()


def get_device() -> torch.device:
    """Detects and returns the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Counts total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable