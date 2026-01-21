from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights

from .config import DEFAULT_BACKBONES

try:
    import timm
    HAS_TIMM = True
except ImportError:
    timm = None
    HAS_TIMM = False

def _create_convnext(pretrained: bool, dropout: float) -> nn.Module:
    if not HAS_TIMM:
        raise ImportError(
            "ConvNeXt requires `timm` library. Install with: pip install timm\n"
            "Or use --backbone effnetv2 which doesn't require timm."
        )

    model = timm.create_model(
        'convnextv2_tiny.fcmae_ft_in22k_in1k',
        pretrained=pretrained,
        num_classes=1,
        drop_rate=dropout
    )

    return model


def _create_effnetv2(pretrained: bool, dropout: float) -> nn.Module:
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1)
    )
    return model

BACKBONE_REGISTRY: dict[str, tuple[Callable[[bool, float], nn.Module], str]] = {
    "convnext": (_create_convnext, "head"),
    "effnetv2": (_create_effnetv2, "classifier"),
}

assert set(BACKBONE_REGISTRY.keys()) == DEFAULT_BACKBONES, (
    f"BACKBONE_REGISTRY keys {set(BACKBONE_REGISTRY.keys())} "
    f"must match DEFAULT_BACKBONES {DEFAULT_BACKBONES}"
)


def register_backbone(
    name: str,
    creator_fn: Callable[[bool, float], nn.Module],
    head_attr: str
) -> None:
    """
    Registers a new backbone type.

    Example:
        def _create_resnet50(pretrained, dropout):
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, 1))
            return model

        register_backbone("resnet50", _create_resnet50, "fc")
    """
    BACKBONE_REGISTRY[name] = (creator_fn, head_attr)


def get_available_backbones() -> list[str]:
    return list(BACKBONE_REGISTRY.keys())


def get_model(
    backbone: str = "convnext",
    pretrained: bool = True,
    dropout: float = 0.3,
    local_weights_path: str | None = None
) -> nn.Module:
    if backbone not in BACKBONE_REGISTRY:
        available = get_available_backbones()
        raise ValueError(f"Unknown backbone: {backbone}. Available: {available}")

    creator_fn, _ = BACKBONE_REGISTRY[backbone]
    use_pretrained = pretrained and local_weights_path is None

    model = creator_fn(use_pretrained, dropout)

    if local_weights_path:
        _load_local_weights(model, local_weights_path)

    return model


def _load_local_weights(model: nn.Module, path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights not found: {path}")

    state_dict = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(state_dict, dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

    model_dict = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    if len(filtered) < len(model_dict) * 0.5:
        print(f"   WARNING: Only {len(filtered)}/{len(model_dict)} layers matched from {path}")

    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    print(f"   Loaded local weights: {path} ({len(filtered)} layers)")


def _get_classifier_head(model: nn.Module, backbone: str) -> nn.Module:
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {backbone}")

    _, head_attr = BACKBONE_REGISTRY[backbone]
    return getattr(model, head_attr)


def freeze_backbone(model: nn.Module, backbone: str) -> int:
    for param in model.parameters():
        param.requires_grad = False

    head = _get_classifier_head(model, backbone)

    for param in head.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    print(f"   Backbone frozen: {frozen:,} params frozen, {trainable:,} trainable")

    return trainable


def unfreeze_all(model: nn.Module) -> int:
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters())
    print(f"   All layers unfrozen: {trainable:,} trainable parameters")

    return trainable


def get_parameter_groups(
    model: nn.Module,
    backbone: str,
    base_lr: float,
    backbone_lr_factor: float = 0.1,
    weight_decay: float = 1e-5
) -> list[dict[str, Any]]:
    head = _get_classifier_head(model, backbone)
    head_params = set(head.parameters())

    backbone_params: list[nn.Parameter] = []
    classifier_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param in head_params:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    param_groups: list[dict[str, Any]] = []

    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': base_lr * backbone_lr_factor,
            'weight_decay': weight_decay,
            'name': 'backbone'
        })

    if classifier_params:
        param_groups.append({
            'params': classifier_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'classifier'
        })

    print(f"   Parameter groups: backbone ({len(backbone_params)} tensors, lr={base_lr * backbone_lr_factor:.2e}), "
          f"classifier ({len(classifier_params)} tensors, lr={base_lr:.2e})")

    return param_groups


def peek_checkpoint(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {path}: {e}")

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")

    metadata: dict[str, Any] = {
        'config': checkpoint.get('config', {}),
        'best_threshold': checkpoint.get('best_threshold'),
        'best_f1': checkpoint.get('best_f1'),
        'best_auc': checkpoint.get('best_auc'),
        'metrics': checkpoint.get('metrics', {}),
        'stage': checkpoint.get('stage'),
        'epoch_in_stage': checkpoint.get('epoch_in_stage'),
        'total_epochs': checkpoint.get('total_epochs'),
        'history': checkpoint.get('history', {}),
    }

    if 'backbone' not in metadata['config'] and 'model_state_dict' in checkpoint:
        metadata['inferred_backbone'] = _infer_backbone_from_state_dict(
            checkpoint['model_state_dict']
        )

    return metadata


def _infer_backbone_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str | None:
    keys = set(state_dict.keys())

    if any('stages.' in k for k in keys) and any('stem.' in k for k in keys):
        return "convnext"

    if any('features.' in k for k in keys) and any('classifier.' in k for k in keys):
        return "effnetv2"

    return None


def validate_checkpoint_compatibility(
    checkpoint_path: str,
    target_backbone: str
) -> tuple[bool, str, str | None]:
    try:
        metadata = peek_checkpoint(checkpoint_path)
    except (FileNotFoundError, RuntimeError) as e:
        return False, str(e), None

    actual_backbone = metadata['config'].get('backbone')
    if actual_backbone is None:
        actual_backbone = metadata.get('inferred_backbone')

    if actual_backbone is None:
        return False, "Could not determine backbone from checkpoint", None

    if actual_backbone != target_backbone:
        return False, (
            f"Backbone mismatch: checkpoint was trained with '{actual_backbone}', "
            f"but trying to load into '{target_backbone}'"
        ), actual_backbone

    return True, "Compatible", actual_backbone


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    config_dict: dict[str, Any],
    metrics: dict[str, Any],
    threshold: float,
    path: str,
    stage: int,
    epoch_in_stage: int,
    history: dict[str, Any]
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'config': config_dict,
        'best_threshold': threshold,
        'best_f1': metrics.get('f1', 0.0),
        'best_auc': metrics.get('auc'),
        'metrics': metrics,
        'stage': stage,
        'epoch_in_stage': epoch_in_stage,
        'total_epochs': len(history.get('train_loss', [])),
        'history': history,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    strict: bool = True
) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    except RuntimeError as e:
        if strict:
            ckpt_config = checkpoint.get('config', {})
            ckpt_backbone = ckpt_config.get('backbone', 'unknown')
            raise RuntimeError(
                f"Failed to load checkpoint into model. "
                f"Checkpoint backbone: '{ckpt_backbone}'. "
                f"Make sure you're using the same backbone architecture.\n"
                f"Original error: {e}"
            )
        else:
            model_dict = model.state_dict()
            ckpt_dict = checkpoint['model_state_dict']
            filtered = {
                k: v for k, v in ckpt_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(filtered)
            model.load_state_dict(model_dict)
            print(f"   Warning: Loaded {len(filtered)}/{len(model_dict)} layers (non-strict mode)")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"   Warning: Could not load optimizer state: {e}")

    if scaler and checkpoint.get('scaler_state_dict'):
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            print(f"   Warning: Could not load scaler state: {e}")

    return checkpoint