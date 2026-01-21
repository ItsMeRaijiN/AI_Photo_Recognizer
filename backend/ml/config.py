from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, replace
from typing import Any

def _default_num_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(cpu // 2, 8))

DEFAULT_BACKBONES: set[str] = {"convnext", "effnetv2"}
DEFAULT_LABEL_MAP: dict[str, int] = {"ai": 1, "nature": 0}
VALID_IMAGE_EXTENSIONS: set[str] = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# 64KB
FILE_HASH_CHUNK_SIZE: int = 65536

# 8x8
PHASH_SIZE: int = 8
MAX_HAMMING_DISTANCE: int = 64

PLOT_DPI: int = 150


@dataclass
class Config:
    data_dir: str = ""
    output_dir: str = "runs/experiment"
    resume: str | None = None

    train_generators: list[str] | None = None
    val_generators: list[str] | None = None

    backbone: str = "convnext"  # convnext | effnetv2
    image_size: int = 224
    dropout: float = 0.3
    local_weights_path: str | None = None

    # Training Stage 1
    stage1_epochs: int = 5
    stage1_lr: float = 1e-3
    stage1_patience: int = 3

    # Training Stage 2
    stage2_epochs: int = 30
    stage2_lr: float = 1e-4
    stage2_backbone_lr_factor: float = 0.1  # Backbone LR = stage2_lr * factor

    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    warmup_epochs: int = 2

    batch_size: int = 64
    num_workers: int = field(default_factory=_default_num_workers)
    seed: int = 42

    # === Optimization ===
    use_amp: bool = True
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-5

    # === Early Stopping ===
    patience: int = 8
    min_delta: float = 1e-4

    use_optuna: bool = False
    optuna_trials: int = 20
    optuna_epochs: int = 10
    optuna_stage1_epochs: int = 2
    optuna_patience: int = 5

    optuna_dropout_min: float = 0.2
    optuna_dropout_max: float = 0.5
    optuna_batch_sizes: list[int] = field(default_factory=lambda: [32, 64, 128])
    optuna_lr_min: float = 1e-5
    optuna_lr_max: float = 1e-3
    optuna_focal_alpha_min: float = 0.1
    optuna_focal_alpha_max: float = 0.5
    optuna_focal_gamma_min: float = 1.0
    optuna_focal_gamma_max: float = 3.0

    # === Augmentation (torchvision) ===
    aug_hflip_prob: float = 0.5
    aug_color_jitter: float = 0.15
    aug_random_erasing_prob: float = 0.1

    # === Augmentation for generalization ===
    aug_jpeg_quality_min: int = 70
    aug_jpeg_prob: float = 0.3
    aug_gaussian_noise_std: float = 0.02
    aug_gaussian_noise_prob: float = 0.2
    aug_blur_prob: float = 0.1
    aug_blur_kernel: int = 3

    # === Regularization ===
    label_smoothing: float = 0.0  # 0.0 = disabled, 0.1 = typical value

    # === Leak Detection ===
    skip_leakage_check: bool = False
    use_perceptual_hash: bool = False  # Slower but catches similar images
    perceptual_hash_threshold: int = 10
    hash_cache_path: str = "runs/hash_cache.pkl"
    perceptual_sample_size: int = 5000
    leakage_examples_to_show: int = 3

    export_onnx: bool = True
    onnx_opset: int = 18

    save_plots: bool = True
    use_tensorboard: bool = True
    list_generators: bool = False
    verbose: bool = True

    class_names: list[str] = field(default_factory=lambda: ["Nature (0)", "AI (1)"])
    label_folders: dict[str, int] = field(default_factory=lambda: DEFAULT_LABEL_MAP.copy())

    def copy_with_overrides(self, **overrides: Any) -> "Config":
        return replace(self, **overrides)

    def validate(self, available_backbones: set[str] | None = None) -> None:
        errors: list[str] = []

        if not self.data_dir:
            errors.append("--data_dir is required")
        elif not os.path.isdir(self.data_dir):
            errors.append(f"data_dir not found: {self.data_dir}")

        valid_backbones = available_backbones if available_backbones else DEFAULT_BACKBONES
        if self.backbone not in valid_backbones:
            errors.append(f"backbone must be one of {valid_backbones}, got: {self.backbone}")

        if not 0.0 <= self.dropout < 1.0:
            errors.append(f"dropout must be in [0, 1), got: {self.dropout}")

        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got: {self.batch_size}")

        if self.image_size < 32:
            errors.append(f"image_size must be >= 32, got: {self.image_size}")

        if self.grad_accum_steps < 1:
            errors.append(f"grad_accum_steps must be >= 1, got: {self.grad_accum_steps}")

        if self.resume and not os.path.exists(self.resume):
            errors.append(f"resume checkpoint not found: {self.resume}")

        if errors:
            raise ValueError("Configuration errors:\n  - " + "\n  - ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="AI Photo Recognizer - GenImage Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("Paths")
    g.add_argument("--data_dir", type=str, required=True, help="Path to GenImage dataset")
    g.add_argument("--output_dir", type=str, default="runs/experiment", help="Output directory")
    g.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    g = parser.add_argument_group("Cross-Generator")
    g.add_argument("--train_generators", nargs="*", default=None, help="Generators for training")
    g.add_argument("--val_generators", nargs="*", default=None, help="Generators for validation")

    g = parser.add_argument_group("Model")
    g.add_argument(
        "--backbone",
        type=str,
        default="convnext",
        choices=list(DEFAULT_BACKBONES),
        help="convnext (timm, recommended) or effnetv2 (torchvision, lighter)",
    )
    g.add_argument("--image_size", type=int, default=224)
    g.add_argument("--dropout", type=float, default=0.3)
    g.add_argument("--local_weights", type=str, default=None, help="Path to local pretrained weights")

    g = parser.add_argument_group("Training")
    g.add_argument("--stage1_epochs", type=int, default=5)
    g.add_argument("--stage1_lr", type=float, default=1e-3)
    g.add_argument("--stage1_patience", type=int, default=3, help="Early stopping patience for stage 1")
    g.add_argument("--stage2_epochs", type=int, default=30)
    g.add_argument("--stage2_lr", type=float, default=1e-4)
    g.add_argument("--backbone_lr_factor", type=float, default=0.1, help="Backbone LR = stage2_lr * factor")
    g.add_argument("--warmup_epochs", type=int, default=2)
    g.add_argument("--batch_size", type=int, default=64)
    g.add_argument("--num_workers", type=int, default=None)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--patience", type=int, default=8)
    g.add_argument("--grad_accum", type=int, default=1)

    g = parser.add_argument_group("Optimization")
    g.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    g.add_argument("--no_focal", action="store_true", help="Use BCE instead of Focal Loss")
    g.add_argument("--focal_alpha", type=float, default=0.25)
    g.add_argument("--focal_gamma", type=float, default=2.0)
    g.add_argument("--weight_decay", type=float, default=1e-5)
    g.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing (0.1 typical)")

    g = parser.add_argument_group("Augmentation (generalization)")
    g.add_argument("--aug_jpeg_prob", type=float, default=0.3, help="JPEG compression probability")
    g.add_argument("--aug_jpeg_quality_min", type=int, default=70, help="Min JPEG quality")
    g.add_argument("--aug_noise_prob", type=float, default=0.2, help="Gaussian noise probability")
    g.add_argument("--aug_noise_std", type=float, default=0.02, help="Gaussian noise std")
    g.add_argument("--aug_blur_prob", type=float, default=0.1, help="Blur probability")

    g = parser.add_argument_group("Optuna")
    g.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter search")
    g.add_argument("--optuna_trials", type=int, default=20)
    g.add_argument("--optuna_epochs", type=int, default=10, help="Epochs per trial for Stage 2")
    g.add_argument("--optuna_stage1_epochs", type=int, default=2, help="Epochs per trial for Stage 1")
    g.add_argument("--optuna_patience", type=int, default=5, help="Early stopping patience during Optuna")

    g = parser.add_argument_group("Safety")
    g.add_argument("--skip_leakage", action="store_true", help="Skip leakage check")
    g.add_argument("--perceptual_hash", action="store_true", help="Use perceptual hashing (slower)")

    g = parser.add_argument_group("Export")
    g.add_argument("--no_onnx", action="store_true", help="Skip ONNX export")
    g.add_argument("--onnx_opset", type=int, default=18)

    g = parser.add_argument_group("Misc")
    g.add_argument("--list_generators", action="store_true", help="List available generators and exit")
    g.add_argument("--no_tensorboard", action="store_true", help="Disable TensorBoard logging")
    g.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resume=args.resume,
        train_generators=args.train_generators,
        val_generators=args.val_generators,
        backbone=args.backbone,
        image_size=args.image_size,
        dropout=args.dropout,
        local_weights_path=args.local_weights,
        stage1_epochs=args.stage1_epochs,
        stage1_lr=args.stage1_lr,
        stage1_patience=args.stage1_patience,
        stage2_epochs=args.stage2_epochs,
        stage2_lr=args.stage2_lr,
        stage2_backbone_lr_factor=args.backbone_lr_factor,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else _default_num_workers(),
        seed=args.seed,
        patience=args.patience,
        grad_accum_steps=args.grad_accum,
        use_amp=not args.no_amp,
        use_focal_loss=not args.no_focal,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        aug_jpeg_prob=args.aug_jpeg_prob,
        aug_jpeg_quality_min=args.aug_jpeg_quality_min,
        aug_gaussian_noise_prob=args.aug_noise_prob,
        aug_gaussian_noise_std=args.aug_noise_std,
        aug_blur_prob=args.aug_blur_prob,
        use_optuna=args.optuna,
        optuna_trials=args.optuna_trials,
        optuna_epochs=args.optuna_epochs,
        optuna_stage1_epochs=args.optuna_stage1_epochs,
        optuna_patience=args.optuna_patience,
        skip_leakage_check=args.skip_leakage,
        use_perceptual_hash=args.perceptual_hash,
        export_onnx=not args.no_onnx,
        onnx_opset=args.onnx_opset,
        use_tensorboard=not args.no_tensorboard,
        list_generators=args.list_generators,
        verbose=not args.quiet,
    )