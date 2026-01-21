from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from .config import Config, parse_args
from .dataset import find_generators, prepare_dataloaders, prescan_samples
from .engine import evaluate_with_threshold, train_one_epoch, validate
from .model import (
    freeze_backbone,
    get_model,
    get_parameter_groups,
    load_checkpoint,
    peek_checkpoint,
    save_checkpoint,
    unfreeze_all,
)
from .utils import (
    EarlyStopping,
    TensorBoardWriter,
    count_parameters,
    export_model,
    get_device,
    get_loss_function,
    save_all_plots,
    set_seed,
)

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False

def get_current_lr(optimizer: optim.Optimizer) -> float:
    """Gets learning rate from last param group."""
    return optimizer.param_groups[-1]['lr']


def _apply_checkpoint_config(cfg: Config, checkpoint_path: str) -> Config:
    """
    Creates new config with values from checkpoint for resume compatibility.
    """
    print(f"\n   Peeking at checkpoint to verify compatibility...")

    try:
        metadata = peek_checkpoint(checkpoint_path)
    except (FileNotFoundError, RuntimeError) as e:
        raise ValueError(f"Cannot resume: {e}")

    ckpt_config = metadata.get('config', {})
    ckpt_backbone = ckpt_config.get('backbone')

    if ckpt_backbone is None:
        ckpt_backbone = metadata.get('inferred_backbone')
        if ckpt_backbone:
            print(f"   Inferred backbone from weights: {ckpt_backbone}")

    if ckpt_backbone is None:
        raise ValueError(
            "Cannot determine backbone from checkpoint. "
            "The checkpoint may be corrupted or from an incompatible version."
        )

    overrides: dict[str, Any] = {}

    if cfg.backbone != ckpt_backbone:
        print(f"WARNING: CLI backbone '{cfg.backbone}' differs from checkpoint '{ckpt_backbone}'")
        print(f"   -> Using checkpoint backbone: {ckpt_backbone}")
        overrides['backbone'] = ckpt_backbone

    ckpt_dropout = ckpt_config.get('dropout')
    if ckpt_dropout is not None and cfg.dropout != ckpt_dropout:
        print(f"   -> Using checkpoint dropout: {ckpt_dropout}")
        overrides['dropout'] = ckpt_dropout

    ckpt_image_size = ckpt_config.get('image_size')
    if ckpt_image_size is not None and cfg.image_size != ckpt_image_size:
        print(f"   -> Using checkpoint image_size: {ckpt_image_size}")
        overrides['image_size'] = ckpt_image_size

    if overrides:
        new_cfg = cfg.copy_with_overrides(**overrides)
    else:
        new_cfg = cfg

    print(f"Checkpoint compatible: backbone={new_cfg.backbone}, dropout={new_cfg.dropout}")

    return new_cfg


def _train_stage(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    cfg: Config,
    history: dict[str, list],
    tb_writer: TensorBoardWriter,
    checkpoint_path: str,
    stage: int,
    num_epochs: int,
    start_epoch: int,
    early_stopping: EarlyStopping,
    best_f1: float,
    best_threshold: float,
    best_epoch: int,
    freeze_bn: bool = False,
    scheduler: ReduceLROnPlateau | None = None,
    warmup_scheduler: LinearLR | None = None,
    warmup_epochs: int = 0,
    trial: Any | None = None,
) -> tuple[float, float, int]:

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, cfg.grad_accum_steps, cfg.grad_clip_norm,
            freeze_bn=freeze_bn
        )

        val_loss, f1, metrics, thresh, y_true, y_pred = validate(
            model, val_loader, criterion, device
        )

        lr = get_current_lr(optimizer)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(f1)
        history['val_auc'].append(metrics['auc'])
        history['lr'].append(lr)

        global_step = len(history['train_loss'])
        tb_writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, global_step)
        tb_writer.add_scalar('F1/val', f1, global_step)
        if stage == 2:
            tb_writer.add_scalar('LR', lr, global_step)
        if metrics['auc']:
            tb_writer.add_scalar('AUC/val', metrics['auc'], global_step)

        if scheduler is not None:
            if warmup_scheduler is not None and epoch <= warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step(f1)

        if trial and HAS_OPTUNA:
            trial.report(f1, global_step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] else "N/A"
        lr_str = f" | LR: {lr:.2e}" if stage == 2 else ""
        print(f"   [S{stage} {epoch}/{num_epochs}] "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"F1: {f1:.4f} | AUC: {auc_str}{lr_str}")

        if f1 > best_f1:
            best_f1, best_threshold, best_epoch = f1, thresh, global_step

            if not trial:
                save_checkpoint(
                    model, optimizer, scaler, cfg.to_dict(), metrics, thresh,
                    checkpoint_path, stage=stage, epoch_in_stage=epoch, history=history
                )
                print(f"New best F1: {f1:.4f} (threshold: {thresh:.4f})")

        if early_stopping(f1):
            print(f"\n   Early stopping at epoch {epoch}")
            break

    return best_f1, best_threshold, best_epoch


def train(
    cfg: Config,
    trial: Any | None = None,
    cached_samples: tuple[list, list] | None = None
) -> float:
    """
    Main training function.
    """
    if cfg.resume:
        if not os.path.exists(cfg.resume):
            raise ValueError(f"Resume checkpoint not found: {cfg.resume}")

        cfg = _apply_checkpoint_config(cfg, cfg.resume)

    cfg.validate()

    device = get_device()
    set_seed(cfg.seed)

    if trial:
        output_dir = os.path.join(cfg.output_dir, f"trial_{trial.number}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(cfg.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    if not trial:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)

    train_loader, val_loader = prepare_dataloaders(cfg, cached_samples=cached_samples)

    model = get_model(
        cfg.backbone,
        pretrained=True,
        dropout=cfg.dropout,
        local_weights_path=cfg.local_weights_path
    ).to(device)

    total_params, _ = count_parameters(model)
    print(f"\nModel: {cfg.backbone}, {total_params:,} total parameters")

    criterion = get_loss_function(
        cfg.use_focal_loss,
        alpha=cfg.focal_alpha,
        gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing
    ).to(device)

    scaler: torch.amp.GradScaler | None = None
    if cfg.use_amp and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("   Mixed precision training enabled")
    elif cfg.use_amp and device.type == 'mps':
        print("   Note: AMP GradScaler not supported on MPS, using standard precision")

    tb_writer = TensorBoardWriter(
        os.path.join(output_dir, "tensorboard"),
        enabled=cfg.use_tensorboard and not trial
    )

    history: dict[str, list] = {
        'train_loss': [], 'val_loss': [],
        'val_f1': [], 'val_auc': [],
        'lr': []
    }
    best_f1, best_threshold, best_epoch = 0.0, 0.5, 0
    overall_best_f1, overall_best_threshold, overall_best_epoch = 0.0, 0.5, 0
    start_stage, start_epoch = 1, 1
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    stage1_checkpoint_path = os.path.join(output_dir, "best_model_stage1.pt")

    if cfg.resume:
        print(f"\n{'='*50}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*50}")
        print(f"   Path: {cfg.resume}")

        ckpt = load_checkpoint(cfg.resume, model, device)
        best_f1 = ckpt.get('best_f1', 0.0)
        best_threshold = ckpt.get('best_threshold', 0.5)
        best_epoch = ckpt.get('total_epochs', 0)
        overall_best_f1, overall_best_threshold, overall_best_epoch = best_f1, best_threshold, best_epoch
        start_stage = ckpt.get('stage', 2)
        start_epoch = ckpt.get('epoch_in_stage', 0) + 1
        history = ckpt.get('history', history)

        print(f"   Resuming from Stage {start_stage}, Epoch {start_epoch}")
        print(f"   Best F1 so far: {best_f1:.4f}")

    # stage 1
    stage1_epochs = cfg.optuna_stage1_epochs if trial else cfg.stage1_epochs
    stage1_patience = cfg.optuna_patience if trial else cfg.stage1_patience

    if start_stage == 1 and stage1_epochs > 0:
        print(f"\n{'='*50}")
        print(f"STAGE 1: Frozen Backbone ({stage1_epochs} epochs)")
        print(f"{'='*50}")

        freeze_backbone(model, cfg.backbone)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.stage1_lr,
            weight_decay=cfg.weight_decay
        )

        early_stop_s1 = EarlyStopping(patience=stage1_patience, min_delta=cfg.min_delta)

        best_f1, best_threshold, best_epoch = _train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            history=history,
            tb_writer=tb_writer,
            checkpoint_path=stage1_checkpoint_path if not trial else checkpoint_path,
            stage=1,
            num_epochs=stage1_epochs,
            start_epoch=start_epoch,
            early_stopping=early_stop_s1,
            best_f1=best_f1,
            best_threshold=best_threshold,
            best_epoch=best_epoch,
            freeze_bn=True,
            trial=trial,
        )

        overall_best_f1, overall_best_threshold, overall_best_epoch = best_f1, best_threshold, best_epoch
        start_epoch = 1

    # stage 2
    stage2_epochs = cfg.optuna_epochs if trial else cfg.stage2_epochs
    patience = cfg.optuna_patience if trial else cfg.patience

    if start_stage <= 2 and stage2_epochs > 0:
        print(f"\n{'='*50}")
        print(f"STAGE 2: Fine-tuning ({stage2_epochs} epochs)")
        print(f"{'='*50}")

        stage1_best_f1 = best_f1
        if start_stage == 1:
            print(f"   (Stage 1 best F1: {stage1_best_f1:.4f})")
            best_f1 = 0.0

        unfreeze_all(model)

        param_groups = get_parameter_groups(
            model, cfg.backbone,
            cfg.stage2_lr, cfg.stage2_backbone_lr_factor,
            cfg.weight_decay
        )
        optimizer = optim.AdamW(param_groups)

        warmup_epochs = min(cfg.warmup_epochs, stage2_epochs // 3)
        warmup_scheduler: LinearLR | None = None
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )

        main_scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience, min_lr=cfg.scheduler_min_lr
        )

        early_stop = EarlyStopping(patience=patience, min_delta=cfg.min_delta)
        s2_start = start_epoch if start_stage == 2 else 1

        stage2_best_f1, stage2_best_threshold, stage2_best_epoch = _train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            history=history,
            tb_writer=tb_writer,
            checkpoint_path=checkpoint_path,
            stage=2,
            num_epochs=stage2_epochs,
            start_epoch=s2_start,
            early_stopping=early_stop,
            best_f1=best_f1,
            best_threshold=best_threshold,
            best_epoch=best_epoch,
            freeze_bn=False,
            scheduler=main_scheduler,
            warmup_scheduler=warmup_scheduler,
            warmup_epochs=warmup_epochs,
            trial=trial,
        )

        if stage2_best_f1 > overall_best_f1:
            overall_best_f1, overall_best_threshold, overall_best_epoch = (
                stage2_best_f1, stage2_best_threshold, stage2_best_epoch
            )
            print(f"\nStage 2 improved over Stage 1: {stage2_best_f1:.4f} > {stage1_best_f1:.4f}")
        elif start_stage == 1 and stage2_best_f1 < overall_best_f1:
            # Stage 2 didn't improve - copy Stage 1 checkpoint as best
            print(f"\nStage 2 did not improve over Stage 1 ({stage2_best_f1:.4f} < {overall_best_f1:.4f})")
            if os.path.exists(stage1_checkpoint_path) and not trial:
                shutil.copy(stage1_checkpoint_path, checkpoint_path)
                print(f"Using Stage 1 checkpoint as best model")

    if trial:
        return overall_best_f1


    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")

    if os.path.exists(checkpoint_path):
        ckpt = load_checkpoint(checkpoint_path, model, device)
        overall_best_threshold = ckpt.get('best_threshold', overall_best_threshold)
        overall_best_epoch = ckpt.get('total_epochs', overall_best_epoch)

    # Final evaluation
    _, metrics, final_y_true, final_y_pred = evaluate_with_threshold(
        model, val_loader, criterion, device, overall_best_threshold
    )

    print(f"\n   Best epoch:     {overall_best_epoch}")
    print(f"   Threshold:      {overall_best_threshold:.4f}")
    print(f"   F1:             {metrics['f1']:.4f}")
    print(f"   Accuracy:       {metrics['accuracy']:.4f}")
    print(f"   Precision:      {metrics['precision']:.4f}")
    print(f"   Recall:         {metrics['recall']:.4f}")
    print(f"   AUC:            {metrics['auc']:.4f}" if metrics['auc'] else "   AUC:            N/A")
    print(f"   Sensitivity:    {metrics['sensitivity']:.4f}" if metrics['sensitivity'] else "   Sensitivity:    N/A")
    print(f"   Specificity:    {metrics['specificity']:.4f}" if metrics['specificity'] else "   Specificity:    N/A")
    print(f"\n   Confusion Matrix:")
    print(f"   {metrics['cm']}")

    if cfg.save_plots and final_y_true is not None:
        print("\n   Saving visualizations...")
        save_all_plots(
            history, final_y_true, final_y_pred,
            metrics['cm'], overall_best_threshold, output_dir,
            class_names=cfg.class_names
        )

    print("\n   Exporting model...")
    export_model(
        model, cfg.image_size, output_dir, device,
        export_onnx=cfg.export_onnx, onnx_opset=cfg.onnx_opset
    )

    summary: dict[str, Any] = {
        'backbone': cfg.backbone,
        'best_epoch': overall_best_epoch,
        'best_threshold': overall_best_threshold,
        'f1': metrics['f1'],
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'auc': metrics['auc'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'train_generators': cfg.train_generators,
        'val_generators': cfg.val_generators,
        'seed': cfg.seed,
        'total_epochs': len(history['train_loss']),
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Training Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Backbone: {cfg.backbone}\n")
        f.write(f"Best epoch: {overall_best_epoch}\n")
        f.write(f"Threshold: {overall_best_threshold:.4f}\n\n")
        f.write(f"Metrics:\n")
        f.write(f"  F1:          {metrics['f1']:.4f}\n")
        f.write(f"  Accuracy:    {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision:   {metrics['precision']:.4f}\n")
        f.write(f"  Recall:      {metrics['recall']:.4f}\n")
        f.write(f"  AUC:         {metrics['auc']}\n")
        f.write(f"  Sensitivity: {metrics['sensitivity']}\n")
        f.write(f"  Specificity: {metrics['specificity']}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Train generators: {cfg.train_generators}\n")
        f.write(f"  Val generators:   {cfg.val_generators}\n")
        f.write(f"  Seed: {cfg.seed}\n")
        f.write(f"  Batch size: {cfg.batch_size}\n")
        f.write(f"  Stage 1 LR: {cfg.stage1_lr}\n")
        f.write(f"  Stage 2 LR: {cfg.stage2_lr}\n")

    tb_writer.close()

    if os.path.exists(stage1_checkpoint_path):
        os.remove(stage1_checkpoint_path)

    print(f"\n{'='*50}")
    print(f"Done! Results saved to: {output_dir}")
    print(f"{'='*50}")

    return overall_best_f1


def run_optuna(cfg: Config) -> dict[str, Any]:
    """
    Runs Optuna hyperparameter optimization.
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna not installed. Run: pip install optuna")

    print(f"\n{'='*50}")
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*50}")
    print(f"   Trials: {cfg.optuna_trials}")
    print(f"   Epochs per trial: Stage 1={cfg.optuna_stage1_epochs}, Stage 2={cfg.optuna_epochs}")
    print("\n   Pre-scanning dataset (will be reused for all trials)...")
    cached_samples = prescan_samples(cfg)

    def objective(trial: optuna.Trial) -> float:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trial_cfg = cfg.copy_with_overrides(
            dropout=trial.suggest_float("dropout", cfg.optuna_dropout_min, cfg.optuna_dropout_max),
            batch_size=trial.suggest_categorical("batch_size", cfg.optuna_batch_sizes),
            stage2_lr=trial.suggest_float("lr", cfg.optuna_lr_min, cfg.optuna_lr_max, log=True),
            focal_alpha=trial.suggest_float("focal_alpha", cfg.optuna_focal_alpha_min, cfg.optuna_focal_alpha_max),
            focal_gamma=trial.suggest_float("focal_gamma", cfg.optuna_focal_gamma_min, cfg.optuna_focal_gamma_max),
            aug_jpeg_prob=0.0,
            aug_blur_prob=0.0,
            skip_leakage_check=True,
            use_tensorboard=False,
            save_plots=False,
            export_onnx=False,
            verbose=False,
        )

        return train(trial_cfg, trial=trial, cached_samples=cached_samples)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )

    study.optimize(objective, n_trials=cfg.optuna_trials, show_progress_bar=True)

    print(f"\n{'='*50}")
    print("OPTUNA RESULTS")
    print(f"{'='*50}")
    print(f"   Best F1: {study.best_value:.4f}")
    print(f"   Best parameters:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"      {k}: {v:.6f}")
        else:
            print(f"      {k}: {v}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    results_path = os.path.join(cfg.output_dir, "optuna_results.json")
    with open(results_path, "w") as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
        }, f, indent=2)

    return study.best_params


def main() -> None:
    """Entry point."""
    config = parse_args()

    if config.list_generators:
        try:
            generators = find_generators(config.data_dir)
            print(f"\nAvailable generators in {config.data_dir}:")
            for g in generators:
                print(f"   - {g}")
            print(f"\nTotal: {len(generators)} generators")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        sys.exit(0)

    if config.use_optuna:
        best_params = run_optuna(config)
        print(f"\nTo train with best parameters (remember to enable augmentations):")
        print(f"   python -m ml.train --data_dir {config.data_dir} "
              f"--dropout {best_params['dropout']:.3f} "
              f"--batch_size {best_params['batch_size']} "
              f"--stage2_lr {best_params['lr']:.6f} "
              f"--focal_alpha {best_params['focal_alpha']:.3f} "
              f"--focal_gamma {best_params['focal_gamma']:.3f} "
              f"--aug_jpeg_prob 0.3 --aug_blur_prob 0.1")
    else:
        train(config)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()