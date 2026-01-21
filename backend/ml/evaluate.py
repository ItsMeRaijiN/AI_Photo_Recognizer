from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import (
    collate_fn,
    find_generators,
    GenImageDataset,
    get_val_transforms,
    scan_generator,
)
from .engine import get_predictions
from .model import get_model
from .utils import calculate_metrics, get_dataloader_kwargs, get_device, validate_threshold


def evaluate_generator(
    model: torch.nn.Module,
    root_dir: str,
    generator: str,
    device: torch.device,
    threshold: float,
    config: Config,
    batch_size: int = 32,
    num_workers: int = 4
) -> dict[str, Any] | None:
    """
    Evaluates model on a single generator using shared inference logic.
    """
    samples = scan_generator(root_dir, generator, "val")
    if not samples:
        return None

    transform = get_val_transforms(config)
    dataset = GenImageDataset(samples, root_dir, transform)

    loader_kwargs = get_dataloader_kwargs(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=config.image_size,
        collate_fn_base=collate_fn,
        device=device,
        persistent_workers=False,
    )

    loader = DataLoader(dataset, shuffle=False, **loader_kwargs)

    y_true, y_pred, _ = get_predictions(model, loader, device)

    if len(y_true) == 0:
        return None

    metrics = calculate_metrics(y_true, y_pred, threshold)
    metrics['n_samples'] = len(y_true)
    metrics['n_ai'] = int(y_true.sum())
    metrics['n_nature'] = len(y_true) - int(y_true.sum())

    return metrics


def main() -> None:
    """Main entry point for cross-generator evaluation."""
    parser = argparse.ArgumentParser(
        description="AI Photo Recognizer - Cross-Generator Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (default: use from checkpoint)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results (default: same as model)")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset not found: {args.data_dir}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    device = get_device()

    print(f"\nLoading model: {args.model}")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)

    ckpt_config_dict = ckpt.get('config', {})

    try:
        ckpt_config_dict['data_dir'] = args.data_dir
        config = Config(**ckpt_config_dict)
    except Exception as e:
        print(f"Warning: Could not fully reconstruct Config from checkpoint: {e}")
        print("Falling back to default config with overrides.")
        config = Config(data_dir=args.data_dir, image_size=ckpt_config_dict.get('image_size', 224))

    backbone = config.backbone
    dropout = config.dropout
    trained_on = config.train_generators

    raw_threshold = args.threshold if args.threshold is not None else ckpt.get('best_threshold', 0.5)
    threshold = validate_threshold(raw_threshold)

    print(f"   Backbone: {backbone}")
    print(f"   Threshold: {threshold:.4f}")
    if trained_on:
        print(f"   Trained on: {trained_on}")

    model = get_model(backbone, pretrained=False, dropout=dropout).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    generators = find_generators(args.data_dir)
    print(f"\nFound {len(generators)} generators: {generators}")

    if not generators:
        print("Error: No generators found!")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("CROSS-GENERATOR EVALUATION")
    print(f"{'='*70}")

    results: dict[str, dict[str, Any]] = {}
    for gen in generators:
        metrics = evaluate_generator(
            model, args.data_dir, gen, device, threshold,
            config,
            args.batch_size, args.num_workers
        )
        if metrics:
            results[gen] = metrics

    if not results:
        print("Error: No results!")
        sys.exit(1)

    print(f"\n{'Generator':<15} {'Samples':>8} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'Trained'}")
    print("-" * 80)

    for gen in sorted(results.keys()):
        m = results[gen]
        is_seen = trained_on and gen in trained_on
        marker = "  *" if is_seen else ""

        auc_str = f"{m['auc']*100:.1f}%" if m['auc'] else "N/A"
        sens_str = f"{m['sensitivity']*100:.1f}%" if m['sensitivity'] is not None else "N/A"
        spec_str = f"{m['specificity']*100:.1f}%" if m['specificity'] is not None else "N/A"

        print(f"{gen:<15} {m['n_samples']:>8} "
              f"{m['accuracy']*100:>7.1f}% {m['f1']*100:>7.1f}% "
              f"{auc_str:>8} {sens_str:>8} {spec_str:>8}{marker}")

    print("-" * 80)

    if trained_on:
        seen_results = {g: m for g, m in results.items() if g in trained_on}
        unseen_results = {g: m for g, m in results.items() if g not in trained_on}

        if seen_results:
            avg_acc = np.mean([m['accuracy'] for m in seen_results.values()])
            avg_f1 = np.mean([m['f1'] for m in seen_results.values()])
            auc_vals = [m['auc'] for m in seen_results.values() if m['auc']]
            avg_auc = np.mean(auc_vals) if auc_vals else None
            auc_str = f"{avg_auc*100:.1f}%" if avg_auc else "N/A"
            n_samples = sum(m['n_samples'] for m in seen_results.values())
            print(f"{'SEEN AVG':<15} {n_samples:>8} {avg_acc*100:>7.1f}% {avg_f1*100:>7.1f}% {auc_str:>8}")

        if unseen_results:
            avg_acc = np.mean([m['accuracy'] for m in unseen_results.values()])
            avg_f1 = np.mean([m['f1'] for m in unseen_results.values()])
            auc_vals = [m['auc'] for m in unseen_results.values() if m['auc']]
            avg_auc = np.mean(auc_vals) if auc_vals else None
            auc_str = f"{avg_auc*100:.1f}%" if avg_auc else "N/A"
            n_samples = sum(m['n_samples'] for m in unseen_results.values())
            print(f"{'UNSEEN AVG':<15} {n_samples:>8} {avg_acc*100:>7.1f}% {avg_f1*100:>7.1f}% {auc_str:>8}")

    avg_acc = np.mean([m['accuracy'] for m in results.values()])
    avg_f1 = np.mean([m['f1'] for m in results.values()])
    auc_vals = [m['auc'] for m in results.values() if m['auc']]
    avg_auc = np.mean(auc_vals) if auc_vals else None
    auc_str = f"{avg_auc*100:.1f}%" if avg_auc else "N/A"
    n_samples = sum(m['n_samples'] for m in results.values())

    print(f"{'OVERALL AVG':<15} {n_samples:>8} {avg_acc*100:>7.1f}% {avg_f1*100:>7.1f}% {auc_str:>8}")
    print(f"\n* = seen during training")

    output_dir = args.output if args.output else os.path.dirname(args.model)
    os.makedirs(output_dir, exist_ok=True)

    output_data: dict[str, Any] = {
        'threshold': threshold,
        'trained_on': trained_on,
        'generators': {},
        'summary': {
            'overall_accuracy': avg_acc,
            'overall_f1': avg_f1,
            'overall_auc': avg_auc,
            'total_samples': n_samples,
        }
    }

    for gen, m in results.items():
        output_data['generators'][gen] = {
            'accuracy': m['accuracy'],
            'f1': m['f1'],
            'auc': m['auc'],
            'sensitivity': m['sensitivity'],
            'specificity': m['specificity'],
            'n_samples': m['n_samples'],
            'n_ai': m['n_ai'],
            'n_nature': m['n_nature'],
            'seen': trained_on and gen in trained_on,
        }

    results_path = os.path.join(output_dir, "cross_generator_eval.json")
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()