from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .config import (
    DEFAULT_LABEL_MAP,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VALID_IMAGE_EXTENSIONS,
)
from .utils import HashCache, get_dataloader_kwargs, hamming_distance

if TYPE_CHECKING:
    from .config import Config

def find_generators(
    root_dir: str,
    label_map: dict[str, int] | None = None
) -> list[str]:
    """Finds valid generator folders in root_dir."""
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Data directory not found: {root_dir}")

    if label_map is None:
        label_map = DEFAULT_LABEL_MAP

    generators: list[str] = []
    with os.scandir(root_path) as it:
        for entry in sorted(it, key=lambda e: e.name):
            if not entry.is_dir():
                continue

            train_path = Path(entry.path) / 'train'
            if not train_path.is_dir():
                continue

            for folder in label_map.keys():
                if (train_path / folder).is_dir():
                    generators.append(entry.name)
                    break

    return generators


def scan_generator(
    root_dir: str,
    generator: str,
    split: str,
    label_map: dict[str, int] | None = None
) -> list[tuple[str, int]]:
    """Scans a single generator and returns list of (relative_path, label)."""
    if label_map is None:
        label_map = DEFAULT_LABEL_MAP

    base_path = Path(root_dir) / generator / split
    samples: list[tuple[str, int]] = []

    for folder_name, label in label_map.items():
        folder_path = base_path / folder_name
        if not folder_path.is_dir():
            continue

        for root, _, files in os.walk(folder_path):
            files.sort()
            for fname in files:
                if Path(fname).suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    full_path = Path(root) / fname
                    rel_path = str(full_path.relative_to(root_dir))
                    samples.append((rel_path, label))

    return sorted(samples)


def scan_dataset(
    root_dir: str,
    generators: list[str] | None,
    split: str,
    verbose: bool = True,
    label_map: dict[str, int] | None = None
) -> list[tuple[str, int]]:
    """Scans dataset for selected generators."""
    if label_map is None:
        label_map = DEFAULT_LABEL_MAP

    available = find_generators(root_dir, label_map)
    if not available:
        raise ValueError(f"No valid generators found in {root_dir}")

    if generators:
        available_lower = {g.lower(): g for g in available}
        selected: list[str] = []
        for g in generators:
            match = available_lower.get(g.lower())
            if match:
                selected.append(match)
            else:
                raise ValueError(f"Generator not found: '{g}'. Available: {available}")
        generators = selected
    else:
        generators = available

    if verbose:
        print(f"\n   Scanning {split}: {generators}")

    all_samples: list[tuple[str, int]] = []
    for gen in sorted(generators):
        samples = scan_generator(root_dir, gen, split, label_map)
        if verbose:
            ai_count = sum(1 for _, lbl in samples if lbl == 1)
            nature_count = len(samples) - ai_count
            print(f"      {gen}: {len(samples):,} (AI: {ai_count:,}, Nature: {nature_count:,})")
        all_samples.extend(samples)

    if verbose:
        print(f"      Total: {len(all_samples):,}")

    return sorted(all_samples)


def check_leakage(
    train_samples: list[tuple[str, int]],
    val_samples: list[tuple[str, int]],
    root_dir: str,
    seed: int,
    use_perceptual: bool = False,
    perceptual_threshold: int = 10,
    cache_path: str = "runs/hash_cache.pkl",
    perceptual_sample_size: int = 5000,
    examples_to_show: int = 3
) -> tuple[int, int]:
    """
    Checks for data leakage.
    Returns: (exact_overlap_paths_count, similar_hits_count)
    """
    print("\n" + "=" * 50)
    print("LEAKAGE CHECK")
    print("=" * 50)

    rng = random.Random(seed)
    cache = HashCache(cache_path)

    def get_hashes(samples: list[tuple[str, int]], name: str) -> dict[str, list[str]]:
        hashes: dict[str, list[str]] = defaultdict(list)
        for rp, _ in tqdm(samples, desc=f"   Hashing {name}", leave=False):
            full_path = os.path.join(root_dir, rp)
            h = cache.get_md5(full_path)
            if h:
                hashes[h].append(rp)
        return hashes

    train_hashes = get_hashes(train_samples, "train")
    val_hashes = get_hashes(val_samples, "val")
    cache.save()

    exact_overlap_hashes = set(train_hashes.keys()) & set(val_hashes.keys())
    exact_overlap_paths_count = sum(len(val_hashes[h]) for h in exact_overlap_hashes)

    print(f"\n   Train unique hashes: {len(train_hashes):,}")
    print(f"   Val unique hashes:   {len(val_hashes):,}")

    if exact_overlap_hashes:
        pct = exact_overlap_paths_count / len(val_samples) * 100
        print(f"\nEXACT DUPLICATES:")
        print(f"      Hashes involved: {len(exact_overlap_hashes)}")
        print(f"      Validation images affected: {exact_overlap_paths_count} ({pct:.2f}%)")

        shown = 0
        for h in exact_overlap_hashes:
            if shown >= examples_to_show:
                break
            print(f"      Hash: {h}")
            print(f"      Train paths: {train_hashes[h]}")
            print(f"      Val paths:   {val_hashes[h]}")
            print()
            shown += 1
    else:
        print("\nNo exact duplicates found")

    similar_hits = 0

    if use_perceptual:
        print("\n   Checking perceptual similarity...")
        val_phashes: dict[str, list[str]] = defaultdict(list)

        val_sample_list = val_samples
        if len(val_samples) > perceptual_sample_size:
            val_sample_list = rng.sample(val_samples, perceptual_sample_size)

        for rel_path, _ in tqdm(val_sample_list, desc="   Perceptual hash (val sample)", leave=False):
            ph = cache.get_phash(os.path.join(root_dir, rel_path))
            if ph:
                val_phashes[ph].append(rel_path)

        stopped_early = False
        train_sample = train_samples
        if len(train_samples) > perceptual_sample_size:
            train_sample = rng.sample(train_samples, perceptual_sample_size)

        actual_sample_size = len(val_sample_list)

        for rel_path, _ in tqdm(train_sample, desc="   Checking similarity", leave=False):
            ph = cache.get_phash(os.path.join(root_dir, rel_path))
            if not ph:
                continue

            for val_ph, val_paths_list in val_phashes.items():
                dist = hamming_distance(ph, val_ph)
                if dist < perceptual_threshold:
                    similar_hits += 1
                    if similar_hits <= examples_to_show:
                        print(f"      Similar (dist={dist}):")
                        print(f"      Train: {rel_path}")
                        print(f"      Val:   {val_paths_list[0]}")
                    break

            if similar_hits > examples_to_show + 50:
                stopped_early = True
                break

        cache.save()

        if similar_hits > 0:
            msg = f"> {similar_hits}" if stopped_early else f"{similar_hits}"
            print(f"\nSIMILAR IMAGES (in random sample of {actual_sample_size}): {msg}")
            if stopped_early:
                print("      (Stopped early due to high number of matches)")
        else:
            print("   ✓ No similar images found in sample")

    # --- Final Verdict ---
    print("\n   --- Verdict ---")
    if exact_overlap_paths_count == 0:
        print("   ✓ Exact duplicates: PASS")
    else:
        exact_leak_ratio = exact_overlap_paths_count / len(val_samples)
        if exact_leak_ratio > 0.05:
            print(f"CRITICAL: High exact leakage ({exact_leak_ratio*100:.1f}%)")
        else:
            print(f"WARNING: Some exact leakage ({exact_leak_ratio*100:.1f}%)")

    return exact_overlap_paths_count, similar_hits


# =============================================================================
# AUGMENTATIONS
# =============================================================================

class JPEGCompression:
    """Simulates JPEG compression artifacts."""

    def __init__(self, quality_min: int = 70, quality_max: int = 95, p: float = 0.3):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=random.randint(self.quality_min, self.quality_max))
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class GaussianNoise:
    """Adds Gaussian noise to tensor."""

    def __init__(self, std: float = 0.02, p: float = 0.2):
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std


class RandomBlur:
    """Applies Gaussian blur to PIL image."""

    def __init__(self, kernel_size: int = 3, p: float = 0.1):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        from PIL import ImageFilter
        return img.filter(ImageFilter.GaussianBlur(radius=self.kernel_size // 2))


def get_train_transforms(cfg: "Config") -> transforms.Compose:
    """Creates training transforms with augmentation."""
    ops: list = [
        transforms.RandomCrop(cfg.image_size, pad_if_needed=True, padding_mode='constant', fill=0),
        transforms.RandomHorizontalFlip(p=cfg.aug_hflip_prob),
        transforms.ColorJitter(
            brightness=cfg.aug_color_jitter,
            contrast=cfg.aug_color_jitter,
            saturation=cfg.aug_color_jitter * 0.5,
            hue=0.02
        ),
    ]

    if cfg.aug_jpeg_prob > 0:
        ops.append(JPEGCompression(quality_min=cfg.aug_jpeg_quality_min, p=cfg.aug_jpeg_prob))

    if cfg.aug_blur_prob > 0:
        ops.append(RandomBlur(kernel_size=cfg.aug_blur_kernel, p=cfg.aug_blur_prob))

    ops.append(transforms.ToTensor())

    if cfg.aug_random_erasing_prob > 0:
        ops.append(transforms.RandomErasing(
            p=cfg.aug_random_erasing_prob,
            scale=(0.02, 0.10),
            value='random'
        ))

    ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    if cfg.aug_gaussian_noise_prob > 0:
        ops.append(GaussianNoise(std=cfg.aug_gaussian_noise_std, p=cfg.aug_gaussian_noise_prob))

    return transforms.Compose(ops)


def get_val_transforms(cfg: "Config") -> transforms.Compose:
    """Creates validation transforms."""
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class GenImageDataset(Dataset):
    """Dataset for GenImage-style data structure."""

    def __init__(
        self,
        samples: list[tuple[str, int]],
        root_dir: str,
        transform: transforms.Compose | None = None
    ):
        self.samples = samples
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str] | None:
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)

        try:
            with Image.open(img_path) as img:
                img.load()
                image = img.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, torch.tensor(label, dtype=torch.float32), rel_path
        except (OSError, UnidentifiedImageError, ValueError):
            return None

    def get_labels(self) -> np.ndarray:
        return np.array([label for _, label in self.samples])


def collate_fn(
    batch: list,
    image_size: int = 224
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Custom collate function that filters out None samples.
    """
    valid = [item for item in batch if item is not None]
    if not valid:
        return torch.empty((0, 3, image_size, image_size)), torch.empty((0,)), []

    images = torch.stack([item[0] for item in valid])
    labels = torch.stack([item[1] for item in valid])
    paths = [item[2] for item in valid]
    return images, labels, paths


def worker_init_fn(worker_id: int) -> None:
    """
    Initializes random state for each DataLoader worker.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        base_seed = worker_info.seed
        seed = (base_seed + worker_id) % (2**32)
    else:
        seed = (torch.initial_seed() + worker_id) % (2**32)

    np.random.seed(seed)
    random.seed(seed)


def prepare_dataloaders(
    cfg: "Config",
    cached_samples: tuple[list[tuple[str, int]], list[tuple[str, int]]] | None = None
) -> tuple[DataLoader, DataLoader]:
    """
    Prepares train and validation DataLoaders.
    """
    print(f"\n{'='*50}\nLOADING DATASET\n{'='*50}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cached_samples is not None:
        train_samples, val_samples = cached_samples
        print(f"   Using cached samples: {len(train_samples):,} train, {len(val_samples):,} val")
    else:
        train_gens = cfg.train_generators
        val_gens = cfg.val_generators if cfg.val_generators else train_gens
        label_map = getattr(cfg, 'label_folders', DEFAULT_LABEL_MAP)

        train_samples = scan_dataset(cfg.data_dir, train_gens, "train", cfg.verbose, label_map)
        val_samples = scan_dataset(cfg.data_dir, val_gens, "val", cfg.verbose, label_map)

        if not cfg.skip_leakage_check and train_gens == val_gens:
            check_leakage(
                train_samples, val_samples, cfg.data_dir,
                seed=cfg.seed,
                use_perceptual=cfg.use_perceptual_hash,
                perceptual_threshold=cfg.perceptual_hash_threshold,
                cache_path=cfg.hash_cache_path,
                perceptual_sample_size=cfg.perceptual_sample_size,
                examples_to_show=cfg.leakage_examples_to_show
            )

    train_ds = GenImageDataset(train_samples, cfg.data_dir, get_train_transforms(cfg))
    val_ds = GenImageDataset(val_samples, cfg.data_dir, get_val_transforms(cfg))

    loader_kwargs = get_dataloader_kwargs(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        collate_fn_base=collate_fn,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(train_ds, shuffle=True, generator=g, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    print(f"\n   Train: {len(train_ds):,} samples, {len(train_loader):,} batches")
    print(f"   Val:   {len(val_ds):,} samples, {len(val_loader):,} batches")
    print("=" * 50)

    return train_loader, val_loader


def prescan_samples(cfg: "Config") -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """
    Pre-scans dataset samples without creating DataLoaders.
    """
    print(f"\n{'='*50}\nPRE-SCANNING DATASET\n{'='*50}")

    train_gens = cfg.train_generators
    val_gens = cfg.val_generators if cfg.val_generators else train_gens
    label_map = getattr(cfg, 'label_folders', DEFAULT_LABEL_MAP)

    train_samples = scan_dataset(cfg.data_dir, train_gens, "train", cfg.verbose, label_map)
    val_samples = scan_dataset(cfg.data_dir, val_gens, "val", cfg.verbose, label_map)

    if not cfg.skip_leakage_check and train_gens == val_gens:
        check_leakage(
            train_samples, val_samples, cfg.data_dir,
            seed=cfg.seed,
            use_perceptual=cfg.use_perceptual_hash,
            perceptual_threshold=cfg.perceptual_hash_threshold,
            cache_path=cfg.hash_cache_path,
            perceptual_sample_size=cfg.perceptual_sample_size,
            examples_to_show=cfg.leakage_examples_to_show
        )

    print(f"\n   Pre-scanned: {len(train_samples):,} train, {len(val_samples):,} val samples")
    print("=" * 50)

    return train_samples, val_samples