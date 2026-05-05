# experiment/plateau_cliff/src/data.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import ExperimentConfig


@dataclass
class DataBundle:
    train: DataLoader
    valid: DataLoader
    phys_eval: DataLoader
    adv_eval: DataLoader
    test_dataset: torch.utils.data.Dataset


def _subset(dataset: torch.utils.data.Dataset, size: int | None) -> torch.utils.data.Dataset:
    if size is None or size >= len(dataset):
        return dataset
    return Subset(dataset, range(size))


def _check_mnist_root(cfg: ExperimentConfig) -> None:
    """Fail early with a clear path-specific message when MNIST is absent."""
    raw_dir = cfg.data_dir / "MNIST" / "raw"
    processed_dir = cfg.data_dir / "MNIST" / "processed"
    if cfg.download_data or raw_dir.exists() or processed_dir.exists():
        return
    raise FileNotFoundError(
        "MNIST dataset was not found. "
        f"Configured data_dir is '{cfg.data_dir}'. "
        "Expected torchvision layout under data_dir/MNIST/raw or data_dir/MNIST/processed. "
        "Either set data_dir correctly in config.json, set MNIST_ROOT, or set download_data=true."
    )


def build_data_bundle(cfg: ExperimentConfig) -> DataBundle:
    """Build MNIST or synthetic smoke-test loaders with consistent tensor shape."""
    transform = transforms.Compose([transforms.ToTensor()])
    if cfg.use_fake_data:
        train_ds = datasets.FakeData(
            size=cfg.train_size or 16,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        test_ds = datasets.FakeData(
            size=max(cfg.valid_size or 8, cfg.phys_eval_size or 8, cfg.adv_eval_size or 4),
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
    else:
        _check_mnist_root(cfg)
        train_ds = datasets.MNIST(root=str(cfg.data_dir), train=True, download=cfg.download_data, transform=transform)
        test_ds = datasets.MNIST(root=str(cfg.data_dir), train=False, download=cfg.download_data, transform=transform)

    train_loader = DataLoader(
        _subset(train_ds, cfg.train_size),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.type == "cuda",
    )
    valid_loader = DataLoader(
        _subset(test_ds, cfg.valid_size),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.type == "cuda",
    )
    phys_loader = DataLoader(
        _subset(test_ds, cfg.phys_eval_size),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.type == "cuda",
    )
    adv_loader = DataLoader(
        _subset(test_ds, cfg.adv_eval_size),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.type == "cuda",
    )
    return DataBundle(train=train_loader, valid=valid_loader, phys_eval=phys_loader, adv_eval=adv_loader, test_dataset=test_ds)
