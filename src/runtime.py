# experiment/plateau_cliff/src/runtime.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import ExperimentConfig


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_parallel(model: nn.Module, cfg: ExperimentConfig) -> nn.Module:
    """Wrap a model with DataParallel when the server profile has multiple GPUs."""
    if cfg.use_data_parallel and cfg.device.type == "cuda":
        return nn.DataParallel(model, device_ids=cfg.gpu_ids)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when DataParallel is active."""
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(model: nn.Module, path: Path, metadata: dict[str, Any] | None = None) -> None:
    """Save model parameters and lightweight metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": unwrap_model(model).state_dict(), "metadata": metadata or {}}
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint saved by save_checkpoint, with legacy state-dict support."""
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"])
        return dict(payload.get("metadata", {}))
    model.load_state_dict(payload)
    return {}
