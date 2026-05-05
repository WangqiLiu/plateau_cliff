# experiment/plateau_cliff/src/train.py
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .logging_utils import logger
from .models import create_model
from .regularizers import PlateauCliffPatchRegularizer
from .runtime import load_checkpoint, maybe_parallel, save_checkpoint


RegularizerFactory = Callable[[], nn.Module | None]


def fmt_float(value: float) -> str:
    """Format a float for stable checkpoint names."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def checkpoint_path(cfg: ExperimentConfig, model_name: str, tag: str) -> Path:
    return cfg.model_dir / f"{model_name}_{tag}.pt"


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    if was_training:
        model.train()
    return correct / max(total, 1)


def train_or_load_classifier(
    model_name: str,
    tag: str,
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    regularizer_factory: RegularizerFactory,
    force_retrain: bool = False,
) -> nn.Module:
    """Train a model with an optional regularizer, or load a satisfactory checkpoint."""
    raw_model = create_model(model_name, cfg.device)
    path = checkpoint_path(cfg, model_name, tag)

    if cfg.eval_only:
        if force_retrain:
            raise ValueError("eval_only=true conflicts with force_retrain=true. Disable one of them in config.json.")
        if not path.exists():
            raise FileNotFoundError(
                f"eval_only=true requires an existing checkpoint, but it was not found: {path}"
            )
        metadata = load_checkpoint(raw_model, path, cfg.device)
        model = maybe_parallel(raw_model, cfg)
        acc = evaluate_accuracy(model, valid_loader, cfg.device)
        logger.info(
            "Eval-only loaded {} [{}] from {} | valid_acc={:.2%} | metadata={}",
            model_name,
            tag,
            path,
            acc,
            metadata,
        )
        return model

    if path.exists() and not force_retrain:
        metadata = load_checkpoint(raw_model, path, cfg.device)
        model = maybe_parallel(raw_model, cfg)
        acc = evaluate_accuracy(model, valid_loader, cfg.device)
        logger.info("Loaded {} [{}] from {} | valid_acc={:.2%}", model_name, tag, path, acc)
        if acc >= cfg.target_acc:
            if cfg.stop_at_target:
                logger.info(
                    "Using stored target-threshold checkpoint. target={:.2%}, actual_valid_acc={:.2%}, metadata={}",
                    cfg.target_acc,
                    acc,
                    metadata,
                )
            return model
        logger.info("Checkpoint metadata: {}", metadata)
        logger.info("Continuing training because valid_acc is below target {:.2%}", cfg.target_acc)
    else:
        model = maybe_parallel(raw_model, cfg)
        logger.info("Training {} [{}] from scratch", model_name, tag)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    regularizer = regularizer_factory()

    best_acc = evaluate_accuracy(model, valid_loader, cfg.device)
    saved_target_checkpoint = False
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_reg = 0.0
        pbar = tqdm(train_loader, desc=f"{model_name}:{tag}:epoch{epoch}", leave=False, ncols=100)
        for x, y in pbar:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            if regularizer is not None:
                x.requires_grad_(True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss_ce = criterion(logits, y)
            loss_reg = logits.new_tensor(0.0)
            if regularizer is not None:
                loss_reg = regularizer(x, logits, y)
            loss = loss_ce + loss_reg
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_reg += float(loss_reg.detach().cpu())
            pbar.set_postfix(loss=f"{loss_ce.item():.3f}", reg=f"{loss_reg.item():.3e}")

        valid_acc = evaluate_accuracy(model, valid_loader, cfg.device)
        scheduler.step(valid_acc)
        if valid_acc > best_acc:
            best_acc = valid_acc
            if not cfg.stop_at_target:
                save_checkpoint(
                    model,
                    path,
                    metadata={
                        "model_name": model_name,
                        "tag": tag,
                        "valid_acc": valid_acc,
                        "epoch": epoch,
                        "selection": "best_valid_acc",
                        "target_acc": cfg.target_acc,
                    },
                )
        logger.info(
            "{} [{}] epoch {}/{} | valid_acc={:.2%} | best={:.2%} | loss={:.4f} | reg={:.4e}",
            model_name,
            tag,
            epoch,
            cfg.max_epochs,
            valid_acc,
            best_acc,
            total_loss / max(len(train_loader), 1),
            total_reg / max(len(train_loader), 1),
        )

        if cfg.stop_at_target and epoch >= cfg.min_epochs and valid_acc >= cfg.target_acc:
            save_checkpoint(
                model,
                path,
                metadata={
                    "model_name": model_name,
                    "tag": tag,
                    "valid_acc": valid_acc,
                    "epoch": epoch,
                    "selection": "first_reaches_target",
                    "target_acc": cfg.target_acc,
                },
            )
            saved_target_checkpoint = True
            logger.info(
                "{} [{}] reached the shared clean-accuracy target: {:.2%} >= {:.2%}",
                model_name,
                tag,
                valid_acc,
                cfg.target_acc,
            )
            break

    if (cfg.stop_at_target and not saved_target_checkpoint) or not cfg.stop_at_target:
        save_checkpoint(
            model,
            path,
            metadata={
                "model_name": model_name,
                "tag": tag,
                "valid_acc": best_acc,
                "selection": "best_available",
                "target_acc": cfg.target_acc,
            },
        )
    return model


def train_plateau_cliff_patch(
    model_name: str,
    source_tag: str,
    target_tag: str,
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    force_retrain: bool = False,
) -> nn.Module:
    """Fine-tune a Cross-Lip model with the Plateau-Cliff patch regularizer."""
    source_path = checkpoint_path(cfg, model_name, source_tag)
    target_path = checkpoint_path(cfg, model_name, target_tag)
    raw_model = create_model(model_name, cfg.device)

    if target_path.exists() and not force_retrain:
        load_checkpoint(raw_model, target_path, cfg.device)
        model = maybe_parallel(raw_model, cfg)
        acc = evaluate_accuracy(model, valid_loader, cfg.device)
        logger.info("Loaded patched model {} [{}] | valid_acc={:.2%}", model_name, target_tag, acc)
        return model

    if not source_path.exists():
        raise FileNotFoundError(f"Patch training requires a Cross-Lip checkpoint first: {source_path}")
    load_checkpoint(raw_model, source_path, cfg.device)
    model = maybe_parallel(raw_model, cfg)
    logger.info("Patch fine-tuning {} from {} to {}", model_name, source_tag, target_tag)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.patch_lr, weight_decay=cfg.weight_decay)
    regularizer = PlateauCliffPatchRegularizer(
        plateau_lambda=cfg.plateau_lambda,
        beta=cfg.plateau_beta,
        cliff_reward=cfg.cliff_reward,
        tau=cfg.cliff_tau,
        temperature=cfg.cliff_temperature,
        score_space=cfg.score_space,
    )

    best_acc = evaluate_accuracy(model, valid_loader, cfg.device)
    for epoch in range(1, cfg.patch_epochs + 1):
        model.train()
        total_loss = 0.0
        total_reg = 0.0
        pbar = tqdm(train_loader, desc=f"{model_name}:pcp:epoch{epoch}", leave=False, ncols=100)
        for x, y in pbar:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            x.requires_grad_(True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss_ce = criterion(logits, y)
            loss_reg = regularizer(x, logits, y)
            loss = loss_ce + loss_reg
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_reg += float(loss_reg.detach().cpu())
            pbar.set_postfix(loss=f"{loss_ce.item():.3f}", pcp=f"{loss_reg.item():.3e}")

        valid_acc = evaluate_accuracy(model, valid_loader, cfg.device)
        best_acc = max(best_acc, valid_acc)
        logger.info(
            "{} [{}] patch epoch {}/{} | valid_acc={:.2%} | best={:.2%} | loss={:.4f} | pcp={:.4e} | stats={}",
            model_name,
            target_tag,
            epoch,
            cfg.patch_epochs,
            valid_acc,
            best_acc,
            total_loss / max(len(train_loader), 1),
            total_reg / max(len(train_loader), 1),
            regularizer.last_stats,
        )

    save_checkpoint(model, target_path, metadata={"model_name": model_name, "tag": target_tag, "valid_acc": best_acc})
    return model
