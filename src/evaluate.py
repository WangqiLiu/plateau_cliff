# experiment/plateau_cliff/src/evaluate.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import DataBundle
from .logging_utils import logger


ADVERSARIAL_ATTACKS = {"PGD", "AutoPGD", "SquareAttack"}


def apply_physical_perturbation(x: torch.Tensor, attack_type: str, param: Any) -> torch.Tensor:
    """Apply non-adversarial input corruptions used by the benchmark."""
    if attack_type == "Noise":
        return torch.clamp(x + torch.randn_like(x) * float(param), 0.0, 1.0)
    if attack_type == "Blur":
        kernel_size = int(param)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        channels = x.shape[1]
        coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - padding
        sigma = max(float(kernel_size) / 3.0, 1e-3)
        kernel_1d = torch.exp(-(coords.pow(2)) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
        return torch.clamp(F.conv2d(x, kernel_2d, padding=padding, groups=channels), 0.0, 1.0)
    if attack_type == "Resize":
        size = max(1, int(param))
        down = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return torch.clamp(F.interpolate(down, size=(28, 28), mode="bilinear", align_corners=False), 0.0, 1.0)
    raise ValueError(f"Unknown physical perturbation: {attack_type}")


def _project_linf(x_adv: torch.Tensor, x_clean: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x_adv, x_clean + eps), x_clean - eps).clamp(0.0, 1.0)


def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    """White-box L-infinity PGD attack."""
    eps = float(params.get("eps", 0.1))
    steps = int(params.get("steps", 40))
    step_size = float(params.get("step_size", max(eps / max(steps / 3, 1), 1e-4)))
    random_start = bool(params.get("random_start", True))
    x_adv = x.detach().clone()
    if random_start:
        x_adv = _project_linf(x_adv + torch.empty_like(x_adv).uniform_(-eps, eps), x, eps)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv = _project_linf(x_adv.detach() + step_size * grad.sign(), x, eps)
    return x_adv.detach()


def auto_pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    """A lightweight multi-restart PGD approximation used when ART AutoPGD is unavailable."""
    restarts = int(params.get("n_restarts", 1))
    best_adv = x.detach().clone()
    best_loss = torch.full((x.shape[0],), -float("inf"), device=x.device)
    for _ in range(restarts):
        adv = pgd_attack(model, x, y, params)
        with torch.no_grad():
            losses = F.cross_entropy(model(adv), y, reduction="none")
        mask = losses > best_loss
        best_loss[mask] = losses[mask]
        best_adv[mask] = adv[mask]
    return best_adv.detach()


def square_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    """Simple black-box square attack under an L-infinity budget."""
    eps = float(params.get("eps", 0.2))
    max_iter = int(params.get("max_iter", 300))
    adv = x.detach().clone()
    batch, _, height, width = adv.shape
    with torch.no_grad():
        best_loss = F.cross_entropy(model(adv), y, reduction="none")

    for idx in range(max_iter):
        proposal = adv.clone()
        frac = max(0.08, 0.5 * (1.0 - idx / max(max_iter, 1)))
        square = max(1, int(round(min(height, width) * frac)))
        for sample_idx in range(batch):
            top = torch.randint(0, height - square + 1, (1,), device=x.device).item()
            left = torch.randint(0, width - square + 1, (1,), device=x.device).item()
            sign = 1.0 if torch.rand((), device=x.device).item() > 0.5 else -1.0
            proposal[
                sample_idx,
                :,
                top : top + square,
                left : left + square,
            ] = x[sample_idx, :, top : top + square, left : left + square] + sign * eps
        proposal = _project_linf(proposal, x, eps)
        with torch.no_grad():
            losses = F.cross_entropy(model(proposal), y, reduction="none")
        mask = losses > best_loss
        best_loss[mask] = losses[mask]
        adv[mask] = proposal[mask]
    return adv.detach()


def build_art_classifier(model: nn.Module, cfg: ExperimentConfig) -> Any | None:
    """Build an ART classifier when the optional adversarial-robustness-toolbox is installed."""
    if cfg.attack_backend != "art":
        return None
    try:
        from art.estimators.classification import PyTorchClassifier
    except ImportError as exc:
        logger.warning("ART backend requested but unavailable; falling back to torch attacks: {}", exc)
        return None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        device_type="gpu" if cfg.device.type == "cuda" else "cpu",
    )


def art_attack(art_classifier: Any, x: torch.Tensor, attack_type: str, params: dict[str, Any]) -> torch.Tensor | None:
    """Generate adversarial examples with ART for canonical PGD/APGD/SquareAttack runs."""
    try:
        from art.attacks.evasion import AutoProjectedGradientDescent, ProjectedGradientDescent, SquareAttack
    except ImportError:
        return None

    if attack_type == "PGD":
        attacker = ProjectedGradientDescent(
            art_classifier,
            eps=float(params.get("eps", 0.1)),
            max_iter=int(params.get("steps", 40)),
            verbose=False,
        )
    elif attack_type == "AutoPGD":
        attacker = AutoProjectedGradientDescent(
            art_classifier,
            eps=float(params.get("eps", 0.1)),
            max_iter=int(params.get("steps", 100)),
            nb_random_init=int(params.get("n_restarts", 1)),
            verbose=False,
        )
    elif attack_type == "SquareAttack":
        attacker = SquareAttack(
            art_classifier,
            eps=float(params.get("eps", 0.2)),
            max_iter=int(params.get("max_iter", 300)),
            verbose=False,
        )
    else:
        return None

    x_adv = attacker.generate(x.detach().cpu().numpy())
    return torch.from_numpy(x_adv).to(device=x.device, dtype=x.dtype)


def generate_adversarial(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    attack_type: str,
    params: Any,
    art_classifier: Any | None = None,
) -> torch.Tensor:
    """Dispatch adversarial attack generation."""
    params = dict(params)
    if art_classifier is not None:
        x_art = art_attack(art_classifier, x, attack_type, params)
        if x_art is not None:
            return x_art

    if attack_type == "PGD":
        return pgd_attack(model, x, y, params)
    if attack_type == "AutoPGD":
        return auto_pgd_attack(model, x, y, params)
    if attack_type == "SquareAttack":
        return square_attack(model, x, y, params)
    raise ValueError(f"Unknown adversarial attack: {attack_type}")


def _save_tensors(
    cfg: ExperimentConfig,
    run_name: str,
    model_name: str,
    attack_type: str,
    level_name: str,
    xs: list[torch.Tensor],
    ys: list[torch.Tensor],
    preds: list[torch.Tensor],
) -> None:
    if not cfg.save_attack_tensors:
        return
    save_dir = cfg.artifact_dir / run_name / model_name / attack_type / level_name
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(xs), save_dir / "images.pt")
    torch.save(torch.cat(ys), save_dir / "labels.pt")
    torch.save(torch.cat(preds), save_dir / "preds.pt")


def evaluate_one_setting(
    model: nn.Module,
    loader: DataLoader,
    cfg: ExperimentConfig,
    attack_type: str,
    params: Any,
    art_classifier: Any | None = None,
) -> tuple[float, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Evaluate one model under one attack configuration."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    all_x: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    all_pred: list[torch.Tensor] = []

    iterator = tqdm(loader, desc=f"{attack_type}", leave=False, ncols=90)
    for x, y in iterator:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        if attack_type in ADVERSARIAL_ATTACKS:
            x_eval = generate_adversarial(model, x, y, attack_type, params, art_classifier=art_classifier)
        else:
            with torch.no_grad():
                x_eval = apply_physical_perturbation(x, attack_type, params)

        with torch.no_grad():
            pred = model(x_eval).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        all_x.append(x_eval.detach().cpu())
        all_y.append(y.detach().cpu())
        all_pred.append(pred.detach().cpu())

    if was_training:
        model.train()
    return correct / max(total, 1), all_x, all_y, all_pred


def run_robustness_benchmark(
    models: dict[str, nn.Module],
    data: DataBundle,
    cfg: ExperimentConfig,
    run_name: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run all configured robustness tests and return a nested accuracy dictionary."""
    results: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, model in models.items():
        logger.info("Benchmarking model: {}", model_name)
        art_classifier = build_art_classifier(model, cfg)
        results[model_name] = {}
        for attack_type, levels in cfg.attack_configs.items():
            results[model_name][attack_type] = {}
            loader = data.adv_eval if attack_type in ADVERSARIAL_ATTACKS else data.phys_eval
            for level_name, params in levels.items():
                logger.info("Running {} [{}] on {}", attack_type, level_name, model_name)
                acc, xs, ys, preds = evaluate_one_setting(
                    model,
                    loader,
                    cfg,
                    attack_type,
                    params,
                    art_classifier=art_classifier,
                )
                results[model_name][attack_type][level_name] = acc
                _save_tensors(cfg, run_name, model_name, attack_type, level_name, xs, ys, preds)
                logger.info("{} [{}] on {} | acc={:.2%}", attack_type, level_name, model_name, acc)
    return results
