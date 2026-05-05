# experiment/plateau_cliff/src/regularizers.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLipschitzRegularizer(nn.Module):
    """Cross-Lipschitz regularization from Hein and Andriushchenko (2017)."""

    def __init__(self, lambda_reg: float = 0.005, num_classes: int | None = None):
        super().__init__()
        self.lambda_reg = float(lambda_reg)
        self.num_classes = num_classes
        self.last_stats: dict[str, float] = {}

    def forward(self, x: torch.Tensor, logits: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        if self.lambda_reg <= 0:
            return logits.new_tensor(0.0)

        batch_size = x.shape[0]
        num_classes = self.num_classes or logits.shape[1]
        class_grads: list[torch.Tensor] = []

        for cls_idx in range(num_classes):
            grad_cls = torch.autograd.grad(
                outputs=logits[:, cls_idx].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            class_grads.append(grad_cls.reshape(batch_size, -1))

        reg_sum = logits.new_tensor(0.0)
        for left in range(num_classes):
            for right in range(left + 1, num_classes):
                grad_diff = class_grads[left] - class_grads[right]
                reg_sum = reg_sum + 2.0 * grad_diff.pow(2).sum(dim=1).sum()

        penalty = reg_sum / (num_classes * num_classes * batch_size)
        self.last_stats = {"cross_lip": float(penalty.detach().cpu())}
        return self.lambda_reg * penalty


class PlateauCliffPatchRegularizer(nn.Module):
    """
    Distance-aware gradient shaping used after Cross-Lipschitz robust training.

    Near confident, correct one-hot outputs, the coefficient is large and positive,
    encouraging a flat plateau. Near ambiguous outputs, the coefficient decays and
    may become slightly negative, allowing a steep cliff if cliff_reward > 0.
    """

    def __init__(
        self,
        plateau_lambda: float = 0.005,
        beta: float = 8.0,
        cliff_reward: float = 0.0,
        tau: float = 0.35,
        temperature: float = 0.05,
        score_space: str = "probability",
        max_negative_weight: float = 0.001,
    ):
        super().__init__()
        self.plateau_lambda = float(plateau_lambda)
        self.beta = float(beta)
        self.cliff_reward = float(cliff_reward)
        self.tau = float(tau)
        self.temperature = float(temperature)
        self.score_space = score_space
        self.max_negative_weight = float(max_negative_weight)
        self.last_stats: dict[str, float] = {}

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.score_space == "unit_logits":
            return F.normalize(logits, p=2, dim=1, eps=1e-12)
        if self.score_space == "probability":
            return F.softmax(logits, dim=1)
        raise ValueError(f"Unknown score_space: {self.score_space}")

    def forward(self, x: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.plateau_lambda <= 0 and self.cliff_reward <= 0:
            return logits.new_tensor(0.0)

        scores = self._scores(logits)
        one_hot = F.one_hot(target, num_classes=logits.shape[1]).to(dtype=logits.dtype, device=logits.device)
        distance = 0.5 * (scores - one_hot).pow(2).sum(dim=1)

        with torch.no_grad():
            plateau_weight = self.plateau_lambda * torch.exp(-self.beta * distance)
            cliff_gate = torch.sigmoid((distance - self.tau) / max(self.temperature, 1e-6))
            weight = plateau_weight - self.cliff_reward * cliff_gate
            weight = torch.clamp(weight, min=-self.max_negative_weight)

        target_scores = scores.gather(1, target.view(-1, 1)).sum()
        grad_target = torch.autograd.grad(
            outputs=target_scores,
            inputs=x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_norm_sq = grad_target.reshape(x.shape[0], -1).pow(2).sum(dim=1)
        penalty = (weight * grad_norm_sq).mean()

        self.last_stats = {
            "pcp_distance": float(distance.detach().mean().cpu()),
            "pcp_weight": float(weight.detach().mean().cpu()),
            "pcp_grad_norm_sq": float(grad_norm_sq.detach().mean().cpu()),
            "pcp_penalty": float(penalty.detach().cpu()),
        }
        return penalty


class CrossLipPlusPlateauCliffRegularizer(nn.Module):
    """
    Joint regularizer: Cross-Lipschitz plus the local Plateau-Cliff norm shaping term.

    This is the main experimental method after the first patch-only attempt failed:
    the Plateau-Cliff term no longer fine-tunes a finished CLR model by itself.
    Instead, Cross-Lip remains active throughout training and stabilizes the class
    score differences while the local norm term shapes confident regions.
    """

    def __init__(
        self,
        lambda_clr: float = 0.005,
        plateau_lambda: float = 0.0005,
        beta: float = 8.0,
        cliff_reward: float = 0.0,
        tau: float = 0.35,
        temperature: float = 0.05,
        score_space: str = "probability",
        num_classes: int | None = None,
    ):
        super().__init__()
        self.cross_lip = CrossLipschitzRegularizer(lambda_reg=lambda_clr, num_classes=num_classes)
        self.plateau_cliff = PlateauCliffPatchRegularizer(
            plateau_lambda=plateau_lambda,
            beta=beta,
            cliff_reward=cliff_reward,
            tau=tau,
            temperature=temperature,
            score_space=score_space,
        )
        self.last_stats: dict[str, float] = {}

    def forward(self, x: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_clr = self.cross_lip(x, logits, target)
        loss_pcp = self.plateau_cliff(x, logits, target)
        self.last_stats = {
            **self.cross_lip.last_stats,
            **self.plateau_cliff.last_stats,
            "loss_clr_scaled": float(loss_clr.detach().cpu()),
            "loss_pcp_scaled": float(loss_pcp.detach().cpu()),
        }
        return loss_clr + loss_pcp
