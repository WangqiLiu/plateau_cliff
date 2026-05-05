# experiment/plateau_cliff/src/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def _full_attack_configs() -> dict[str, dict[str, Any]]:
    return {
        "Blur": {"Weak": 5, "Medium": 7, "Extreme": 12},
        "Resize": {"Weak": 14, "Medium": 7, "Extreme": 4},
        "Noise": {"Weak": 0.10, "Medium": 0.30, "Extreme": 0.50},
        "PGD": {
            "Weak": {"eps": 0.07, "steps": 10},
            "Medium": {"eps": 0.10, "steps": 40},
            "Extreme": {"eps": 0.15, "steps": 80},
        },
        "AutoPGD": {
            "Weak": {"eps": 0.07, "steps": 20, "n_restarts": 1},
            "Medium": {"eps": 0.10, "steps": 40, "n_restarts": 3},
            "Extreme": {"eps": 0.15, "steps": 80, "n_restarts": 5},
        },
        "SquareAttack": {
            "Weak": {"eps": 0.10, "max_iter": 100},
            "Medium": {"eps": 0.20, "max_iter": 300},
            "Extreme": {"eps": 0.30, "max_iter": 500},
        },
    }


def _smoke_attack_configs() -> dict[str, dict[str, Any]]:
    return {
        "Blur": {"Smoke": 3},
        "Resize": {"Smoke": 14},
        "Noise": {"Smoke": 0.10},
        "PGD": {"Smoke": {"eps": 0.05, "steps": 2}},
        "AutoPGD": {"Smoke": {"eps": 0.05, "steps": 2, "n_restarts": 1}},
        "SquareAttack": {"Smoke": {"eps": 0.10, "max_iter": 2}},
    }


@dataclass
class ExperimentConfig:
    """Central configuration for training, patch fine-tuning, and robustness evaluation."""

    project_root: Path
    profile: str = "server"
    seed: int = 20260420
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    gpu_ids: list[int] = field(default_factory=list)
    use_data_parallel: bool = False

    data_dir: Path = field(default_factory=Path)
    download_data: bool = False
    use_fake_data: bool = False

    model_dir: Path = field(default_factory=Path)
    result_dir: Path = field(default_factory=Path)
    log_dir: Path = field(default_factory=Path)
    artifact_dir: Path = field(default_factory=Path)

    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 20
    min_epochs: int = 3
    target_acc: float = 0.97
    stop_at_target: bool = True
    eval_only: bool = False

    lambda_clr: float = 0.005
    patch_epochs: int = 10
    patch_lr: float = 3e-4
    plateau_lambda: float = 0.0005
    plateau_beta: float = 8.0
    cliff_reward: float = 0.0
    cliff_tau: float = 0.35
    cliff_temperature: float = 0.05
    score_space: str = "probability"

    train_size: int | None = None
    valid_size: int | None = None
    phys_eval_size: int | None = None
    adv_eval_size: int | None = None
    attack_configs: dict[str, dict[str, Any]] = field(default_factory=_full_attack_configs)
    attack_backend: str = "art"
    save_attack_tensors: bool = False

    @classmethod
    def from_profile(
        cls,
        project_root: Path,
        profile: str,
        data_dir: str | Path | None = None,
        gpu_ids: list[int] | None = None,
    ) -> "ExperimentConfig":
        project_root = project_root.resolve()
        cfg = cls(project_root=project_root, profile=profile)
        cfg.model_dir = project_root / "models"
        cfg.result_dir = project_root / "results"
        cfg.log_dir = project_root / "log"
        cfg.artifact_dir = project_root / "artifacts"

        if data_dir is None:
            env_root = os.getenv("MNIST_ROOT")
            cfg.data_dir = Path(env_root) if env_root else project_root / "data"
        else:
            cfg.data_dir = Path(data_dir)

        if profile == "smoke":
            cfg.device = torch.device("cpu")
            cfg.gpu_ids = []
            cfg.use_data_parallel = False
            cfg.use_fake_data = True
            cfg.batch_size = 8
            cfg.eval_batch_size = 8
            cfg.lr = 1e-3
            cfg.max_epochs = 1
            cfg.min_epochs = 1
            cfg.target_acc = 0.0
            cfg.stop_at_target = True
            cfg.eval_only = False
            cfg.train_size = 16
            cfg.valid_size = 8
            cfg.phys_eval_size = 8
            cfg.adv_eval_size = 4
            cfg.lambda_clr = 1e-4
            cfg.patch_epochs = 1
            cfg.patch_lr = 1e-3
            cfg.plateau_lambda = 1e-4
            cfg.plateau_beta = 4.0
            cfg.attack_configs = _smoke_attack_configs()
            cfg.attack_backend = "torch"
            cfg.save_attack_tensors = False
            return cfg

        requested_gpus = [0, 1, 2, 3] if gpu_ids is None else gpu_ids
        if torch.cuda.is_available() and requested_gpus:
            cfg.device = torch.device(f"cuda:{requested_gpus[0]}")
            cfg.gpu_ids = requested_gpus
            cfg.use_data_parallel = len(requested_gpus) > 1
            cfg.num_workers = 4
        else:
            cfg.device = torch.device("cpu")
            cfg.gpu_ids = []
            cfg.use_data_parallel = False
            cfg.num_workers = 0

        cfg.train_size = None
        cfg.valid_size = None
        cfg.phys_eval_size = None
        cfg.adv_eval_size = 3000
        cfg.attack_configs = _full_attack_configs()
        return cfg

    def ensure_dirs(self) -> None:
        for path in [self.model_dir, self.result_dir, self.log_dir, self.artifact_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def apply_user_config(self, user_config: dict[str, Any]) -> None:
        """Apply JSON-defined experiment knobs after profile defaults are created."""
        mapping = {
            "seed": "seed",
            "batch_size": "batch_size",
            "eval_batch_size": "eval_batch_size",
            "lr": "lr",
            "weight_decay": "weight_decay",
            "max_epochs": "max_epochs",
            "min_epochs": "min_epochs",
            "target_acc": "target_acc",
            "stop_at_target": "stop_at_target",
            "eval_only": "eval_only",
            "train_size": "train_size",
            "valid_size": "valid_size",
            "phys_eval_size": "phys_eval_size",
            "adv_eval_size": "adv_eval_size",
            "lambda_clr": "lambda_clr",
            "patch_epochs": "patch_epochs",
            "patch_lr": "patch_lr",
            "plateau_lambda": "plateau_lambda",
            "plateau_beta": "plateau_beta",
            "cliff_reward": "cliff_reward",
            "cliff_tau": "cliff_tau",
            "cliff_temperature": "cliff_temperature",
            "score_space": "score_space",
            "attack_backend": "attack_backend",
            "save_attack_tensors": "save_attack_tensors",
            "download_data": "download_data",
        }
        for json_key, attr_name in mapping.items():
            if json_key in user_config and user_config[json_key] is not None:
                setattr(self, attr_name, user_config[json_key])

        if user_config.get("num_workers") is not None:
            self.num_workers = int(user_config["num_workers"])
        if user_config.get("attack_configs") is not None:
            self.attack_configs = user_config["attack_configs"]

    def as_jsonable(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, torch.device):
                data[key] = str(value)
        return data
