# experiment/plateau_cliff/test.py
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_gpu_arg(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _server_test_config(root_config: dict) -> dict:
    base = {key: value for key, value in root_config.items() if key != "server_test"}
    overrides = root_config.get("server_test", {})
    merged = {**base, **overrides}
    merged["profile"] = "server"
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Server-side small GPU test for Plateau-Cliff experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.json"),
        help="Config JSON. The server_test section controls this script.",
    )
    return parser


def _model_list(raw_models: object) -> list[str]:
    if isinstance(raw_models, str):
        return [name.strip() for name in raw_models.split(",") if name.strip()]
    if isinstance(raw_models, list):
        return [str(name).strip() for name in raw_models if str(name).strip()]
    return ["Mamba"]


def main() -> None:
    args = build_parser().parse_args()
    source_config_path = Path(args.config).resolve()
    root_config = _load_json(source_config_path)
    user_config = _server_test_config(root_config)

    gpus = str(user_config.get("gpus", "0"))
    requested_gpus = _parse_gpu_arg(gpus)
    if requested_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        visible_gpu_ids = list(range(len(requested_gpus)))
    else:
        visible_gpu_ids = []

    try:
        from src.config import ExperimentConfig
        from src.data import build_data_bundle
        from src.evaluate import run_robustness_benchmark
        from src.logging_utils import logger, setup_logger
        from src.plotting import plot_heatmap, save_results_csv
        from src.regularizers import CrossLipPlusPlateauCliffRegularizer, CrossLipschitzRegularizer
        from src.runtime import set_seed
        from src.train import fmt_float, train_or_load_classifier
    except ModuleNotFoundError as exc:
        req = PROJECT_ROOT / "requirements.txt"
        raise SystemExit(f"Missing dependency '{exc.name}'. Install the environment from {req}.") from exc

    cfg = ExperimentConfig.from_profile(
        project_root=PROJECT_ROOT,
        profile="server",
        data_dir=user_config.get("data_dir"),
        gpu_ids=visible_gpu_ids,
    )
    cfg.apply_user_config(user_config)
    cfg.ensure_dirs()

    run_name = f"server_test_{time.strftime('%Y%m%d_%H%M%S')}"
    setup_logger(cfg.log_dir, run_name)
    set_seed(cfg.seed)

    if cfg.device.type != "cuda":
        raise SystemExit("Server test requires CUDA. No CPU/FakeData fallback is used now.")

    logger.info("Starting server GPU chain test")
    logger.info("Config file: {}", source_config_path)
    logger.info("Device: {} | DataParallel: {} | gpu_ids={}", cfg.device, cfg.use_data_parallel, cfg.gpu_ids)
    logger.info(
        "Test scale: train={} valid={} phys_eval={} adv_eval={} max_epochs={}",
        cfg.train_size,
        cfg.valid_size,
        cfg.phys_eval_size,
        cfg.adv_eval_size,
        cfg.max_epochs,
    )

    data = build_data_bundle(cfg)
    target_tag = f"tgt{fmt_float(cfg.target_acc)}"
    clr_tag = f"clr_lam{fmt_float(cfg.lambda_clr)}_{target_tag}_server_test"
    joint_tag = (
        f"clr_pcp_joint_clr{fmt_float(cfg.lambda_clr)}"
        f"_plat{fmt_float(cfg.plateau_lambda)}"
        f"_beta{fmt_float(cfg.plateau_beta)}_{target_tag}_server_test"
    )
    force_retrain = bool(user_config.get("force_retrain", True))

    models = {}
    for model_name in _model_list(user_config.get("models", ["Mamba"])):
        base = train_or_load_classifier(
            model_name=model_name,
            tag=f"base_{target_tag}_server_test",
            cfg=cfg,
            train_loader=data.train,
            valid_loader=data.valid,
            regularizer_factory=lambda: None,
            force_retrain=force_retrain,
        )
        models[f"{model_name}_Base"] = base

    clr = train_or_load_classifier(
        model_name="Mamba",
        tag=clr_tag,
        cfg=cfg,
        train_loader=data.train,
        valid_loader=data.valid,
        regularizer_factory=lambda: CrossLipschitzRegularizer(lambda_reg=cfg.lambda_clr, num_classes=10),
        force_retrain=force_retrain,
    )
    joint = train_or_load_classifier(
        model_name="Mamba",
        cfg=cfg,
        tag=joint_tag,
        train_loader=data.train,
        valid_loader=data.valid,
        regularizer_factory=lambda: CrossLipPlusPlateauCliffRegularizer(
            lambda_clr=cfg.lambda_clr,
            plateau_lambda=cfg.plateau_lambda,
            beta=cfg.plateau_beta,
            cliff_reward=cfg.cliff_reward,
            tau=cfg.cliff_tau,
            temperature=cfg.cliff_temperature,
            score_space=cfg.score_space,
            num_classes=10,
        ),
        force_retrain=force_retrain,
    )
    models[f"Mamba_CLR_{cfg.lambda_clr:g}"] = clr
    models[f"Mamba_CLR_PCP_Joint_{cfg.lambda_clr:g}"] = joint

    results = run_robustness_benchmark(models, data, cfg, run_name=run_name)
    summary_path = cfg.result_dir / f"{run_name}_summary.json"
    csv_path = cfg.result_dir / f"{run_name}_summary.csv"
    heatmap_path = cfg.result_dir / f"{run_name}_heatmap.png"
    resolved_config_path = cfg.result_dir / f"{run_name}_config.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with resolved_config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg.as_jsonable(), f, indent=2, ensure_ascii=False)
    save_results_csv(results, csv_path)
    plot_heatmap(results, heatmap_path)

    logger.info("Server test summary JSON: {}", summary_path)
    logger.info("Server test CSV: {}", csv_path)
    logger.info("Server test heatmap: {}", heatmap_path)
    logger.info("Server GPU chain test passed")


if __name__ == "__main__":
    main()
