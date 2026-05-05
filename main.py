# experiment/plateau_cliff/main.py
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plateau-Cliff robustness experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.json"),
        help="JSON config file. All experiment parameters should be edited there.",
    )
    return parser


def load_json_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = build_parser().parse_args()
    source_config_path = Path(args.config).resolve()
    user_config = load_json_config(source_config_path)

    profile = user_config.get("profile", "server")
    gpus = user_config.get("gpus", "0,1,2,3")
    requested_gpus = _parse_gpu_arg(str(gpus))
    if profile == "server" and requested_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
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
        profile=profile,
        data_dir=user_config.get("data_dir"),
        gpu_ids=visible_gpu_ids if profile == "server" else [],
    )
    cfg.apply_user_config(user_config)
    cfg.ensure_dirs()

    run_name = f"plateau_cliff_{profile}_{time.strftime('%Y%m%d_%H%M%S')}"
    setup_logger(cfg.log_dir, run_name)
    set_seed(cfg.seed)
    logger.info("Project root: {}", PROJECT_ROOT)
    logger.info("Config file: {}", source_config_path)
    logger.info("Device: {} | DataParallel: {} | gpu_ids={}", cfg.device, cfg.use_data_parallel, cfg.gpu_ids)
    logger.info("MNIST root: {} | fake_data={}", cfg.data_dir, cfg.use_fake_data)
    logger.info(
        "Fair training target: {:.2%} | stop_at_target={} | max_epochs={}",
        cfg.target_acc,
        cfg.stop_at_target,
        cfg.max_epochs,
    )

    data = build_data_bundle(cfg)
    raw_models = user_config.get("models", ["Mamba", "DNN", "nmODE"])
    if isinstance(raw_models, str):
        baseline_names = [name.strip() for name in raw_models.split(",") if name.strip()]
    else:
        baseline_names = [str(name).strip() for name in raw_models if str(name).strip()]
    force_retrain = bool(user_config.get("force_retrain", False))
    models = {}
    target_tag = f"tgt{fmt_float(cfg.target_acc)}"

    for model_name in baseline_names:
        model = train_or_load_classifier(
            model_name=model_name,
            tag=f"base_{target_tag}",
            cfg=cfg,
            train_loader=data.train,
            valid_loader=data.valid,
            regularizer_factory=lambda: None,
            force_retrain=force_retrain,
        )
        models[f"{model_name}_Base"] = model

    clr_tag = f"clr_lam{fmt_float(cfg.lambda_clr)}_{target_tag}"
    mamba_clr = train_or_load_classifier(
        model_name="Mamba",
        tag=clr_tag,
        cfg=cfg,
        train_loader=data.train,
        valid_loader=data.valid,
        regularizer_factory=lambda: CrossLipschitzRegularizer(lambda_reg=cfg.lambda_clr, num_classes=10),
        force_retrain=force_retrain,
    )
    models[f"Mamba_CLR_{cfg.lambda_clr:g}"] = mamba_clr

    joint_tag = (
        f"clr_pcp_joint_clr{fmt_float(cfg.lambda_clr)}"
        f"_plat{fmt_float(cfg.plateau_lambda)}"
        f"_beta{fmt_float(cfg.plateau_beta)}"
        f"_reward{fmt_float(cfg.cliff_reward)}_{target_tag}"
    )
    mamba_joint = train_or_load_classifier(
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
    models[f"Mamba_CLR_PCP_Joint_{cfg.lambda_clr:g}"] = mamba_joint

    results = run_robustness_benchmark(models, data, cfg, run_name=run_name)
    summary_path = cfg.result_dir / f"{run_name}_summary.json"
    resolved_config_path = cfg.result_dir / f"{run_name}_config.json"
    csv_path = cfg.result_dir / f"{run_name}_summary.csv"
    heatmap_path = cfg.result_dir / f"{run_name}_heatmap.png"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with resolved_config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg.as_jsonable(), f, indent=2, ensure_ascii=False)
    used_config_path = cfg.result_dir / f"{run_name}_input_config.json"
    with used_config_path.open("w", encoding="utf-8") as f:
        json.dump(user_config, f, indent=2, ensure_ascii=False)
    save_results_csv(results, csv_path)
    plot_heatmap(results, heatmap_path)

    logger.info("Summary JSON: {}", summary_path)
    logger.info("Summary CSV: {}", csv_path)
    logger.info("Resolved config JSON: {}", resolved_config_path)
    logger.info("Input config copy: {}", used_config_path)
    logger.info("Experiment finished")


if __name__ == "__main__":
    main()
