# experiment/plateau_cliff/src/plotting.py
from __future__ import annotations

import csv
from pathlib import Path

from .logging_utils import logger


def flatten_results(results: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, str | float]]:
    """Flatten nested benchmark results for CSV export and plotting."""
    rows: list[dict[str, str | float]] = []
    for model_name, attack_map in results.items():
        for attack_name, level_map in attack_map.items():
            for level_name, acc in level_map.items():
                rows.append({"model": model_name, "attack": attack_name, "level": level_name, "accuracy": acc})
    return rows


def save_results_csv(results: dict[str, dict[str, dict[str, float]]], path: Path) -> None:
    """Save benchmark results as a tidy CSV table."""
    rows = flatten_results(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "attack", "level", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(results: dict[str, dict[str, dict[str, float]]], path: Path) -> None:
    """Render a heatmap matching the JSON benchmark structure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        logger.warning("Skipping heatmap because plotting dependency is missing: {}", exc)
        return

    model_names = list(results.keys())
    if not model_names:
        return

    attack_order = ["Blur", "Resize", "Noise", "PGD", "AutoPGD", "SquareAttack"]
    attack_labels = {
        "Blur": "B",
        "Resize": "R",
        "Noise": "N",
        "PGD": "P",
        "AutoPGD": "AP",
        "SquareAttack": "SQ",
    }
    level_labels = {
        "Weak": "W",
        "Medium": "M",
        "Extreme": "E",
        "Smoke": "S",
    }
    level_order = ["Weak", "Medium", "Extreme", "Smoke"]
    matrix_rows = []
    for model_name in model_names:
        row = {}
        for attack in attack_order:
            if attack not in results[model_name]:
                continue
            levels = results[model_name][attack]
            ordered_levels = [level for level in level_order if level in levels]
            ordered_levels += [level for level in levels.keys() if level not in ordered_levels]
            for level in ordered_levels:
                attack_label = attack_labels.get(attack, attack)
                level_label = level_labels.get(level, level[:1])
                row[f"{attack_label}-{level_label}"] = levels[level]
        matrix_rows.append(row)

    df = pd.DataFrame(matrix_rows, index=model_names)
    if df.empty:
        return

    width = max(10, 0.55 * len(df.columns) + 2)
    height = max(3, 0.65 * len(df.index) + 2)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        df,
        annot=True,
        fmt=".1%",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        cbar=False,
        annot_kws={"size": 9},
        ax=ax,
    )
    ax.set_title("Robustness Benchmark Accuracy", fontsize=14, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    legend_text = (
        "Abbrev: B=Blur, R=Resize, N=Noise, P=PGD, AP=AutoPGD, SQ=SquareAttack; "
        "W=Weak, M=Medium, E=Extreme."
    )
    fig.text(0.5, 0.02, legend_text, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap saved to {}", path)
