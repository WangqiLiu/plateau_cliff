# experiment/plateau_cliff/plot_only.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate a heatmap from an existing summary JSON.")
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to an existing *_summary.json file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output PNG path. Defaults to replacing _summary.json with _heatmap.png.",
    )
    return parser


def default_output_path(summary_path: Path) -> Path:
    name = summary_path.name
    if name.endswith("_summary.json"):
        return summary_path.with_name(name.replace("_summary.json", "_heatmap.png"))
    return summary_path.with_suffix(".heatmap.png")


def main() -> None:
    args = build_parser().parse_args()
    summary_path = Path(args.summary).resolve()
    if not summary_path.exists():
        raise SystemExit(f"Summary JSON not found: {summary_path}")

    output_path = Path(args.output).resolve() if args.output else default_output_path(summary_path)

    try:
        from src.plotting import plot_heatmap
    except ModuleNotFoundError as exc:
        req = PROJECT_ROOT / "requirements.txt"
        raise SystemExit(f"Missing dependency '{exc.name}'. Install the environment from {req}.") from exc

    with summary_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    plot_heatmap(results, output_path)
    print(f"Heatmap saved to: {output_path}")


if __name__ == "__main__":
    main()
