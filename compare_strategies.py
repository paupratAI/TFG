"""
compare_strategies.py — Compare the four ES neuron-selection strategies.

Runs omp, r2, cfs, and cfs_auc on the same dataset split, saves:
  - a PNG with train-accuracy curves and final validation/test accuracy bars
  - a JSON file with final metrics and per-epoch histories
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".font_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import train


STRATEGIES = ("omp", "r2", "cfs", "cfs_auc")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_comparison(args: argparse.Namespace) -> dict[str, object]:
    results: dict[str, object] = {}

    for strategy in args.strategies:
        print(f"\n=== Running strategy: {strategy} ===")
        state = train(
            fitness_strategy=strategy,
            epochs=args.epochs,
            mu=args.mu,
            n_offspring=args.offspring,
            sigma_init=args.sigma_init,
            gamma=args.gamma,
            data_id=args.data_id,
            data_name=args.data_name,
            data_version=args.data_version,
            test_size=args.test_size,
            validation_size=args.validation_size,
            data_home=args.data_home,
            verbose=args.verbose,
            seed=args.seed,
            save_dir=None,
        )
        results[strategy] = {
            "metrics": state.metrics,
            "history": state.history,
        }

    return results


def save_plot(results: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_curve, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.5),
        constrained_layout=True,
    )

    all_train_accs = []
    
    for strategy, payload in results.items():
        history = payload["history"]
        epochs = [int(row["epoch"]) for row in history]
        train_acc = [float(row["train_acc"]) for row in history]
        ax_curve.plot(epochs, train_acc, linewidth=2.0, label=strategy)

    if all_train_accs:
        min_acc = min(all_train_accs)
        # Ajustamos el límite inferior con un pequeño margen del 5% para que no toque el borde
        margin = (1.0 - min_acc) * 0.05
        ax_curve.set_ylim(min_acc - margin, 1.02)

    ax_curve.set_title("Train accuracy by generation")
    ax_curve.set_xlabel("Generation")
    ax_curve.set_ylabel("Train accuracy")
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend()

    strategies = list(results.keys())
    x = np.arange(len(strategies))
    width = 0.36
    val_acc = [
        float(results[strategy]["metrics"]["validation_acc"])
        for strategy in strategies
    ]
    test_acc = [
        float(results[strategy]["metrics"]["test_acc"])
        for strategy in strategies
    ]

    ax_bar.bar(x - width / 2, val_acc, width, label="Validation")
    ax_bar.bar(x + width / 2, test_acc, width, label="Test")
    ax_bar.set_title("Final validation/test accuracy")
    ax_bar.set_xlabel("Strategy")
    ax_bar.set_ylabel("Accuracy")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(strategies)
    ax_bar.set_ylim(0.0, 1.02)
    ax_bar.grid(True, axis="y", alpha=0.25)
    ax_bar.legend()

    for idx, (v_acc, t_acc) in enumerate(zip(val_acc, test_acc)):
        ax_bar.text(idx - width / 2, v_acc + 0.01, f"{v_acc:.3f}", ha="center", fontsize=8)
        ax_bar.text(idx + width / 2, t_acc + 0.01, f"{t_acc:.3f}", ha="center", fontsize=8)

    fig.suptitle("Michigan ES selection strategy comparison")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_json(results: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(results), fh, indent=2, sort_keys=True)


def print_summary(results: dict[str, object]) -> None:
    print("\n| Strategy | Final Train Acc | Validation Acc | Test Acc | Final LogLoss |")
    print("|---|---:|---:|---:|---:|")
    for strategy, payload in results.items():
        metrics = payload["metrics"]
        print(
            f"| {strategy} | "
            f"{float(metrics['final_train_acc']):.4f} | "
            f"{float(metrics['validation_acc']):.4f} | "
            f"{float(metrics['test_acc']):.4f} | "
            f"{float(metrics['final_log_loss']):.6f} |"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ES neuron-selection strategies on one split."
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mu", type=int, default=50)
    parser.add_argument("--offspring", type=int, default=350)
    parser.add_argument("--sigma-init", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--data-id", type=int, default=59)
    parser.add_argument("--data-name", default=None)
    parser.add_argument("--data-version", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--data-home", default=".openml_cache")
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs/strategy_comparison")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    results = run_comparison(args)

    plot_path = output_dir / "strategy_comparison.png"
    json_path = output_dir / "strategy_comparison.json"
    save_plot(results, plot_path)
    save_json(results, json_path)
    print_summary(results)
    print(f"\nSaved plot: {plot_path}")
    print(f"Saved metrics: {json_path}")


if __name__ == "__main__":
    main()
