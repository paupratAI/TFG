"""
compare_strategies.py — Compare the four ES neuron-selection strategies.

Runs omp, r2, cfs, and cfs_auc on the same dataset split, saves:
  - a PNG with train-accuracy curves and final validation/test accuracy bars
  - a PNG with validation-accuracy curves
  - a PNG with diagnostic curves for gap, cross-entropy, calibration, and timing
  - a JSON file with final metrics and per-epoch histories
  - when several seeds/datasets are requested, aggregate JSON/PNG summaries
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
DATASET_PRESETS = {
    "ionosphere": {"id": 59, "name": None, "label": "Ionosphere"},
    "sonar": {"id": 40, "name": None, "label": "Sonar"},
    "breast-w": {"id": 15, "name": None, "label": "Breast-W"},
}


def _history_series(
    payload: dict[str, object],
    key: str,
    default: float | None = None,
) -> list[float]:
    values: list[float] = []
    for row in payload["history"]:
        if key in row and row[key] is not None:
            values.append(float(row[key]))
        elif default is not None:
            values.append(float(default))
    return values


def _epochs(payload: dict[str, object]) -> list[int]:
    return [int(row["epoch"]) for row in payload["history"]]


def _set_accuracy_ylim(ax, values: list[float]) -> None:
    if not values:
        ax.set_ylim(0.0, 1.02)
        return
    min_acc = min(values)
    margin = max((1.0 - min_acc) * 0.05, 0.01)
    ax.set_ylim(max(0.0, min_acc - margin), 1.02)


def _final_accuracy_bar(
    ax,
    results: dict[str, object],
    title: str = "Selected validation/test accuracy",
) -> None:
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

    ax.bar(x - width / 2, val_acc, width, label="Validation")
    ax.bar(x + width / 2, test_acc, width, label="Test")
    ax.set_title(title)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    for idx, (v_acc, t_acc) in enumerate(zip(val_acc, test_acc)):
        ax.text(idx - width / 2, v_acc + 0.01, f"{v_acc:.3f}", ha="center", fontsize=8)
        ax.text(idx + width / 2, t_acc + 0.01, f"{t_acc:.3f}", ha="center", fontsize=8)


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


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def _format_mean_std(summary: dict[str, float]) -> str:
    return f"{summary['mean']:.4f} +/- {summary['std']:.4f}"


def _selected_datasets(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.datasets:
        return [DATASET_PRESETS[name] for name in args.datasets]
    preset_by_id = {
        int(dataset["id"]): dataset
        for dataset in DATASET_PRESETS.values()
    }
    if args.data_name is None and args.data_id in preset_by_id:
        preset = preset_by_id[args.data_id]
        return [
            {
                "id": preset["id"],
                "name": None,
                "label": preset["label"],
            }
        ]
    return [
        {
            "id": args.data_id,
            "name": args.data_name,
            "label": args.data_name or f"OpenML-{args.data_id}",
        }
    ]


def _selected_seeds(args: argparse.Namespace) -> list[int]:
    return [int(seed) for seed in (args.seeds if args.seeds else [args.seed])]


def run_comparison(
    args: argparse.Namespace,
    dataset: dict[str, object],
    seed: int,
) -> dict[str, object]:
    results: dict[str, object] = {}

    for strategy in args.strategies:
        print(
            f"\n=== Dataset: {dataset['label']} | seed={seed} | "
            f"strategy: {strategy} ==="
        )
        state = train(
            fitness_strategy=strategy,
            epochs=args.epochs,
            mu=args.mu,
            n_offspring=args.offspring,
            sigma_init=args.sigma_init,
            gamma=args.gamma,
            data_id=int(dataset["id"]),
            data_name=dataset["name"],
            data_version=args.data_version,
            test_size=args.test_size,
            validation_size=args.validation_size,
            data_home=args.data_home,
            verbose=args.verbose,
            seed=seed,
            output_layer="logistic",
            save_dir=None,
        )
        results[strategy] = {
            "metrics": state.metrics,
            "history": state.history,
        }

    return results


def run_all_comparisons(args: argparse.Namespace) -> dict[str, object]:
    datasets = _selected_datasets(args)
    seeds = _selected_seeds(args)
    runs: list[dict[str, object]] = []

    for dataset in datasets:
        for seed in seeds:
            results = run_comparison(args, dataset, seed)
            runs.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "results": results,
                }
            )

    return {
        "config": {
            "strategies": list(args.strategies),
            "datasets": datasets,
            "seeds": seeds,
            "epochs": args.epochs,
            "mu": args.mu,
            "offspring": args.offspring,
            "sigma_init": args.sigma_init,
            "gamma": args.gamma,
            "test_size": args.test_size,
            "validation_size": args.validation_size,
        },
        "runs": runs,
        "summary": summarise_runs(runs, args.strategies),
    }


def summarise_runs(
    runs: list[dict[str, object]],
    strategies: list[str],
) -> dict[str, object]:
    metric_names = (
        "final_train_acc",
        "validation_acc",
        "test_acc",
        "train_mse",
        "validation_mse",
        "test_mse",
        "train_ce",
        "validation_ce",
        "test_ce",
        "train_brier",
        "validation_brier",
        "test_brier",
        "train_ece",
        "validation_ece",
        "test_ece",
        "train_roc_auc",
        "validation_roc_auc",
        "test_roc_auc",
        "final_gcv",
        "final_lambda",
        "selected_epoch",
        "select_ms_mean",
        "elapsed_seconds",
    )
    summary: dict[str, object] = {}
    dataset_labels = sorted({str(run["dataset"]["label"]) for run in runs})

    for dataset_label in dataset_labels:
        dataset_runs = [
            run for run in runs if str(run["dataset"]["label"]) == dataset_label
        ]
        summary[dataset_label] = {}
        for strategy in strategies:
            strategy_metrics = [
                run["results"][strategy]["metrics"]
                for run in dataset_runs
                if strategy in run["results"]
            ]
            metric_summary: dict[str, object] = {"n_runs": len(strategy_metrics)}
            for metric_name in metric_names:
                values = [
                    float(metrics[metric_name])
                    for metrics in strategy_metrics
                    if metrics.get(metric_name) is not None
                    and np.isfinite(float(metrics[metric_name]))
                ]
                if values:
                    metric_summary[metric_name] = _mean_std(values)
            summary[dataset_label][strategy] = metric_summary

    summary["overall"] = {}
    for strategy in strategies:
        strategy_metrics = [
            run["results"][strategy]["metrics"]
            for run in runs
            if strategy in run["results"]
        ]
        metric_summary = {"n_runs": len(strategy_metrics)}
        for metric_name in metric_names:
            values = [
                float(metrics[metric_name])
                for metrics in strategy_metrics
                if metrics.get(metric_name) is not None
                and np.isfinite(float(metrics[metric_name]))
            ]
            if values:
                metric_summary[metric_name] = _mean_std(values)
        summary["overall"][strategy] = metric_summary

    return summary


def save_train_plot(results: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_curve, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.5),
        constrained_layout=True,
    )

    all_train_accs = []
    for strategy, payload in results.items():
        epochs = _epochs(payload)
        train_acc = _history_series(payload, "train_acc")
        all_train_accs.extend(train_acc)
        ax_curve.plot(epochs, train_acc, linewidth=2.0, label=strategy)

    _set_accuracy_ylim(ax_curve, all_train_accs)
    ax_curve.set_title("Train accuracy by generation")
    ax_curve.set_xlabel("Generation")
    ax_curve.set_ylabel("Train accuracy")
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend()
    _final_accuracy_bar(ax_bar, results)

    fig.suptitle("Michigan ES selection strategy comparison")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_validation_plot(results: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_curve, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.5),
        constrained_layout=True,
    )

    all_val_accs = []
    for strategy, payload in results.items():
        val_acc = _history_series(payload, "validation_acc")
        if not val_acc:
            continue
        epochs = _epochs(payload)
        all_val_accs.extend(val_acc)
        line, = ax_curve.plot(epochs, val_acc, linewidth=2.0, label=strategy)
        selected_epoch = payload["metrics"].get("selected_epoch")
        if selected_epoch is not None:
            selected_idx = int(selected_epoch) - 1
            if 0 <= selected_idx < len(val_acc):
                ax_curve.scatter(
                    [int(selected_epoch)],
                    [val_acc[selected_idx]],
                    color=line.get_color(),
                    s=38,
                    zorder=3,
                )

    _set_accuracy_ylim(ax_curve, all_val_accs)
    ax_curve.set_title("Validation accuracy by generation")
    ax_curve.set_xlabel("Generation")
    ax_curve.set_ylabel("Validation accuracy")
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend()
    _final_accuracy_bar(ax_bar, results)

    fig.suptitle("Validation-driven strategy comparison")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_diagnostics_plot(results: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 9.0),
        constrained_layout=True,
    )
    ax_gap, ax_ce, ax_ece, ax_time = axes.ravel()

    for strategy, payload in results.items():
        epochs = _epochs(payload)
        train_acc = _history_series(payload, "train_acc")
        val_acc = _history_series(payload, "validation_acc")
        if train_acc and val_acc:
            gap = [tr - va for tr, va in zip(train_acc, val_acc)]
            ax_gap.plot(epochs, gap, linewidth=2.0, label=strategy)

        train_ce = np.asarray(_history_series(payload, "train_ce"), dtype=np.float64)
        if train_ce.size:
            ax_ce.plot(
                epochs,
                np.maximum(train_ce, 1e-12),
                linewidth=2.0,
                label=strategy,
            )

        val_ece = np.asarray(_history_series(payload, "validation_ece"), dtype=np.float64)
        if val_ece.size:
            ax_ece.plot(
                epochs,
                np.maximum(val_ece, 1e-12),
                linewidth=2.0,
                label=strategy,
            )

    strategies = list(results.keys())
    select_ms = [
        float(results[strategy]["metrics"]["select_ms_mean"])
        for strategy in strategies
    ]
    ax_time.bar(strategies, select_ms)

    ax_gap.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax_gap.set_title("Train-validation accuracy gap")
    ax_gap.set_xlabel("Generation")
    ax_gap.set_ylabel("Train acc - validation acc")
    ax_gap.grid(True, alpha=0.25)
    ax_gap.legend()

    ax_ce.set_yscale("log")
    ax_ce.set_title("Train cross-entropy by generation")
    ax_ce.set_xlabel("Generation")
    ax_ce.set_ylabel("Cross-entropy")
    ax_ce.grid(True, alpha=0.25)
    ax_ce.legend()

    ax_ece.set_title("Validation ECE by generation")
    ax_ece.set_xlabel("Generation")
    ax_ece.set_ylabel("Expected calibration error")
    ax_ece.grid(True, alpha=0.25)
    ax_ece.legend()

    ax_time.set_title("Mean selection cost")
    ax_time.set_xlabel("Strategy")
    ax_time.set_ylabel("Milliseconds")
    ax_time.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Selection strategy diagnostics")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_aggregate_accuracy_plot(
    summary: dict[str, object],
    strategies: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = [name for name in summary.keys() if name != "overall"]
    if not datasets:
        datasets = ["overall"]

    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(12.0, max(4.2, 3.8 * len(datasets))),
        constrained_layout=True,
        squeeze=False,
    )

    width = 0.36
    for ax, dataset_label in zip(axes.ravel(), datasets):
        dataset_summary = summary[dataset_label]
        x = np.arange(len(strategies))
        val_means = [
            dataset_summary[strategy]["validation_acc"]["mean"]
            for strategy in strategies
        ]
        val_stds = [
            dataset_summary[strategy]["validation_acc"]["std"]
            for strategy in strategies
        ]
        test_means = [
            dataset_summary[strategy]["test_acc"]["mean"]
            for strategy in strategies
        ]
        test_stds = [
            dataset_summary[strategy]["test_acc"]["std"]
            for strategy in strategies
        ]
        ax.bar(x - width / 2, val_means, width, yerr=val_stds, capsize=3, label="Validation")
        ax.bar(x + width / 2, test_means, width, yerr=test_stds, capsize=3, label="Test")
        ax.set_title(f"{dataset_label}: validation/test accuracy across seeds")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

    fig.suptitle("Aggregate strategy accuracy")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_aggregate_score_plot(
    summary: dict[str, object],
    strategies: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overall = summary["overall"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 4.8),
        constrained_layout=True,
    )
    metric_specs = [
        ("test_ce", "Test cross-entropy", "Lower is better"),
        ("test_roc_auc", "Test ROC-AUC", "Higher is better"),
        ("test_ece", "Test ECE", "Lower is better"),
    ]

    x = np.arange(len(strategies))
    for ax, (metric_name, title, ylabel) in zip(axes, metric_specs):
        means = [overall[strategy][metric_name]["mean"] for strategy in strategies]
        stds = [overall[strategy][metric_name]["std"] for strategy in strategies]
        ax.bar(x, means, yerr=stds, capsize=3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
        if metric_name == "test_roc_auc":
            ax.set_ylim(0.0, 1.02)

    fig.suptitle("Aggregate logistic probability-quality metrics")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_json(results: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(results), fh, indent=2, sort_keys=True)


def print_summary(results: dict[str, object]) -> None:
    print(
        "\n| Strategy | Selected Gen | Final Train Acc | Validation Acc | Test Acc | "
        "Test CE | Test Brier | Test ECE | Test ROC-AUC |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for strategy, payload in results.items():
        metrics = payload["metrics"]
        print(
            f"| {strategy} | "
            f"{int(metrics['selected_epoch']):d} | "
            f"{float(metrics['final_train_acc']):.4f} | "
            f"{float(metrics['validation_acc']):.4f} | "
            f"{float(metrics['test_acc']):.4f} | "
            f"{float(metrics['test_ce']):.6f} | "
            f"{float(metrics['test_brier']):.6f} | "
            f"{float(metrics['test_ece']):.6f} | "
            f"{float(metrics['test_roc_auc']):.4f} |"
        )


def print_aggregate_summary(summary: dict[str, object], strategies: list[str]) -> None:
    print("\n## Aggregate Multi-Seed/Multi-Dataset Summary")
    for dataset_label, dataset_summary in summary.items():
        if dataset_label == "overall":
            continue
        print(f"\n### {dataset_label}")
        print("| Strategy | Runs | Val Acc | Test Acc | Test CE | Test Brier | Test ECE | Test ROC-AUC |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for strategy in strategies:
            stats = dataset_summary[strategy]
            print(
                f"| {strategy} | {int(stats['n_runs'])} | "
                f"{_format_mean_std(stats['validation_acc'])} | "
                f"{_format_mean_std(stats['test_acc'])} | "
                f"{_format_mean_std(stats['test_ce'])} | "
                f"{_format_mean_std(stats['test_brier'])} | "
                f"{_format_mean_std(stats['test_ece'])} | "
                f"{_format_mean_std(stats['test_roc_auc'])} | "
            )

    print("\n### Overall")
    print("| Strategy | Runs | Val Acc | Test Acc | Test CE | Test Brier | Test ECE | Test ROC-AUC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for strategy in strategies:
        stats = summary["overall"][strategy]
        print(
            f"| {strategy} | {int(stats['n_runs'])} | "
            f"{_format_mean_std(stats['validation_acc'])} | "
            f"{_format_mean_std(stats['test_acc'])} | "
            f"{_format_mean_std(stats['test_ce'])} | "
            f"{_format_mean_std(stats['test_brier'])} | "
            f"{_format_mean_std(stats['test_ece'])} | "
            f"{_format_mean_std(stats['test_roc_auc'])} | "
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
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_PRESETS),
        default=None,
        help=(
            "Run preset datasets. If omitted, the script uses --data-id/"
            "--data-name as a single dataset."
        ),
    )
    parser.add_argument("--data-version", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--data-home", default=".openml_cache")
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run several random seeds. If omitted, --seed is used.",
    )
    parser.add_argument("--output-dir", default="runs/strategy_comparison")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    payload = run_all_comparisons(args)
    first_run = payload["runs"][0]
    results = first_run["results"]
    summary = payload["summary"]

    train_plot_path = output_dir / "strategy_comparison.png"
    validation_plot_path = output_dir / "validation_accuracy.png"
    diagnostics_plot_path = output_dir / "strategy_diagnostics.png"
    aggregate_accuracy_path = output_dir / "aggregate_accuracy.png"
    aggregate_score_path = output_dir / "aggregate_score_metrics.png"
    json_path = output_dir / "strategy_comparison.json"
    all_runs_json_path = output_dir / "strategy_comparison_all_runs.json"
    summary_json_path = output_dir / "strategy_comparison_summary.json"
    save_train_plot(results, train_plot_path)
    save_validation_plot(results, validation_plot_path)
    save_diagnostics_plot(results, diagnostics_plot_path)
    save_aggregate_accuracy_plot(summary, list(args.strategies), aggregate_accuracy_path)
    save_aggregate_score_plot(summary, list(args.strategies), aggregate_score_path)
    save_json(results, json_path)
    save_json(payload, all_runs_json_path)
    save_json(summary, summary_json_path)
    print_summary(results)
    print_aggregate_summary(summary, list(args.strategies))
    print(f"\nSaved train plot: {train_plot_path}")
    print(f"Saved validation plot: {validation_plot_path}")
    print(f"Saved diagnostics plot: {diagnostics_plot_path}")
    print(f"Saved aggregate accuracy plot: {aggregate_accuracy_path}")
    print(f"Saved aggregate score plot: {aggregate_score_path}")
    print(f"Saved metrics: {json_path}")
    print(f"Saved all runs: {all_runs_json_path}")
    print(f"Saved aggregate summary: {summary_json_path}")


if __name__ == "__main__":
    main()
