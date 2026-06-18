"""
make_final_plots.py

Generate the final thesis figures that are directly used by the Overleaf
memory. The script reads saved experiment summaries where available and uses
the final reported regression table values for the regression-only diagnostic
plots. It does not run any training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".font_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS_DIR = Path("runs")
PLOTS_DIR = Path("plots_overleaf")

BLUE = "#1f77b4"
ORANGE = "#d07a24"
GREEN = "#2b8a3e"
RED = "#b23b3b"
GREY = "#8c8c8c"
DARK = "#2f2f2f"
LIGHT_BLUE = "#b8d8f2"
LIGHT_ORANGE = "#f0c08a"


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    for directory in (RUNS_DIR, PLOTS_DIR):
        fig.savefig(directory / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _mean(summary: dict, key: str) -> float:
    return float(summary[key]["mean"])


def _std(summary: dict, key: str) -> float:
    return float(summary[key].get("std", 0.0))


def _sem(summary: dict, key: str) -> float:
    return float(summary[key].get("sem", 0.0))


def save_binary_benchmark_summary() -> None:
    payload = _load_json("runs/binary_logistic/summary.json")
    overall = payload["baseline_suite"]["summary"]["overall"]
    ranks = payload["baseline_suite"]["test_accuracy_ranks"]

    models = [
        "RBF SVM",
        "Random Forest",
        "HistGradientBoosting",
        "Logistic Regression",
        "ES-CFS Logistic CE",
        "Linear Logistic CE",
        "Gradient MLP",
        "Random tanh Logistic CE",
    ]
    labels = [
        "RBF SVM",
        "Random\nForest",
        "HistGB",
        "Logistic\nRegression",
        "ES-CFS\nLogistic",
        "Linear\nLogistic",
        "Gradient\nMLP",
        "Random\nTanh",
    ]
    colors = [
        GREY,
        GREY,
        GREY,
        GREY,
        BLUE,
        ORANGE,
        GREY,
        LIGHT_ORANGE,
    ]

    acc = [_mean(overall[m], "test_acc") for m in models]
    auc = [_mean(overall[m], "test_roc_auc") for m in models]
    rank_mean = [float(ranks[m]["mean"]) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(models))

    ax = axes[0]
    bars = ax.bar(x, acc, color=colors, edgecolor="white")
    ax.set_title("Binary classification: test accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.78, 0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{acc[idx]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = ax.twinx()
    ax2.plot(x, auc, color=DARK, marker="o", linewidth=1.8, label="Test AUC")
    ax2.set_ylabel("AUC", color=DARK)
    ax2.set_ylim(0.84, 1.0)
    ax2.tick_params(axis="y", labelcolor=DARK)

    ax = axes[1]
    bars = ax.bar(x, rank_mean, color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_title("Average test-accuracy rank")
    ax.set_ylabel("Rank (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.18,
            f"{rank_mean[idx]:.1f}",
            ha="center",
            va="top",
            fontsize=8,
            color="white" if colors[idx] in {BLUE, ORANGE, GREY} else DARK,
        )

    fig.suptitle("Binary logistic benchmark: performance and robustness across splits")
    _save(fig, "binary_benchmark_summary.png")


def save_strategy_cost_summary() -> None:
    payload = _load_json("runs/strategy_comparison/strategy_comparison_summary.json")
    overall = payload["overall"]
    strategies = ["r2", "omp", "cfs", "cfs_auc"]
    labels = [r"$R^2$", "OMP", "CFS", "CFS-AUC"]
    colors = [GREY, LIGHT_ORANGE, BLUE, ORANGE]
    select_ms = [overall[strategy]["select_ms_mean"]["mean"] for strategy in strategies]
    elapsed = [overall[strategy]["elapsed_seconds"]["mean"] for strategy in strategies]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    x = np.arange(len(strategies))

    bars = axes[0].bar(x, select_ms, color=colors, edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_title("Selection cost per generation")
    axes[0].set_ylabel("Mean selection time (ms, log scale)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(True, axis="y", alpha=0.25, which="both")
    for bar, value in zip(bars, select_ms):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:.3f}", ha="center", fontsize=8)

    bars = axes[1].bar(x, elapsed, color=colors, edgecolor="white")
    axes[1].set_title("Total training time")
    axes[1].set_ylabel("Mean elapsed time (s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, elapsed):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.3f}", ha="center", fontsize=8)

    fig.suptitle("Computational cost of neuron-selection strategies")
    _save(fig, "strategy_cost_summary.png")


REGRESSION_ROWS = {
    "Higgs-12000": {
        "Linear Ridge+GCV": {"test_rmse": 0.471, "test_r2": 0.111},
        "Random tanh Ridge+GCV": {"test_rmse": 0.469, "test_r2": 0.121},
        "ES-selected Ridge+GCV": {"test_rmse": 0.461, "test_r2": 0.148},
        "HistGradientBoosting": {"test_rmse": 0.443, "test_r2": 0.217},
        "Gradient MLP": {"test_rmse": 0.466, "test_r2": 0.130},
    },
    "Friedman1-12000": {
        "Linear Ridge+GCV": {"test_rmse": 2.733, "test_r2": 0.703},
        "Random tanh Ridge+GCV": {"test_rmse": 2.722, "test_r2": 0.705},
        "ES-selected Ridge+GCV": {"test_rmse": 2.070, "test_r2": 0.830},
        "HistGradientBoosting": {"test_rmse": 1.178, "test_r2": 0.945},
        "Gradient MLP": {"test_rmse": 1.278, "test_r2": 0.935},
    },
    "Diabetes": {
        "Linear Ridge+GCV": {"test_rmse": 54.236, "test_r2": 0.445},
        "Random tanh Ridge+GCV": {"test_rmse": 53.880, "test_r2": 0.452},
        "ES-selected Ridge+GCV": {"test_rmse": 50.544, "test_r2": 0.518},
        "HistGradientBoosting": {"test_rmse": 50.865, "test_r2": 0.512},
        "Gradient MLP": {"test_rmse": 55.973, "test_r2": 0.409},
    },
}


SOFTMAX_ROWS = {
    "Digits": {
        "Linear Softmax CE": {"test_accuracy": 0.960},
        "Random tanh Softmax CE": {"test_accuracy": 0.955},
        "ES-CFS Softmax CE": {"test_accuracy": 0.960},
    },
    "GaussianQuantiles-3000": {
        "Linear Softmax CE": {"test_accuracy": 0.329},
        "Random tanh Softmax CE": {"test_accuracy": 0.995},
        "ES-CFS Softmax CE": {"test_accuracy": 0.998},
    },
    "Spiral3-3000": {
        "Linear Softmax CE": {"test_accuracy": 0.333},
        "Random tanh Softmax CE": {"test_accuracy": 0.855},
        "ES-CFS Softmax CE": {"test_accuracy": 0.985},
    },
}


def save_regression_model_comparison() -> None:
    models = ["Linear Ridge+GCV", "Random tanh Ridge+GCV", "ES-selected Ridge+GCV"]
    labels = ["Linear\nRidge+GCV", "Random tanh\nRidge+GCV", "ES-selected\nRidge+GCV"]
    colors = [GREY, LIGHT_ORANGE, BLUE]
    values = []
    for model in models:
        normalised = []
        for rows in REGRESSION_ROWS.values():
            baseline = rows["Linear Ridge+GCV"]["test_rmse"]
            normalised.append(rows[model]["test_rmse"] / baseline)
        values.append(float(np.mean(normalised)))

    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    x = np.arange(len(models))
    bars = ax.bar(x, values, color=colors, edgecolor="white")
    ax.axhline(1.0, color=DARK, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title("Mean normalised test RMSE across regression datasets")
    ax.set_ylabel("Normalised RMSE (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.84, 1.03)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}", ha="center", fontsize=9)

    fig.suptitle("Regression Ridge representation comparison")
    _save(fig, "regression_model_comparison.png")


def save_regression_value_summary() -> None:
    datasets = list(REGRESSION_ROWS)
    simple_models = ["Linear Ridge+GCV", "Random tanh Ridge+GCV"]
    rmse_gain = []
    for dataset in datasets:
        rows = REGRESSION_ROWS[dataset]
        best_simple_rmse = min(rows[m]["test_rmse"] for m in simple_models)
        es_rmse = rows["ES-selected Ridge+GCV"]["test_rmse"]
        rmse_gain.append(100.0 * (best_simple_rmse - es_rmse) / best_simple_rmse)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), constrained_layout=True)
    x = np.arange(len(datasets))

    bars = ax.bar(x, rmse_gain, color=BLUE, edgecolor="white")
    ax.axhline(0.0, color=DARK, linewidth=1)
    ax.set_title("Gain over the best simple Ridge variant")
    ax.set_ylabel("Test RMSE reduction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=12, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, rmse_gain):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}%", ha="center", fontsize=9)

    fig.suptitle("Regression Ridge benchmark: what the evolved representation adds")
    _save(fig, "regression_value_summary.png")


def save_softmax_model_comparison() -> None:
    models = ["Linear Softmax CE", "Random tanh Softmax CE", "ES-CFS Softmax CE"]
    labels = ["Linear\nSoftmax CE", "Random tanh\nSoftmax CE", "ES-CFS\nSoftmax CE"]
    colors = [GREY, LIGHT_ORANGE, BLUE]
    values = []
    for model in models:
        acc = [rows[model]["test_accuracy"] for rows in SOFTMAX_ROWS.values()]
        values.append(float(np.mean(acc)))

    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    x = np.arange(len(models))
    bars = ax.bar(x, values, color=colors, edgecolor="white")
    ax.set_title("Mean test accuracy across multiclass datasets")
    ax.set_ylabel("Accuracy (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)

    fig.suptitle("Multiclass softmax representation comparison")
    _save(fig, "softmax_model_comparison.png")


def save_michigan_pittsburgh_summary() -> None:
    payload = _load_json("runs/michigan_pittsburgh/michigan_pittsburgh_summary.json")
    overall = payload["summary"]["overall"]
    order = ["michigan_cfs", "pittsburgh_matched"]
    labels = ["Michigan\nES-CFS", "Layer-level\nmatched"]
    colors = [BLUE, ORANGE]

    test_acc = [_mean(overall[k], "test_acc") for k in order]
    test_sem = [_sem(overall[k], "test_acc") for k in order]
    val_acc = [_mean(overall[k], "validation_acc") for k in order]
    gap = [_mean(overall[k], "validation_test_gap") for k in order]
    gap_sem = [_sem(overall[k], "validation_test_gap") for k in order]
    seconds = [_mean(overall[k], "elapsed_seconds") for k in order]
    output_fits = [_mean(overall[k], "output_fits_per_generation") for k in order]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.34

    axes[0].bar(x - width / 2, val_acc, width, color=LIGHT_BLUE, label="Validation")
    axes[0].bar(
        x + width / 2,
        test_acc,
        width,
        yerr=test_sem,
        capsize=3,
        color=BLUE,
        label="Test",
    )
    axes[0].set_title("Validation vs test accuracy")
    axes[0].set_ylim(0.76, 0.98)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(x, gap, yerr=gap_sem, capsize=3, color=colors, edgecolor="white")
    axes[1].set_title("Validation-test gap")
    axes[1].set_ylabel("Accuracy gap")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(gap):
        axes[1].text(idx, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)

    sizes = [70 + 6.0 * fits for fits in output_fits]
    axes[2].scatter(seconds, test_acc, s=sizes, color=colors, edgecolor=DARK, linewidth=0.8)
    for idx, (label, sx, sy, fits) in enumerate(zip(labels, seconds, test_acc, output_fits)):
        offset = (-118, 4) if idx == len(labels) - 1 else (8, 4)
        axes[2].annotate(
            f"{label.replace(chr(10), ' ')}\n{fits:.0f} output fits/gen",
            (sx, sy),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
        )
    axes[2].set_title("Accuracy-time trade-off")
    axes[2].set_xlabel("Fit time (s)")
    axes[2].set_ylabel("Test accuracy")
    axes[2].grid(True, alpha=0.25)
    axes[2].set_xlim(0.0, max(seconds) * 1.22)
    axes[2].set_ylim(0.78, 0.91)

    fig.suptitle("Matched-budget layer-level control")
    _save(fig, "michigan_pittsburgh_summary.png")


def save_full_network_pittsburgh_summary() -> None:
    payload = _load_json("runs/full_network_pittsburgh/full_network_pittsburgh_summary.json")
    overall = payload["summary"]["overall"]
    names = ["michigan_cfs", "full_pittsburgh_large"]
    labels = ["Michigan\nES-CFS", "Pittsburgh"]
    arch_colors = [BLUE, ORANGE]
    combined = {"michigan_cfs": payload["reference_michigan"]["summary"], **overall}

    val_acc = [_mean(combined[name], "validation_acc") for name in names]
    test_acc = [_mean(combined[name], "test_acc") for name in names]
    test_sem = [_sem(combined[name], "test_acc") for name in names]
    elapsed = [_mean(combined[name], "elapsed_seconds") for name in names]
    elapsed_sem = [_sem(combined[name], "elapsed_seconds") for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    x = np.arange(len(names))
    width = 0.34

    metric_colors = {"Validation": LIGHT_BLUE, "Test": BLUE}
    axes[0].bar(
        x - width / 2,
        val_acc,
        width,
        color=metric_colors["Validation"],
        label="Validation",
    )
    axes[0].bar(
        x + width / 2,
        test_acc,
        width,
        yerr=test_sem,
        capsize=3,
        color=metric_colors["Test"],
        label="Test",
    )
    axes[0].set_title("Validation vs test accuracy")
    axes[0].set_ylabel("Accuracy")
    acc_min = min(test_acc + val_acc)
    acc_max = max(test_acc + val_acc)
    axes[0].set_ylim(max(0.0, acc_min - 0.07), min(1.02, acc_max + 0.035))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    for idx, value in enumerate(test_acc):
        axes[0].text(
            idx + width / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )

    axes[1].bar(
        x,
        elapsed,
        yerr=elapsed_sem,
        capsize=3,
        color=arch_colors,
        edgecolor="white",
    )
    axes[1].set_title("Mean training time")
    axes[1].set_ylabel("Seconds")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(elapsed):
        axes[1].text(idx, value + max(elapsed) * 0.025, f"{value:.2f}s", ha="center", fontsize=8)

    full_param_values = [
        _mean(combined[name], "new_parameters_per_generation")
        for name in names
        if name != "michigan_cfs"
    ]
    max_full_params = max(full_param_values) if full_param_values else 1.0
    sizes = []
    for name in names:
        if name == "michigan_cfs":
            sizes.append(120.0)
        else:
            params = _mean(combined[name], "new_parameters_per_generation")
            sizes.append(100.0 + 330.0 * params / max_full_params)
    axes[2].scatter(
        elapsed,
        test_acc,
        s=sizes,
        color=arch_colors,
        edgecolor=DARK,
        linewidth=0.8,
    )
    for label, sx, sy in zip(labels, elapsed, test_acc):
        axes[2].annotate(
            label.replace("\n", " "),
            (sx, sy),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
        )
    axes[2].set_title("Accuracy-time trade-off")
    axes[2].set_xlabel("Mean training time (s)")
    axes[2].set_ylabel("Test accuracy")
    test_min = min(test_acc)
    test_max = max(test_acc)
    axes[2].set_ylim(max(0.0, test_min - 0.05), min(1.02, test_max + 0.035))
    axes[2].set_xlim(0.0, max(elapsed) * 1.22)
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Michigan vs Pittsburgh: full-network representation")
    _save(fig, "full_network_pittsburgh_summary.png")


def main() -> None:
    save_binary_benchmark_summary()
    save_strategy_cost_summary()
    save_regression_model_comparison()
    save_regression_value_summary()
    save_softmax_model_comparison()
    save_michigan_pittsburgh_summary()
    save_full_network_pittsburgh_summary()
    print("Generated final thesis plots in runs/ and plots_overleaf/.")


if __name__ == "__main__":
    main()
