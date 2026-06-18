"""
benchmark_full_network_pittsburgh.py

Full-network Pittsburgh benchmark for the binary logistic architecture.

This experiment is the main Michigan-Pittsburgh comparison used in the final
memory.  The Pittsburgh individual encodes the complete one-hidden-layer
network, including input-to-hidden weights, hidden biases, output weights, and
output bias.

The full-network Pittsburgh model is evaluated directly.  No output layer is
fitted after decoding the individual.  This makes the experiment a useful
benchmark for the thesis question: can Michigan obtain a better
accuracy-time trade-off by evolving hidden neurons and fitting the output
layer separately, compared with evolving all network parameters at once?
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".font_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_level1 import (
    DATASET_PRESETS,
    DatasetSplit,
    _binary_metrics,
    _json_safe,
    _mean_sem,
)
from data import load_dataset_with_validation
from logistic_output import binary_log_loss_from_proba, labels_to_binary, sigmoid


BLUE = "#0072B2"
ORANGE = "#E69F00"
LIGHT_ORANGE = "#F5D7A1"
DARK = "#2B2B2B"
GREY = "#7F7F7F"


@dataclass(frozen=True)
class FullNetworkConfig:
    name: str
    label: str
    parents: int
    offspring: int
    hidden_units: int
    epochs: int
    sigma_init: float
    sigma_min: float = 1e-4
    sigma_max: float = 1.0


def _load_split(
    dataset: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> DatasetSplit:
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_dataset_with_validation(
        data_id=int(dataset["id"]),
        data_name=dataset["name"],
        version=args.data_version,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=seed,
        data_home=args.data_home,
    )
    return DatasetSplit(X_tr, X_val, X_te, y_tr, y_val, y_te)


def _theta_dim(hidden_units: int, input_size: int) -> int:
    hidden_params = hidden_units * (input_size + 1)
    output_params = hidden_units + 1
    return hidden_params + output_params


def _initial_population(
    rng: np.random.Generator,
    config: FullNetworkConfig,
    input_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    h = config.hidden_units
    d = input_size + 1
    dim = _theta_dim(h, input_size)

    theta = np.zeros((config.parents, dim), dtype=np.float64)
    hidden_end = h * d
    bound = 1.0 / np.sqrt(input_size)
    theta[:, :hidden_end] = rng.uniform(
        low=-bound,
        high=bound,
        size=(config.parents, hidden_end),
    )
    theta[:, hidden_end:] = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(config.parents, h + 1),
    )
    sigma = np.full_like(theta, config.sigma_init, dtype=np.float64)
    return theta, sigma


def _mutate_population(
    rng: np.random.Generator,
    theta_pop: np.ndarray,
    sigma_pop: np.ndarray,
    config: FullNetworkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    pop_size, dim = theta_pop.shape
    parent_idx = rng.integers(0, pop_size, size=config.offspring)
    theta = theta_pop[parent_idx].copy()
    sigma = sigma_pop[parent_idx].copy()

    tau_global = 1.0 / np.sqrt(2.0 * dim)
    tau_local = 1.0 / np.sqrt(2.0 * np.sqrt(dim))
    global_noise = rng.standard_normal((config.offspring, 1))
    local_noise = rng.standard_normal((config.offspring, dim))
    sigma *= np.exp(tau_global * global_noise + tau_local * local_noise)
    sigma = np.clip(sigma, config.sigma_min, config.sigma_max)
    theta += sigma * rng.standard_normal((config.offspring, dim))
    return theta.astype(np.float64), sigma.astype(np.float64)


def _unpack_theta(
    theta: np.ndarray,
    hidden_units: int,
    input_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta2 = np.asarray(theta, dtype=np.float64)
    if theta2.ndim == 1:
        theta2 = theta2[None, :]
    h = hidden_units
    d = input_size + 1
    hidden_end = h * d
    alpha = theta2[:, :hidden_end].reshape(theta2.shape[0], h, d)
    beta = theta2[:, hidden_end : hidden_end + h]
    bias = theta2[:, hidden_end + h]
    return alpha, beta, bias


def _predict_pool_proba(
    theta: np.ndarray,
    X: np.ndarray,
    hidden_units: int,
) -> np.ndarray:
    input_size = X.shape[1]
    alpha, beta, bias = _unpack_theta(theta, hidden_units, input_size)
    W = alpha[:, :, :input_size]
    b = alpha[:, :, input_size]
    hidden = np.tanh(np.einsum("mn,khn->kmh", X, W) + b[:, None, :])
    logits = np.einsum("kmh,kh->km", hidden, beta) + bias[:, None]
    return sigmoid(logits)


def _pool_acc_ce(
    theta: np.ndarray,
    X: np.ndarray,
    y_pm1: np.ndarray,
    hidden_units: int,
) -> tuple[np.ndarray, np.ndarray]:
    y01 = labels_to_binary(y_pm1).reshape(-1)
    proba = np.clip(_predict_pool_proba(theta, X, hidden_units), 1e-12, 1.0 - 1e-12)
    pred = np.where(proba >= 0.5, 1.0, -1.0)
    acc = np.mean(pred == y_pm1.reshape(1, -1), axis=1)
    ce = -np.mean(
        y01.reshape(1, -1) * np.log(proba)
        + (1.0 - y01.reshape(1, -1)) * np.log(1.0 - proba),
        axis=1,
    )
    return acc.astype(np.float64), ce.astype(np.float64)


def _single_metrics(
    theta: np.ndarray,
    split: DatasetSplit,
    hidden_units: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, X, y in (
        ("train", split.X_tr, split.y_tr),
        ("validation", split.X_val, split.y_val),
        ("test", split.X_te, split.y_te),
    ):
        proba = _predict_pool_proba(theta.reshape(1, -1), X, hidden_units).reshape(-1)
        pred = np.where(proba >= 0.5, 1.0, -1.0)
        out[name] = _binary_metrics(y, proba, pred)
        out[name]["ce"] = binary_log_loss_from_proba(labels_to_binary(y), proba)
    return out


def _is_better(
    val_acc: float,
    val_ce: float,
    best: dict[str, Any] | None,
) -> bool:
    if best is None:
        return True
    best_acc = float(best["validation"]["acc"])
    best_ce = float(best["validation"]["ce"])
    return val_acc > best_acc + 1e-12 or (
        abs(val_acc - best_acc) <= 1e-12 and val_ce < best_ce
    )


def train_full_network_pittsburgh(
    split: DatasetSplit,
    config: FullNetworkConfig,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    input_size = split.X_tr.shape[1]
    theta_pop, sigma_pop = _initial_population(rng, config, input_size)

    history: list[dict[str, float | int]] = []
    best_checkpoint: dict[str, Any] | None = None

    for epoch in range(1, config.epochs + 1):
        epoch_t0 = time.perf_counter()
        off_theta, off_sigma = _mutate_population(rng, theta_pop, sigma_pop, config)
        pool_theta = np.vstack([theta_pop, off_theta])
        pool_sigma = np.vstack([sigma_pop, off_sigma])

        train_acc, train_ce = _pool_acc_ce(
            pool_theta,
            split.X_tr,
            split.y_tr,
            config.hidden_units,
        )
        val_acc, val_ce = _pool_acc_ce(
            pool_theta,
            split.X_val,
            split.y_val,
            config.hidden_units,
        )

        for idx in range(pool_theta.shape[0]):
            if _is_better(float(val_acc[idx]), float(val_ce[idx]), best_checkpoint):
                metrics = _single_metrics(pool_theta[idx], split, config.hidden_units)
                best_checkpoint = {
                    "epoch": epoch,
                    "theta": pool_theta[idx].copy(),
                    "sigma": pool_sigma[idx].copy(),
                    "train": metrics["train"],
                    "validation": metrics["validation"],
                }

        selected_idx = np.lexsort((val_ce, -val_acc))[: config.parents]
        theta_pop = pool_theta[selected_idx].copy()
        sigma_pop = pool_sigma[selected_idx].copy()
        best_idx = int(selected_idx[0])

        history.append(
            {
                "epoch": epoch,
                "train_acc": float(train_acc[best_idx]),
                "validation_acc": float(val_acc[best_idx]),
                "train_ce": float(train_ce[best_idx]),
                "validation_ce": float(val_ce[best_idx]),
                "epoch_ms": float((time.perf_counter() - epoch_t0) * 1e3),
                "evaluated_networks": int(pool_theta.shape[0]),
            }
        )

    if best_checkpoint is None:
        raise RuntimeError("Full-network Pittsburgh produced no checkpoint.")

    final = _single_metrics(
        np.asarray(best_checkpoint["theta"], dtype=np.float64),
        split,
        config.hidden_units,
    )
    elapsed = time.perf_counter() - t0
    dim = _theta_dim(config.hidden_units, input_size)
    evaluated_networks = config.parents + config.offspring
    metrics = {
        "architecture": config.name,
        "label": config.label,
        "kind": "full_network_pittsburgh",
        "hidden_units": config.hidden_units,
        "epochs": config.epochs,
        "selected_epoch": int(best_checkpoint["epoch"]),
        "train_acc": final["train"]["acc"],
        "validation_acc": final["validation"]["acc"],
        "test_acc": final["test"]["acc"],
        "validation_test_gap": float(final["validation"]["acc"] - final["test"]["acc"]),
        "train_ce": final["train"]["ce"],
        "validation_ce": final["validation"]["ce"],
        "test_ce": final["test"]["ce"],
        "train_brier": final["train"]["brier"],
        "validation_brier": final["validation"]["brier"],
        "test_brier": final["test"]["brier"],
        "train_ece": final["train"]["ece"],
        "validation_ece": final["validation"]["ece"],
        "test_ece": final["test"]["ece"],
        "train_roc_auc": final["train"]["roc_auc"],
        "validation_roc_auc": final["validation"]["roc_auc"],
        "test_roc_auc": final["test"]["roc_auc"],
        "elapsed_seconds": float(elapsed),
        "parents": int(config.parents),
        "offspring": int(config.offspring),
        "network_parameters": int(dim),
        "evaluated_networks_per_generation": int(evaluated_networks),
        "new_networks_per_generation": int(config.offspring),
        "new_parameters_per_generation": int(config.offspring * dim),
        "output_fits_per_generation": 0,
    }
    return {"metrics": metrics, "history": history}


def _configs(args: argparse.Namespace) -> list[FullNetworkConfig]:
    return [
        FullNetworkConfig(
            name="full_pittsburgh_large",
            label="Pittsburgh ES",
            parents=args.full_large_parents,
            offspring=args.full_large_offspring,
            hidden_units=args.hidden_units,
            epochs=args.epochs,
            sigma_init=args.full_sigma_init,
        ),
    ]


def _selected_datasets(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [DATASET_PRESETS[name] for name in args.datasets]


def _summarise_runs(
    runs: list[dict[str, Any]],
    configs: list[FullNetworkConfig],
) -> dict[str, Any]:
    metric_names = (
        "train_acc",
        "validation_acc",
        "test_acc",
        "validation_test_gap",
        "test_ce",
        "test_brier",
        "test_ece",
        "test_roc_auc",
        "elapsed_seconds",
        "selected_epoch",
        "network_parameters",
        "evaluated_networks_per_generation",
        "new_networks_per_generation",
        "new_parameters_per_generation",
        "output_fits_per_generation",
    )
    dataset_labels = sorted({str(run["dataset"]["label"]) for run in runs})
    summary: dict[str, Any] = {"overall": {}}
    for dataset_label in dataset_labels:
        summary[dataset_label] = {}
        dataset_runs = [
            run for run in runs if str(run["dataset"]["label"]) == dataset_label
        ]
        for config in configs:
            arch_runs = [
                run for run in dataset_runs if run["architecture"] == config.name
            ]
            summary[dataset_label][config.name] = _summarise_arch_runs(
                arch_runs,
                metric_names,
            )
    for config in configs:
        arch_runs = [run for run in runs if run["architecture"] == config.name]
        summary["overall"][config.name] = _summarise_arch_runs(
            arch_runs,
            metric_names,
        )
    return summary


def _summarise_arch_runs(
    arch_runs: list[dict[str, Any]],
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_runs": len(arch_runs),
        "label": arch_runs[0]["label"] if arch_runs else "",
    }
    for metric in metric_names:
        values = [
            float(run["metrics"][metric])
            for run in arch_runs
            if run["metrics"].get(metric) is not None
            and np.isfinite(float(run["metrics"][metric]))
        ]
        if values:
            out[metric] = _mean_sem(values)
    if (
        "test_acc" in out
        and "elapsed_seconds" in out
        and float(out["elapsed_seconds"]["mean"]) > 0.0
    ):
        out["accuracy_per_second"] = {
            "mean": float(out["test_acc"]["mean"] / out["elapsed_seconds"]["mean"]),
            "std": 0.0,
            "sem": 0.0,
        }
    return out


def _load_reference_michigan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    michigan = payload["summary"]["overall"]["michigan_cfs"]
    return {
        "label": "Michigan ES-CFS",
        "source": str(path),
        "summary": michigan,
        "config": payload["config"],
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    configs = _configs(args)
    runs: list[dict[str, Any]] = []

    for dataset in _selected_datasets(args):
        for seed in args.seeds:
            print(f"\n=== Dataset={dataset['label']} | seed={seed} ===")
            split = _load_split(dataset, args, seed)
            for config in configs:
                print(f"  -> Training {config.label}")
                result = train_full_network_pittsburgh(split, config, seed)
                runs.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "architecture": config.name,
                        "label": config.label,
                        "metrics": result["metrics"],
                        "history": result["history"],
                    }
                )
                metrics = result["metrics"]
                print(
                    "     "
                    f"val={float(metrics['validation_acc']):.4f} "
                    f"test={float(metrics['test_acc']):.4f} "
                    f"auc={float(metrics['test_roc_auc']):.4f} "
                    f"time={float(metrics['elapsed_seconds']):.2f}s"
                )

    payload = {
        "config": {
            "datasets": args.datasets,
            "seeds": [int(seed) for seed in args.seeds],
            "epochs": int(args.epochs),
            "hidden_units": int(args.hidden_units),
            "full_medium_parents": int(args.full_medium_parents),
            "full_medium_offspring": int(args.full_medium_offspring),
            "full_large_parents": int(args.full_large_parents),
            "full_large_offspring": int(args.full_large_offspring),
            "full_sigma_init": float(args.full_sigma_init),
            "validation_size": float(args.validation_size),
            "test_size": float(args.test_size),
        },
        "runs": runs,
    }
    payload["summary"] = _summarise_runs(runs, configs)
    if args.reference_michigan_summary:
        ref_path = Path(args.reference_michigan_summary)
        if ref_path.exists():
            payload["reference_michigan"] = _load_reference_michigan(ref_path)
    return payload


def _metric(summary: dict[str, Any], metric: str, field: str = "mean") -> float:
    return float(summary[metric][field])


def save_plots(payload: dict[str, Any], output_dir: Path, plots_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    overall = payload["summary"]["overall"]
    names = (
        ["full_pittsburgh_large"]
        if "full_pittsburgh_large" in overall
        else list(overall.keys())
    )
    labels = ["Pittsburgh"] if names == ["full_pittsburgh_large"] else [
        overall[name]["label"].replace(" ", "\n", 2) for name in names
    ]
    arch_colors = [ORANGE] if names == ["full_pittsburgh_large"] else [ORANGE, LIGHT_ORANGE]

    if "reference_michigan" in payload:
        names = ["michigan_cfs"] + names
        labels = ["Michigan\nES-CFS"] + labels
        arch_colors = [BLUE] + arch_colors
        combined = {"michigan_cfs": payload["reference_michigan"]["summary"], **overall}
    else:
        combined = overall

    test_acc = [_metric(combined[name], "test_acc") for name in names]
    test_sem = [_metric(combined[name], "test_acc", "sem") for name in names]
    val_acc = [_metric(combined[name], "validation_acc") for name in names]
    elapsed = [_metric(combined[name], "elapsed_seconds") for name in names]
    elapsed_sem = [_metric(combined[name], "elapsed_seconds", "sem") for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    x = np.arange(len(names))
    width = 0.34

    metric_colors = {"Validation": "#D6EAF5", "Test": BLUE}
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

    sizes = []
    for name in names:
        if name == "michigan_cfs":
            sizes.append(110.0)
        else:
            params = _metric(combined[name], "new_parameters_per_generation")
            max_params = max(
                _metric(combined[n], "new_parameters_per_generation")
                for n in names
                if n != "michigan_cfs"
            )
            sizes.append(100.0 + params / max(1.0, max_params) * 330.0)
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

    fig.suptitle("Michigan vs full-network Pittsburgh")
    for directory in (output_dir, plots_dir):
        fig.savefig(directory / "full_network_pittsburgh_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_json(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "full_network_pittsburgh_summary.json").open(
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(_json_safe(payload), fh, indent=2, sort_keys=True)


def print_summary(payload: dict[str, Any]) -> None:
    rows: list[tuple[str, dict[str, Any]]] = []
    if "reference_michigan" in payload:
        rows.append(("Michigan ES-CFS", payload["reference_michigan"]["summary"]))
    for name, summary in payload["summary"]["overall"].items():
        rows.append((summary["label"], summary))

    print("\n## Full-network Pittsburgh Benchmark")
    print("| Architecture | Val Acc | Test Acc | Test AUC | Fit s | New params/gen |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, row in rows:
        new_params = row.get("new_parameters_per_generation", {"mean": 0.0})
        print(
            f"| {label} | "
            f"{row['validation_acc']['mean']:.4f} +/- {row['validation_acc']['sem']:.4f} | "
            f"{row['test_acc']['mean']:.4f} +/- {row['test_acc']['sem']:.4f} | "
            f"{row['test_roc_auc']['mean']:.4f} +/- {row['test_roc_auc']['sem']:.4f} | "
            f"{row['elapsed_seconds']['mean']:.2f} +/- {row['elapsed_seconds']['sem']:.2f} | "
            f"{new_params['mean']:.0f} |"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-network Pittsburgh benchmark for the binary logistic architecture."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_PRESETS.keys()),
        default=["breast-w", "ionosphere", "sonar"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-units", type=int, default=50)
    parser.add_argument("--full-medium-parents", type=int, default=40)
    parser.add_argument("--full-medium-offspring", type=int, default=160)
    parser.add_argument("--full-large-parents", type=int, default=80)
    parser.add_argument("--full-large-offspring", type=int, default=320)
    parser.add_argument("--full-sigma-init", type=float, default=0.08)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-version", type=int, default=1)
    parser.add_argument("--data-home", default=".openml_cache")
    parser.add_argument("--output-dir", default="runs/full_network_pittsburgh")
    parser.add_argument("--plots-dir", default="plots_overleaf")
    parser.add_argument(
        "--reference-michigan-summary",
        default="runs/michigan_pittsburgh/michigan_pittsburgh_summary.json",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    payload = run_benchmark(args)
    save_json(payload, output_dir)
    save_plots(payload, output_dir, plots_dir)
    print_summary(payload)
    print(f"\nSaved artefacts under {output_dir}")
    print(f"Saved thesis plot under {plots_dir}")


if __name__ == "__main__":
    main()
