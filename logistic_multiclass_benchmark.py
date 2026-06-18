"""
logistic_multiclass_benchmark.py — Softmax/CE benchmark for the ES prototype.

This script tests the probabilistic architecture on multiclass classification:
hidden neurons are still evolved with a Michigan-style ES, but the output layer
is trained as an unregularised softmax classifier with cross-entropy.

Default experiment:
  - Digits: real 10-class image/tabular dataset;
  - Gaussian quantiles: controlled nonlinear 3-class dataset.
  - Spiral-3: controlled nonlinear 3-class dataset with intertwined classes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".font_cache").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine, make_gaussian_quantiles
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from core_es import MichiganESState, compute_activations, mutate_population
from logistic_output import compute_softmax_output, predict_softmax_proba
from selection import (
    diversity_penalised_selection,
    multiclass_cfs_selection,
    multiclass_omp_residual_selection,
    multiclass_r2_fitness,
)


@dataclass
class MulticlassSplit:
    dataset: str
    X_tr: np.ndarray
    X_val: np.ndarray
    X_te: np.ndarray
    y_tr: np.ndarray
    y_val: np.ndarray
    y_te: np.ndarray
    n_classes: int


def make_spiral_dataset(
    n_samples: int,
    n_classes: int,
    seed: int,
    noise: float = 0.28,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a compact 2-D multiclass spiral classification problem."""
    rng = np.random.default_rng(seed)
    per_class = n_samples // n_classes
    remainder = n_samples - per_class * n_classes
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for cls in range(n_classes):
        n_cls = per_class + (1 if cls < remainder else 0)
        r = np.linspace(0.08, 1.0, n_cls)
        theta = 2.4 * np.pi * r + cls * (2.0 * np.pi / n_classes)
        theta += rng.normal(0.0, noise, size=n_cls)
        radius = r + rng.normal(0.0, noise * 0.08, size=n_cls)
        X_cls = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
        X_cls += rng.normal(0.0, noise * 0.05, size=X_cls.shape)
        X_parts.append(X_cls)
        y_parts.append(np.full(n_cls, cls, dtype=int))
    X = np.vstack(X_parts).astype(np.float64)
    y = np.concatenate(y_parts)
    order = rng.permutation(len(y))
    return X[order], y[order]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_multiclass_split(
    dataset: str,
    seed: int,
    max_samples: int,
    validation_size: float,
    test_size: float,
) -> MulticlassSplit:
    if dataset == "wine":
        raw = load_wine()
        X = raw.data.astype(np.float64)
        y = raw.target
        label = "Wine"
    elif dataset == "iris":
        raw = load_iris()
        X = raw.data.astype(np.float64)
        y = raw.target
        label = "Iris"
    elif dataset == "digits":
        raw = load_digits()
        X = raw.data.astype(np.float64)
        y = raw.target
        label = "Digits"
    elif dataset == "gaussian-quantiles":
        X, y = make_gaussian_quantiles(
            mean=None,
            cov=3.0,
            n_samples=max_samples,
            n_features=2,
            n_classes=3,
            random_state=seed,
        )
        label = f"GaussianQuantiles-{max_samples}"
    elif dataset == "spiral-3":
        X, y = make_spiral_dataset(
            n_samples=max_samples,
            n_classes=3,
            seed=seed,
        )
        label = f"Spiral3-{max_samples}"
    else:
        raise ValueError(f"Unknown multiclass dataset: {dataset!r}")

    encoder = LabelEncoder()
    y_idx = encoder.fit_transform(y).astype(int)
    n_classes = int(len(encoder.classes_))

    X_trainval_raw, X_te_raw, y_trainval, y_te = train_test_split(
        X,
        y_idx,
        test_size=test_size,
        random_state=seed,
        stratify=y_idx,
    )
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_trainval_raw,
        y_trainval,
        test_size=validation_size,
        random_state=seed + 1,
        stratify=y_trainval,
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw).astype(np.float64)
    X_val = scaler.transform(X_val_raw).astype(np.float64)
    X_te = scaler.transform(X_te_raw).astype(np.float64)

    return MulticlassSplit(
        dataset=label,
        X_tr=X_tr,
        X_val=X_val,
        X_te=X_te,
        y_tr=y_tr.astype(int),
        y_val=y_val.astype(int),
        y_te=y_te.astype(int),
        n_classes=n_classes,
    )


def _linear_design(X: np.ndarray) -> np.ndarray:
    return np.c_[X.astype(np.float64), np.ones(X.shape[0], dtype=np.float64)]


def _random_tanh_designs(
    split: MulticlassSplit,
    hidden_units: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    input_dim = split.X_tr.shape[1]
    bound = 1.0 / np.sqrt(input_dim)
    alpha = rng.uniform(
        low=-bound,
        high=bound,
        size=(hidden_units, input_dim + 1),
    ).astype(np.float64)
    Phi_tr = np.c_[compute_activations(alpha, split.X_tr), np.ones(split.X_tr.shape[0])]
    Phi_val = np.c_[compute_activations(alpha, split.X_val), np.ones(split.X_val.shape[0])]
    Phi_te = np.c_[compute_activations(alpha, split.X_te), np.ones(split.X_te.shape[0])]
    return Phi_tr, Phi_val, Phi_te


def multiclass_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(proba, dtype=np.float64)
    confidence = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return float(ece)


def classification_metrics(
    split: MulticlassSplit,
    val_proba: np.ndarray,
    test_proba: np.ndarray,
) -> dict[str, float]:
    labels = np.arange(split.n_classes)
    val_pred = np.argmax(val_proba, axis=1)
    test_pred = np.argmax(test_proba, axis=1)
    return {
        "validation_accuracy": float(accuracy_score(split.y_val, val_pred)),
        "test_accuracy": float(accuracy_score(split.y_te, test_pred)),
        "test_macro_f1": float(f1_score(split.y_te, test_pred, average="macro")),
        "validation_log_loss": float(log_loss(split.y_val, val_proba, labels=labels)),
        "test_log_loss": float(log_loss(split.y_te, test_proba, labels=labels)),
        "test_ece": multiclass_ece(split.y_te, test_proba),
    }


def _is_better(row: dict[str, object], best: dict[str, object] | None) -> bool:
    if best is None:
        return True
    acc = float(row["validation_accuracy"])
    best_acc = float(best["validation_accuracy"])
    if acc > best_acc + 1e-12:
        return True
    if abs(acc - best_acc) <= 1e-12:
        return float(row["validation_log_loss"]) < float(best["validation_log_loss"])
    return False


def fit_softmax_design(
    split: MulticlassSplit,
    model_name: str,
    Phi_tr: np.ndarray,
    Phi_val: np.ndarray,
    Phi_te: np.ndarray,
    output_max_iter: int,
    extra: dict[str, object] | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    t0 = time.perf_counter()
    beta, train_ce, n_iter = compute_softmax_output(
        Phi_tr,
        split.y_tr,
        n_classes=split.n_classes,
        max_iter=output_max_iter,
    )
    fit_seconds = time.perf_counter() - t0
    val_proba = predict_softmax_proba(Phi_val, beta)
    test_proba = predict_softmax_proba(Phi_te, beta)
    row: dict[str, object] = {
        "model": model_name,
        "family": "softmax",
        "train_log_loss": float(train_ce),
        "fit_seconds": float(fit_seconds),
        "output_iterations": int(n_iter),
    }
    if extra:
        row.update(extra)
    row.update(classification_metrics(split, val_proba, test_proba))
    return row, test_proba


def fit_random_tanh_softmax(
    split: MulticlassSplit,
    hidden_grid: list[int],
    seed: int,
    output_max_iter: int,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    best: dict[str, object] | None = None
    best_proba: np.ndarray | None = None
    candidates: list[dict[str, object]] = []
    for hidden_units in hidden_grid:
        Phi_tr, Phi_val, Phi_te = _random_tanh_designs(
            split,
            hidden_units=hidden_units,
            seed=seed + hidden_units * 17,
        )
        row, proba = fit_softmax_design(
            split,
            "Random tanh Softmax CE",
            Phi_tr,
            Phi_val,
            Phi_te,
            output_max_iter=output_max_iter,
            extra={"hidden_units": int(hidden_units), "selected_by": "validation_accuracy_then_log_loss"},
        )
        candidates.append(row)
        if _is_better(row, best):
            best = row
            best_proba = proba
    if best is None or best_proba is None:
        raise RuntimeError("No random tanh softmax candidates were evaluated.")
    return dict(best), candidates, best_proba


def select_multiclass_neurons(
    Phi_pool: np.ndarray,
    y: np.ndarray,
    mu: int,
    strategy: str,
    gamma: float,
) -> np.ndarray:
    if strategy == "omp":
        return multiclass_omp_residual_selection(Phi_pool, y, mu=mu)
    if strategy == "r2":
        scores = multiclass_r2_fitness(Phi_pool, y)
        return diversity_penalised_selection(scores, Phi_pool, mu=mu, gamma=gamma)
    if strategy == "cfs":
        return multiclass_cfs_selection(Phi_pool, y, mu=mu)
    raise ValueError(f"Unknown multiclass strategy {strategy!r}.")


def fit_es_softmax(
    split: MulticlassSplit,
    strategy: str,
    epochs: int,
    mu: int,
    n_offspring: int,
    sigma_init: float,
    gamma: float,
    seed: int,
    output_max_iter: int,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    np.random.seed(seed)
    state = MichiganESState(mu=mu, input_size=split.X_tr.shape[1], sigma_init=sigma_init)
    best_checkpoint: dict[str, object] | None = None
    history: list[dict[str, object]] = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        off_alpha, off_L = mutate_population(state, n_offspring=n_offspring)
        pool_alpha = np.vstack([state.alpha, off_alpha])
        pool_L = np.vstack([state.L, off_L])
        Phi_pool = compute_activations(pool_alpha, split.X_tr)
        selected_idx = select_multiclass_neurons(
            Phi_pool,
            split.y_tr,
            mu=mu,
            strategy=strategy,
            gamma=gamma,
        )
        state.alpha = pool_alpha[selected_idx].copy()
        state.L = pool_L[selected_idx].copy()

        Phi_tr = np.c_[Phi_pool[:, selected_idx], np.ones(split.X_tr.shape[0])]
        beta, train_ce, n_iter = compute_softmax_output(
            Phi_tr,
            split.y_tr,
            n_classes=split.n_classes,
            max_iter=output_max_iter,
        )
        Phi_val = np.c_[compute_activations(state.alpha, split.X_val), np.ones(split.X_val.shape[0])]
        Phi_te = np.c_[compute_activations(state.alpha, split.X_te), np.ones(split.X_te.shape[0])]
        val_proba = predict_softmax_proba(Phi_val, beta)
        test_proba = predict_softmax_proba(Phi_te, beta)
        metrics = classification_metrics(split, val_proba, test_proba)
        row = {
            "epoch": epoch,
            "train_log_loss": float(train_ce),
            "output_iterations": int(n_iter),
            **metrics,
        }
        history.append(row)
        if _is_better(row, best_checkpoint):
            best_checkpoint = {
                "epoch": epoch,
                "alpha": state.alpha.copy(),
                "beta": beta.copy(),
                "test_proba": test_proba.copy(),
                **row,
            }

    if best_checkpoint is None:
        raise RuntimeError("ES softmax did not produce a checkpoint.")

    result = {
        "model": f"ES-{strategy.upper()} Softmax CE",
        "family": "prototype",
        "strategy": strategy,
        "epochs": epochs,
        "hidden_units": mu,
        "offspring": n_offspring,
        "selected_epoch": int(best_checkpoint["epoch"]),
        "fit_seconds": float(time.perf_counter() - t0),
        "train_log_loss": best_checkpoint["train_log_loss"],
        "output_iterations": best_checkpoint["output_iterations"],
        "validation_accuracy": best_checkpoint["validation_accuracy"],
        "test_accuracy": best_checkpoint["test_accuracy"],
        "test_macro_f1": best_checkpoint["test_macro_f1"],
        "validation_log_loss": best_checkpoint["validation_log_loss"],
        "test_log_loss": best_checkpoint["test_log_loss"],
        "test_ece": best_checkpoint["test_ece"],
    }
    return result, history, np.asarray(best_checkpoint["test_proba"])


def fit_es_strategy_sweep(
    split: MulticlassSplit,
    strategies: list[str],
    epochs: int,
    mu: int,
    n_offspring: int,
    sigma_init: float,
    gamma: float,
    seed: int,
    output_max_iter: int,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]], list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, object]]] = {}
    predictions: dict[str, np.ndarray] = {}
    for strategy in strategies:
        row, history, proba = fit_es_softmax(
            split,
            strategy=strategy,
            epochs=epochs,
            mu=mu,
            n_offspring=n_offspring,
            sigma_init=sigma_init,
            gamma=gamma,
            seed=seed,
            output_max_iter=output_max_iter,
        )
        rows.append(row)
        histories[str(row["model"])] = history
        predictions[str(row["model"])] = proba

    best = None
    for row in rows:
        if _is_better(row, best):
            best = row
    if best is None:
        raise RuntimeError("No ES strategies were evaluated.")

    selected = dict(best)
    best_name = str(best["model"])
    if len(strategies) == 1:
        selected["model"] = f"ES-{str(best['strategy']).upper()} Softmax CE"
    else:
        selected["model"] = "ES-selected Softmax CE"
    selected["selected_strategy"] = best["strategy"]
    selected["selection_rule"] = "highest validation_accuracy, then lowest validation_log_loss"
    selected["selected_strategy_fit_seconds"] = best["fit_seconds"]
    selected["fit_seconds"] = float(sum(float(row["fit_seconds"]) for row in rows))
    return selected, histories, rows, predictions[best_name]


def fit_hist_gradient_boosting(split: MulticlassSplit, seed: int) -> tuple[dict[str, object], np.ndarray]:
    model = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.06,
        l2_regularization=1e-4,
        random_state=seed,
    )
    t0 = time.perf_counter()
    model.fit(split.X_tr, split.y_tr)
    fit_seconds = time.perf_counter() - t0
    val_proba = model.predict_proba(split.X_val)
    test_proba = model.predict_proba(split.X_te)
    row: dict[str, object] = {
        "model": "HistGradientBoosting",
        "family": "sklearn",
        "fit_seconds": float(fit_seconds),
    }
    row.update(classification_metrics(split, val_proba, test_proba))
    return row, test_proba


def fit_gradient_mlp(
    split: MulticlassSplit,
    seed: int,
    max_iter: int,
) -> tuple[dict[str, object], np.ndarray]:
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12,
        random_state=seed,
        batch_size=min(256, max(1, int(split.X_tr.shape[0] * 0.8))),
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(split.X_tr, split.y_tr)
    fit_seconds = time.perf_counter() - t0
    val_proba = model.predict_proba(split.X_val)
    test_proba = model.predict_proba(split.X_te)
    row: dict[str, object] = {
        "model": "Gradient MLP",
        "family": "sklearn",
        "fit_seconds": float(fit_seconds),
        "iterations": int(model.n_iter_),
    }
    row.update(classification_metrics(split, val_proba, test_proba))
    return row, test_proba


def run_dataset(split: MulticlassSplit, args: argparse.Namespace, seed: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, object]]] = {}
    candidates: dict[str, list[dict[str, object]]] = {}
    predictions: dict[str, list[list[float]]] = {}

    row, proba = fit_softmax_design(
        split,
        "Linear Softmax CE",
        _linear_design(split.X_tr),
        _linear_design(split.X_val),
        _linear_design(split.X_te),
        output_max_iter=args.output_max_iter,
    )
    rows.append(row)
    predictions[row["model"]] = proba.tolist()

    row, random_candidates, proba = fit_random_tanh_softmax(
        split,
        hidden_grid=args.random_hidden_grid,
        seed=seed,
        output_max_iter=args.output_max_iter,
    )
    rows.append(row)
    candidates["Random tanh Softmax CE"] = random_candidates
    predictions[row["model"]] = proba.tolist()

    row, es_histories, es_strategy_rows, proba = fit_es_strategy_sweep(
        split,
        strategies=args.es_strategies,
        epochs=args.es_epochs,
        mu=args.es_mu,
        n_offspring=args.es_offspring,
        sigma_init=args.sigma_init,
        gamma=args.gamma,
        seed=seed,
        output_max_iter=args.output_max_iter,
    )
    rows.append(row)
    histories.update(es_histories)
    candidates["ES strategy sweep"] = es_strategy_rows
    predictions[row["model"]] = proba.tolist()

    row, proba = fit_hist_gradient_boosting(split, seed=seed)
    rows.append(row)
    predictions[row["model"]] = proba.tolist()

    row, proba = fit_gradient_mlp(split, seed=seed, max_iter=args.mlp_max_iter)
    rows.append(row)
    predictions[row["model"]] = proba.tolist()

    return {
        "dataset": split.dataset,
        "n_train": int(split.X_tr.shape[0]),
        "n_validation": int(split.X_val.shape[0]),
        "n_test": int(split.X_te.shape[0]),
        "n_features": int(split.X_tr.shape[1]),
        "n_classes": int(split.n_classes),
        "rows": rows,
        "histories": histories,
        "candidates": candidates,
        "y_test": split.y_te.tolist(),
        "predictions": predictions,
    }


def _model_order(results: list[dict[str, object]]) -> list[str]:
    available = {row["model"] for result in results for row in result["rows"]}
    preferred = [
        "Linear Softmax CE",
        "Random tanh Softmax CE",
        "ES-CFS Softmax CE",
        "ES-R2 Softmax CE",
        "ES-OMP Softmax CE",
        "ES-selected Softmax CE",
        "Gradient MLP",
        "HistGradientBoosting",
    ]
    models = [model for model in preferred if model in available]
    models.extend(sorted(available.difference(models)))
    return models


def save_model_comparison_plot(results: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prototype_models = [
        row["model"]
        for result in results
        for row in result["rows"]
        if str(row["model"]).startswith("ES-")
    ]
    prototype_name = sorted(set(prototype_models))[0]
    models = ["Linear Softmax CE", "Random tanh Softmax CE", prototype_name]
    labels = ["Linear\nSoftmax CE", "Random tanh\nSoftmax CE", "ES-CFS\nSoftmax CE"]
    values = []
    for model in models:
        acc = []
        for result in results:
            rows_by_model = {row["model"]: row for row in result["rows"]}
            acc.append(float(rows_by_model[model]["test_accuracy"]))
        values.append(float(np.mean(acc)))

    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    x = np.arange(len(models))
    colors = ["#7F7F7F", "#F5D7A1", "#0072B2"]
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
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_representation_value_plot(results: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prototype_models = [
        row["model"]
        for result in results
        for row in result["rows"]
        if str(row["model"]).startswith("ES-")
    ]
    prototype_name = sorted(set(prototype_models))[0]
    models = ["Linear Softmax CE", "Random tanh Softmax CE", prototype_name]
    datasets = [result["dataset"] for result in results]
    x = np.arange(len(datasets))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    colors = ["#7F7F7F", "#F5D7A1", "#0072B2"]
    for i, (offset, model) in enumerate(zip([-width, 0.0, width], models)):
        values = []
        for result in results:
            rows_by_model = {row["model"]: row for row in result["rows"]}
            values.append(float(rows_by_model[model]["test_accuracy"]))
        ax.bar(x + offset, values, width, label=model, color=colors[i], edgecolor="white")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Value of evolved hidden representation with softmax output")
    ax.set_ylabel("Test accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=10, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_latex_notes(paths: dict[str, Path], output_path: Path) -> None:
    captions = {
        "model_comparison": "Multiclass softmax benchmark comparing ES-selected Softmax CE against baselines.",
        "representation_value": "Multiclass softmax ablation showing the value of the evolved hidden representation.",
    }
    lines = [
        "% Generated by logistic_multiclass_benchmark.py.",
        "% Copy snippets manually into report.txt if desired.",
        "",
    ]
    for key, path in paths.items():
        lines.extend(
            [
                f"% Figure: {key}",
                "% \\begin{figure}[H]",
                "% \\centering",
                f"% \\includegraphics[width=0.95\\textwidth]{{{path.as_posix()}}}",
                f"% \\caption{{{captions[key]}}}",
                f"% \\label{{fig:softmax_{key}}}",
                "% \\end{figure}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def print_summary(results: list[dict[str, object]]) -> None:
    print("\n## Multiclass Softmax Benchmark")
    for result in results:
        print(f"\n### {result['dataset']}")
        print("| Model | Val Acc | Test Acc | Macro F1 | Test CE | Test ECE | Fit s |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for row in sorted(result["rows"], key=lambda item: float(item["test_accuracy"]), reverse=True):
            print(
                f"| {row['model']} | "
                f"{float(row['validation_accuracy']):.4f} | "
                f"{float(row['test_accuracy']):.4f} | "
                f"{float(row['test_macro_f1']):.4f} | "
                f"{float(row['test_log_loss']):.4f} | "
                f"{float(row['test_ece']):.4f} | "
                f"{float(row['fit_seconds']):.3f} |"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiclass softmax/CE benchmarks.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["wine", "iris", "digits", "gaussian-quantiles", "spiral-3"],
        default=["digits", "gaussian-quantiles", "spiral-3"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-hidden-grid", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument(
        "--es-strategies",
        nargs="+",
        choices=["omp", "r2", "cfs"],
        default=["omp", "cfs", "r2"],
    )
    parser.add_argument("--es-epochs", type=int, default=35)
    parser.add_argument("--es-mu", type=int, default=70)
    parser.add_argument("--es-offspring", type=int, default=210)
    parser.add_argument("--sigma-init", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--output-max-iter", type=int, default=160)
    parser.add_argument("--mlp-max-iter", type=int, default=160)
    parser.add_argument("--output-dir", default="runs/logistic_multiclass")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    results: list[dict[str, object]] = []
    for dataset in args.datasets:
        print(f"\nRunning multiclass dataset: {dataset}")
        split = load_multiclass_split(
            dataset,
            seed=args.seed,
            max_samples=args.max_samples,
            validation_size=args.validation_size,
            test_size=args.test_size,
        )
        results.append(run_dataset(split, args=args, seed=args.seed))

    paths = {
        "model_comparison": output_dir / "softmax_model_comparison.png",
        "representation_value": output_dir / "softmax_representation_value.png",
    }
    save_model_comparison_plot(results, paths["model_comparison"])
    save_representation_value_plot(results, paths["representation_value"])

    summary_path = output_dir / "summary.json"
    latex_notes_path = output_dir / "latex_notes.tex"
    payload = {
        "config": vars(args),
        "results": results,
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    save_json(payload, summary_path)
    write_latex_notes(paths, latex_notes_path)
    print_summary(results)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved LaTeX notes: {latex_notes_path}")
    for key, path in paths.items():
        print(f"Saved {key}: {path}")


if __name__ == "__main__":
    main()
