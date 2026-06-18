"""
ridge_regression_benchmark.py — Regression benchmark for the Ridge/GCV prototype.

This script adapts the current Ridge architecture to regression:
  - hidden neurons are evolved with the same Michigan-style ES representation;
  - neuron selection uses continuous-target strategies (OMP, R2, CFS);
  - the output layer is solved analytically with Ridge + GCV;
  - model selection uses validation RMSE;
  - final metrics are RMSE, MAE, R2, and fit time.

The experiment is intentionally compact and thesis-oriented. It compares:
  - Linear Ridge+GCV;
  - Random tanh Ridge+GCV;
  - ES-selected Ridge+GCV (the regression prototype, selected by validation RMSE);
  - HistGradientBoostingRegressor;
  - Gradient MLP.

Default datasets:
  - Higgs OpenML id 44092, treated as a numeric regression benchmark;
  - Friedman1 synthetic nonlinear regression;
  - Diabetes built-in sklearn regression.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml, load_diabetes, make_friedman1
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from core_es import MichiganESState, compute_activations, mutate_population
from gcv import compute_ridge_gcv
from selection import (
    cfs_selection,
    diversity_penalised_selection,
    omp_residual_selection,
    r2_fitness,
)


@dataclass
class RegressionSplit:
    dataset: str
    X_tr: np.ndarray
    X_val: np.ndarray
    X_te: np.ndarray
    y_tr: np.ndarray
    y_val: np.ndarray
    y_te: np.ndarray
    y_tr_scaled: np.ndarray
    y_mean: float
    y_std: float


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


def _maybe_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or X.shape[0] <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx], y[idx]


def _preprocess_regression(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    validation_size: float,
    test_size: float,
) -> RegressionSplit:
    stratify = None
    unique = np.unique(y)
    if 1 < len(unique) <= 20:
        _, counts = np.unique(y, return_counts=True)
        if np.all(counts >= 3):
            stratify = y

    X_trainval_raw, X_te_raw, y_trainval, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    stratify_trainval = None
    if stratify is not None:
        stratify_trainval = y_trainval
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_trainval_raw,
        y_trainval,
        test_size=validation_size,
        random_state=seed + 1,
        stratify=stratify_trainval,
    )

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_te_imp = imputer.transform(X_te_raw)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_imp).astype(np.float64)
    X_val = scaler.transform(X_val_imp).astype(np.float64)
    X_te = scaler.transform(X_te_imp).astype(np.float64)

    y_tr = y_tr.astype(np.float64).reshape(-1)
    y_val = y_val.astype(np.float64).reshape(-1)
    y_te = y_te.astype(np.float64).reshape(-1)
    y_mean = float(np.mean(y_tr))
    y_std = float(np.std(y_tr))
    if y_std < 1e-12:
        raise ValueError(f"Dataset {name} has near-zero training target variance.")

    return RegressionSplit(
        dataset=name,
        X_tr=X_tr,
        X_val=X_val,
        X_te=X_te,
        y_tr=y_tr,
        y_val=y_val,
        y_te=y_te,
        y_tr_scaled=((y_tr - y_mean) / y_std).astype(np.float64),
        y_mean=y_mean,
        y_std=y_std,
    )


def load_regression_split(
    dataset: str,
    seed: int,
    data_home: str,
    max_samples: int,
    validation_size: float,
    test_size: float,
) -> RegressionSplit:
    if dataset == "higgs":
        data = fetch_openml(
            data_id=44092,
            as_frame=False,
            parser="auto",
            data_home=data_home,
        )
        X = np.asarray(data.data, dtype=np.float64)
        y = np.asarray(data.target, dtype=np.float64).reshape(-1)
        X, y = _maybe_subsample(X, y, max_samples=max_samples, seed=seed)
        label = f"Higgs-{len(y)}"
    elif dataset == "friedman1":
        X, y = make_friedman1(
            n_samples=max_samples,
            n_features=20,
            noise=1.0,
            random_state=seed,
        )
        label = f"Friedman1-{max_samples}"
    elif dataset == "diabetes":
        data = load_diabetes()
        X = np.asarray(data.data, dtype=np.float64)
        y = np.asarray(data.target, dtype=np.float64).reshape(-1)
        label = "Diabetes"
    else:
        raise ValueError(f"Unknown regression dataset: {dataset!r}")

    return _preprocess_regression(
        label,
        X,
        y,
        seed=seed,
        validation_size=validation_size,
        test_size=test_size,
    )


def _linear_design(X: np.ndarray) -> np.ndarray:
    return np.c_[X.astype(np.float64), np.ones(X.shape[0], dtype=np.float64)]


def _random_tanh_designs(
    split: RegressionSplit,
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


def _inverse_target(split: RegressionSplit, y_scaled: np.ndarray) -> np.ndarray:
    return np.asarray(y_scaled).reshape(-1) * split.y_std + split.y_mean


def regression_metrics(
    split: RegressionSplit,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "validation_rmse": float(np.sqrt(mean_squared_error(split.y_val, val_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(split.y_te, test_pred))),
        "test_mae": float(mean_absolute_error(split.y_te, test_pred)),
        "test_r2": float(r2_score(split.y_te, test_pred)),
    }


def fit_ridge_design(
    split: RegressionSplit,
    model_name: str,
    Phi_tr: np.ndarray,
    Phi_val: np.ndarray,
    Phi_te: np.ndarray,
    extra: dict[str, object] | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    t0 = time.perf_counter()
    lam, beta, gcv = compute_ridge_gcv(Phi_tr, split.y_tr_scaled)
    fit_seconds = time.perf_counter() - t0
    val_pred = _inverse_target(split, Phi_val @ beta)
    test_pred = _inverse_target(split, Phi_te @ beta)
    row: dict[str, object] = {
        "model": model_name,
        "family": "ridge",
        "lambda": float(lam),
        "gcv": float(gcv),
        "fit_seconds": float(fit_seconds),
    }
    if extra:
        row.update(extra)
    row.update(regression_metrics(split, val_pred, test_pred))
    return row, test_pred


def fit_random_tanh_ridge(
    split: RegressionSplit,
    hidden_grid: list[int],
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    best: dict[str, object] | None = None
    best_pred: np.ndarray | None = None
    candidates: list[dict[str, object]] = []
    for hidden_units in hidden_grid:
        Phi_tr, Phi_val, Phi_te = _random_tanh_designs(
            split,
            hidden_units=hidden_units,
            seed=seed + hidden_units * 13,
        )
        row, pred = fit_ridge_design(
            split,
            "Random tanh Ridge+GCV",
            Phi_tr,
            Phi_val,
            Phi_te,
            extra={"hidden_units": int(hidden_units), "selected_by": "validation_rmse"},
        )
        candidates.append(row)
        if best is None or float(row["validation_rmse"]) < float(best["validation_rmse"]):
            best = row
            best_pred = pred
    if best is None or best_pred is None:
        raise RuntimeError("No random tanh candidates were evaluated.")
    return dict(best), candidates, best_pred


def select_regression_neurons(
    Phi_pool: np.ndarray,
    y_scaled: np.ndarray,
    mu: int,
    strategy: str,
    gamma: float,
) -> np.ndarray:
    if strategy == "omp":
        return omp_residual_selection(Phi_pool, y_scaled, mu=mu)
    if strategy == "r2":
        scores = r2_fitness(Phi_pool, y_scaled)
        return diversity_penalised_selection(scores, Phi_pool, mu=mu, gamma=gamma)
    if strategy == "cfs":
        return cfs_selection(Phi_pool, y_scaled, mu=mu)
    raise ValueError(f"Unknown regression strategy {strategy!r}.")


def fit_es_ridge_regression(
    split: RegressionSplit,
    strategy: str,
    epochs: int,
    mu: int,
    n_offspring: int,
    sigma_init: float,
    gamma: float,
    seed: int,
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
        selected_idx = select_regression_neurons(
            Phi_pool,
            split.y_tr_scaled,
            mu=mu,
            strategy=strategy,
            gamma=gamma,
        )
        state.alpha = pool_alpha[selected_idx].copy()
        state.L = pool_L[selected_idx].copy()
        Phi_tr = np.c_[Phi_pool[:, selected_idx], np.ones(split.X_tr.shape[0])]
        lam, beta, gcv = compute_ridge_gcv(Phi_tr, split.y_tr_scaled)
        state.beta = beta

        Phi_val = np.c_[compute_activations(state.alpha, split.X_val), np.ones(split.X_val.shape[0])]
        Phi_te = np.c_[compute_activations(state.alpha, split.X_te), np.ones(split.X_te.shape[0])]
        val_pred = _inverse_target(split, Phi_val @ beta)
        test_pred = _inverse_target(split, Phi_te @ beta)
        metrics = regression_metrics(split, val_pred, test_pred)
        row = {
            "epoch": epoch,
            "lambda": float(lam),
            "gcv": float(gcv),
            **metrics,
        }
        history.append(row)
        if (
            best_checkpoint is None
            or metrics["validation_rmse"] < float(best_checkpoint["validation_rmse"])
        ):
            best_checkpoint = {
                "epoch": epoch,
                "alpha": state.alpha.copy(),
                "beta": beta.copy(),
                "lambda": float(lam),
                "gcv": float(gcv),
                **metrics,
                "test_pred": test_pred.copy(),
            }

    if best_checkpoint is None:
        raise RuntimeError("ES regression did not produce a checkpoint.")

    fit_seconds = time.perf_counter() - t0
    result = {
        "model": f"ES-{strategy.upper()} Ridge+GCV",
        "family": "prototype",
        "strategy": strategy,
        "epochs": epochs,
        "hidden_units": mu,
        "offspring": n_offspring,
        "selected_epoch": int(best_checkpoint["epoch"]),
        "lambda": best_checkpoint["lambda"],
        "gcv": best_checkpoint["gcv"],
        "fit_seconds": float(fit_seconds),
        "validation_rmse": best_checkpoint["validation_rmse"],
        "test_rmse": best_checkpoint["test_rmse"],
        "test_mae": best_checkpoint["test_mae"],
        "test_r2": best_checkpoint["test_r2"],
    }
    return result, history, np.asarray(best_checkpoint["test_pred"])


def fit_es_strategy_sweep(
    split: RegressionSplit,
    strategies: list[str],
    epochs: int,
    mu: int,
    n_offspring: int,
    sigma_init: float,
    gamma: float,
    seed: int,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]], list[dict[str, object]], np.ndarray]:
    strategy_rows: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, object]]] = {}
    predictions: dict[str, np.ndarray] = {}

    for strategy in strategies:
        row, history, pred = fit_es_ridge_regression(
            split,
            strategy=strategy,
            epochs=epochs,
            mu=mu,
            n_offspring=n_offspring,
            sigma_init=sigma_init,
            gamma=gamma,
            seed=seed,
        )
        strategy_rows.append(row)
        histories[str(row["model"])] = history
        predictions[str(row["model"])] = pred

    best = min(strategy_rows, key=lambda item: float(item["validation_rmse"]))
    best_model_name = str(best["model"])
    selected = dict(best)
    selected["model"] = "ES-selected Ridge+GCV"
    selected["selected_strategy"] = best["strategy"]
    selected["selection_rule"] = "lowest validation_rmse among ES strategies"
    selected["selected_strategy_fit_seconds"] = best["fit_seconds"]
    selected["fit_seconds"] = float(sum(float(row["fit_seconds"]) for row in strategy_rows))
    return selected, histories, strategy_rows, predictions[best_model_name]


def fit_hist_gradient_boosting(
    split: RegressionSplit,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    model = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.06,
        l2_regularization=1e-4,
        random_state=seed,
    )
    t0 = time.perf_counter()
    model.fit(split.X_tr, split.y_tr)
    fit_seconds = time.perf_counter() - t0
    val_pred = model.predict(split.X_val)
    test_pred = model.predict(split.X_te)
    row: dict[str, object] = {
        "model": "HistGradientBoosting",
        "family": "sklearn",
        "fit_seconds": float(fit_seconds),
    }
    row.update(regression_metrics(split, val_pred, test_pred))
    return row, test_pred


def fit_gradient_mlp(
    split: RegressionSplit,
    seed: int,
    max_iter: int,
) -> tuple[dict[str, object], np.ndarray]:
    model = MLPRegressor(
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
        model.fit(split.X_tr, split.y_tr_scaled)
    fit_seconds = time.perf_counter() - t0
    val_pred = _inverse_target(split, model.predict(split.X_val))
    test_pred = _inverse_target(split, model.predict(split.X_te))
    row: dict[str, object] = {
        "model": "Gradient MLP",
        "family": "sklearn",
        "fit_seconds": float(fit_seconds),
        "iterations": int(model.n_iter_),
    }
    row.update(regression_metrics(split, val_pred, test_pred))
    return row, test_pred


def run_dataset(split: RegressionSplit, args: argparse.Namespace, seed: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    predictions: dict[str, list[float]] = {}
    histories: dict[str, list[dict[str, object]]] = {}
    candidates: dict[str, list[dict[str, object]]] = {}

    row, pred = fit_ridge_design(
        split,
        "Linear Ridge+GCV",
        _linear_design(split.X_tr),
        _linear_design(split.X_val),
        _linear_design(split.X_te),
    )
    rows.append(row)
    predictions[row["model"]] = pred.tolist()

    row, random_candidates, pred = fit_random_tanh_ridge(
        split,
        hidden_grid=args.random_hidden_grid,
        seed=seed,
    )
    rows.append(row)
    candidates["Random tanh Ridge+GCV"] = random_candidates
    predictions[row["model"]] = pred.tolist()

    row, es_histories, es_strategy_rows, pred = fit_es_strategy_sweep(
        split,
        strategies=args.es_strategies,
        epochs=args.es_epochs,
        mu=args.es_mu,
        n_offspring=args.es_offspring,
        sigma_init=args.sigma_init,
        gamma=args.gamma,
        seed=seed,
    )
    rows.append(row)
    histories.update(es_histories)
    candidates["ES strategy sweep"] = es_strategy_rows
    predictions[row["model"]] = pred.tolist()

    row, pred = fit_hist_gradient_boosting(split, seed=seed)
    rows.append(row)
    predictions[row["model"]] = pred.tolist()

    row, pred = fit_gradient_mlp(split, seed=seed, max_iter=args.mlp_max_iter)
    rows.append(row)
    predictions[row["model"]] = pred.tolist()

    return {
        "dataset": split.dataset,
        "n_train": int(split.X_tr.shape[0]),
        "n_validation": int(split.X_val.shape[0]),
        "n_test": int(split.X_te.shape[0]),
        "n_features": int(split.X_tr.shape[1]),
        "target_mean": split.y_mean,
        "target_std": split.y_std,
        "rows": rows,
        "histories": histories,
        "candidates": candidates,
        "y_test": split.y_te.tolist(),
        "predictions": predictions,
    }


def save_model_comparison_plot(results: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = ["Linear Ridge+GCV", "Random tanh Ridge+GCV", "ES-selected Ridge+GCV"]
    labels = ["Linear\nRidge+GCV", "Random tanh\nRidge+GCV", "ES-selected\nRidge+GCV"]
    values = []
    for model in models:
        normalised = []
        for result in results:
            rows_by_model = {row["model"]: row for row in result["rows"]}
            baseline = float(rows_by_model["Linear Ridge+GCV"]["test_rmse"])
            normalised.append(float(rows_by_model[model]["test_rmse"]) / baseline)
        values.append(float(np.mean(normalised)))

    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    x = np.arange(len(models))
    colors = ["#8c8c8c", "#f0c08a", "#1f77b4"]
    bars = ax.bar(x, values, color=colors, edgecolor="white")
    ax.axhline(1.0, color="#2f2f2f", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title("Mean normalised test RMSE across regression datasets")
    ax.set_ylabel("Normalised RMSE (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.84, 1.03)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("Regression Ridge representation comparison")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_ridge_value_plot(results: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ridge_models = ["Linear Ridge+GCV", "Random tanh Ridge+GCV"]
    prototype_models = [
        row["model"]
        for result in results
        for row in result["rows"]
        if str(row["model"]).startswith("ES-")
    ]
    prototype_name = sorted(set(prototype_models))[0]
    models = ridge_models + [prototype_name]
    datasets = [result["dataset"] for result in results]
    x = np.arange(len(datasets))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    for offset, model in zip([-width, 0.0, width], models):
        values = []
        for result in results:
            rows_by_model = {row["model"]: row for row in result["rows"]}
            values.append(float(rows_by_model[model]["test_r2"]))
        ax.bar(x + offset, values, width, label=model)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    ax.set_title("Value of evolved hidden representation in regression")
    ax.set_ylabel("Test R2 (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=10, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_higgs_prediction_plot(results: list[dict[str, object]], output_path: Path) -> None:
    higgs = next((result for result in results if str(result["dataset"]).startswith("Higgs")), None)
    if higgs is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(higgs["y_test"], dtype=np.float64)
    model_names = [
        "Linear Ridge+GCV",
        "Random tanh Ridge+GCV",
        next(row["model"] for row in higgs["rows"] if str(row["model"]).startswith("ES-")),
        "HistGradientBoosting",
    ]
    fig, axes = plt.subplots(1, len(model_names), figsize=(4.2 * len(model_names), 4.0), constrained_layout=True)
    rng = np.random.default_rng(123)
    sample_n = min(2500, len(y_true))
    idx = rng.choice(len(y_true), size=sample_n, replace=False)
    for ax, model in zip(axes, model_names):
        pred = np.asarray(higgs["predictions"][model], dtype=np.float64)
        ax.scatter(y_true[idx], pred[idx], s=6, alpha=0.22)
        ax.set_title(model)
        ax.set_xlabel("True y")
        ax.set_ylabel("Predicted y")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Higgs regression: predictions vs target")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_latex_notes(paths: dict[str, Path], output_path: Path) -> None:
    captions = {
        "model_comparison": "Regression benchmark comparing the Ridge prototype against linear Ridge, random tanh Ridge, HistGradientBoosting, and a gradient MLP.",
        "ridge_value": "Regression ablation showing the value of the evolved hidden representation over linear and random-feature Ridge.",
        "higgs_predictions": "Higgs regression predictions against the numeric target for representative models.",
    }
    lines = [
        "% Generated by ridge_regression_benchmark.py.",
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
                f"% \\label{{fig:regression_{key}}}",
                "% \\end{figure}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def print_summary(results: list[dict[str, object]]) -> None:
    print("\n## Regression Benchmark")
    for result in results:
        print(f"\n### {result['dataset']}")
        print("| Model | Val RMSE | Test RMSE | Test MAE | Test R2 | Fit s |")
        print("|---|---:|---:|---:|---:|---:|")
        for row in sorted(result["rows"], key=lambda item: float(item["test_rmse"])):
            print(
                f"| {row['model']} | "
                f"{float(row['validation_rmse']):.4f} | "
                f"{float(row['test_rmse']):.4f} | "
                f"{float(row['test_mae']):.4f} | "
                f"{float(row['test_r2']):.4f} | "
                f"{float(row['fit_seconds']):.3f} |"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Ridge/GCV regression benchmarks.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["higgs", "friedman1", "diabetes"],
        default=["higgs", "friedman1", "diabetes"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--data-home", default=".openml_cache")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-hidden-grid", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument(
        "--es-strategies",
        nargs="+",
        choices=["omp", "r2", "cfs"],
        default=["cfs", "omp", "r2"],
        help="ES neuron-selection strategies to evaluate; the reported prototype is selected by validation RMSE.",
    )
    parser.add_argument(
        "--es-strategy",
        choices=["omp", "r2", "cfs"],
        default=None,
        help="Backward-compatible shortcut for running one ES strategy only.",
    )
    parser.add_argument("--es-epochs", type=int, default=35)
    parser.add_argument("--es-mu", type=int, default=60)
    parser.add_argument("--es-offspring", type=int, default=180)
    parser.add_argument("--sigma-init", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--mlp-max-iter", type=int, default=120)
    parser.add_argument("--output-dir", default="runs/ridge_regression")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.es_strategy is not None:
        args.es_strategies = [args.es_strategy]
    output_dir = Path(args.output_dir)
    results: list[dict[str, object]] = []
    for dataset in args.datasets:
        print(f"\nRunning regression dataset: {dataset}")
        split = load_regression_split(
            dataset,
            seed=args.seed,
            data_home=args.data_home,
            max_samples=args.max_samples,
            validation_size=args.validation_size,
            test_size=args.test_size,
        )
        results.append(run_dataset(split, args=args, seed=args.seed))

    paths = {
        "model_comparison": output_dir / "regression_model_comparison.png",
        "ridge_value": output_dir / "regression_ridge_value.png",
    }
    has_higgs = any(str(result["dataset"]).startswith("Higgs") for result in results)
    if has_higgs:
        paths["higgs_predictions"] = output_dir / "higgs_predictions.png"
    save_model_comparison_plot(results, paths["model_comparison"])
    save_ridge_value_plot(results, paths["ridge_value"])
    if has_higgs:
        save_higgs_prediction_plot(results, paths["higgs_predictions"])
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
