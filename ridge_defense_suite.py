"""
ridge_defense_suite.py — Thesis-oriented experiments for the Ridge/GCV branch.

This suite is deliberately broader than a single strategy comparison. It
answers the questions that are most useful in a defence:

1) Is ES + Ridge/GCV competitive against standard tabular baselines?
2) What does each architectural ingredient contribute?
3) How sensitive is Ridge/GCV to lambda?
4) What accuracy/cost trade-off does the prototype occupy?

Outputs are written to runs/ridge_defense by default:
  - summary.json with all raw and aggregate metrics.
  - Several PNG figures.
  - latex_notes.tex with commented figure snippets for manual inclusion.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from core_es import compute_activations
from data import load_dataset_with_validation
from gcv import compute_ridge_gcv
from train import train


DATASETS = {
    "ionosphere": {"id": 59, "name": None, "label": "Ionosphere"},
    "sonar": {"id": 40, "name": None, "label": "Sonar"},
    "breast-w": {"id": 15, "name": None, "label": "Breast-W"},
}

RIDGE_MODELS = (
    "Linear Ridge+GCV",
    "Random tanh Ridge+GCV",
    "ES-CFS Ridge+GCV",
)


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


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0))}


def _fmt(summary: dict[str, float]) -> str:
    return f"{summary['mean']:.4f} +/- {summary['std']:.4f}"


def _linear_design(X: np.ndarray) -> np.ndarray:
    return np.c_[X.astype(np.float64), np.ones(X.shape[0], dtype=np.float64)]


def _random_tanh_designs(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_te: np.ndarray,
    hidden_units: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    input_dim = X_tr.shape[1]
    bound = 1.0 / np.sqrt(input_dim)
    alpha = rng.uniform(
        low=-bound,
        high=bound,
        size=(hidden_units, input_dim + 1),
    ).astype(np.float64)
    Phi_tr = np.c_[compute_activations(alpha, X_tr), np.ones(X_tr.shape[0])]
    Phi_val = np.c_[compute_activations(alpha, X_val), np.ones(X_val.shape[0])]
    Phi_te = np.c_[compute_activations(alpha, X_te), np.ones(X_te.shape[0])]
    return Phi_tr, Phi_val, Phi_te


def _score_metrics(
    y_tr: np.ndarray,
    y_val: np.ndarray,
    y_te: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float]:
    y_tr_v = y_tr.reshape(-1)
    y_val_v = y_val.reshape(-1)
    y_te_v = y_te.reshape(-1)
    train_pred = np.where(train_scores >= 0.0, 1.0, -1.0)
    val_pred = np.where(val_scores >= 0.0, 1.0, -1.0)
    test_pred = np.where(test_scores >= 0.0, 1.0, -1.0)
    return {
        "train_acc": float(accuracy_score(y_tr_v, train_pred)),
        "validation_acc": float(accuracy_score(y_val_v, val_pred)),
        "test_acc": float(accuracy_score(y_te_v, test_pred)),
        "train_mse": float(mean_squared_error(y_tr_v, train_scores)),
        "validation_mse": float(mean_squared_error(y_val_v, val_scores)),
        "test_mse": float(mean_squared_error(y_te_v, test_scores)),
        "validation_roc_auc": float(
            roc_auc_score((y_val_v > 0.0).astype(int), val_scores)
        ),
        "test_roc_auc": float(roc_auc_score((y_te_v > 0.0).astype(int), test_scores)),
    }


def _fit_ridge_scores(
    Phi_tr: np.ndarray,
    Phi_val: np.ndarray,
    Phi_te: np.ndarray,
    y_tr: np.ndarray,
    lambda_override: float | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float]:
    fit_t0 = time.perf_counter()
    lam, beta, gcv = compute_ridge_gcv(Phi_tr, y_tr, lambda_override=lambda_override)
    fit_seconds = time.perf_counter() - fit_t0
    return (
        float(lam),
        float(gcv) if np.isfinite(gcv) else np.nan,
        (Phi_tr @ beta).reshape(-1),
        (Phi_val @ beta).reshape(-1),
        (Phi_te @ beta).reshape(-1),
        float(fit_seconds),
    )


def fit_linear_ridge(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    y_te: np.ndarray,
) -> dict[str, object]:
    lam, gcv, train_scores, val_scores, test_scores, fit_seconds = _fit_ridge_scores(
        _linear_design(X_tr),
        _linear_design(X_val),
        _linear_design(X_te),
        y_tr,
    )
    row: dict[str, object] = {
        "model": "Linear Ridge+GCV",
        "family": "ridge",
        "lambda": lam,
        "gcv": gcv,
        "fit_seconds": fit_seconds,
        "hidden_units": 0,
        "selected_by": "gcv",
    }
    row.update(_score_metrics(y_tr, y_val, y_te, train_scores, val_scores, test_scores))
    return row


def fit_random_tanh_ridge(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    y_te: np.ndarray,
    hidden_units_grid: list[int],
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for hidden_units in hidden_units_grid:
        Phi_tr, Phi_val, Phi_te = _random_tanh_designs(
            X_tr,
            X_val,
            X_te,
            hidden_units=hidden_units,
            seed=seed + hidden_units * 17,
        )
        lam, gcv, train_scores, val_scores, test_scores, fit_seconds = _fit_ridge_scores(
            Phi_tr,
            Phi_val,
            Phi_te,
            y_tr,
        )
        row: dict[str, object] = {
            "model": "Random tanh Ridge+GCV",
            "family": "ridge",
            "lambda": lam,
            "gcv": gcv,
            "fit_seconds": fit_seconds,
            "hidden_units": int(hidden_units),
            "selected_by": "validation_acc_penalised_by_mse",
        }
        row.update(_score_metrics(y_tr, y_val, y_te, train_scores, val_scores, test_scores))
        row["selection_score"] = float(row["validation_acc"]) - 0.05 * np.log1p(
            float(row["validation_mse"])
        )
        candidates.append(row)
        if best is None or float(row["selection_score"]) > float(best["selection_score"]):
            best = row
    if best is None:
        raise RuntimeError("No random tanh Ridge candidates were evaluated.")
    return dict(best), candidates


def fit_es_cfs(
    dataset: dict[str, object],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    fit_t0 = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        state = train(
            fitness_strategy="cfs",
            epochs=args.es_epochs,
            mu=args.es_mu,
            n_offspring=args.es_offspring,
            data_id=int(dataset["id"]),
            data_name=dataset["name"],
            data_version=args.data_version,
            test_size=args.test_size,
            validation_size=args.validation_size,
            data_home=args.data_home,
            verbose=999,
            seed=seed,
            save_dir=None,
        )
    fit_seconds = time.perf_counter() - fit_t0
    metrics = state.metrics
    return {
        "model": "ES-CFS Ridge+GCV",
        "family": "ridge",
        "fit_seconds": float(fit_seconds),
        "lambda": metrics.get("final_lambda"),
        "gcv": metrics.get("final_gcv"),
        "hidden_units": int(args.es_mu),
        "selected_epoch": metrics.get("selected_epoch"),
        "selected_by": "validation_acc_then_mse",
        "train_acc": metrics.get("final_train_acc"),
        "validation_acc": metrics.get("validation_acc"),
        "test_acc": metrics.get("test_acc"),
        "train_mse": metrics.get("train_mse"),
        "validation_mse": metrics.get("validation_mse"),
        "test_mse": metrics.get("test_mse"),
        "validation_roc_auc": metrics.get("validation_roc_auc"),
        "test_roc_auc": metrics.get("test_roc_auc"),
    }


def _sklearn_score(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def fit_sklearn_baseline(
    name: str,
    model,
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    y_te: np.ndarray,
) -> dict[str, object]:
    y_tr01 = (y_tr.reshape(-1) > 0.0).astype(int)
    y_val01 = (y_val.reshape(-1) > 0.0).astype(int)
    y_te01 = (y_te.reshape(-1) > 0.0).astype(int)

    fit_t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_tr, y_tr01)
    fit_seconds = time.perf_counter() - fit_t0

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_te)
    val_score = _sklearn_score(model, X_val)
    test_score = _sklearn_score(model, X_te)

    return {
        "model": name,
        "family": "sklearn",
        "fit_seconds": float(fit_seconds),
        "validation_acc": float(accuracy_score(y_val01, val_pred)),
        "test_acc": float(accuracy_score(y_te01, test_pred)),
        "validation_roc_auc": float(roc_auc_score(y_val01, val_score)),
        "test_roc_auc": float(roc_auc_score(y_te01, test_score)),
    }


def make_sklearn_baselines(seed: int, mlp_max_iter: int) -> list[tuple[str, object]]:
    return [
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed),
        ),
        ("RBF SVM", SVC(kernel="rbf", C=1.0, gamma="scale")),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, random_state=seed),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "Gradient MLP",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=mlp_max_iter,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=12,
                random_state=seed,
                batch_size=64,
            ),
        ),
    ]


def summarise_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    models = sorted({str(row["model"]) for row in rows})
    datasets = sorted({str(row["dataset"]) for row in rows})
    metrics = (
        "validation_acc",
        "test_acc",
        "validation_roc_auc",
        "test_roc_auc",
        "test_mse",
        "fit_seconds",
    )
    summary: dict[str, object] = {"overall": {}, "by_dataset": {}}
    for model in models:
        model_rows = [row for row in rows if str(row["model"]) == model]
        summary["overall"][model] = {"n_runs": len(model_rows)}
        for metric in metrics:
            values = [
                float(row[metric])
                for row in model_rows
                if row.get(metric) is not None and np.isfinite(float(row[metric]))
            ]
            if values:
                summary["overall"][model][metric] = _mean_std(values)

    for dataset in datasets:
        summary["by_dataset"][dataset] = {}
        for model in models:
            model_rows = [
                row
                for row in rows
                if str(row["dataset"]) == dataset and str(row["model"]) == model
            ]
            summary["by_dataset"][dataset][model] = {"n_runs": len(model_rows)}
            for metric in metrics:
                values = [
                    float(row[metric])
                    for row in model_rows
                    if row.get(metric) is not None and np.isfinite(float(row[metric]))
                ]
                if values:
                    summary["by_dataset"][dataset][model][metric] = _mean_std(values)
    return summary


def compute_test_accuracy_ranks(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), int(row["seed"])), []).append(row)

    rank_rows: list[dict[str, object]] = []
    for (dataset, seed), group in grouped.items():
        ordered = sorted(group, key=lambda row: float(row["test_acc"]), reverse=True)
        for rank, row in enumerate(ordered, start=1):
            rank_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model": row["model"],
                    "rank": rank,
                    "test_acc": row["test_acc"],
                }
            )

    models = sorted({str(row["model"]) for row in rank_rows})
    return {
        model: _mean_std(
            [float(row["rank"]) for row in rank_rows if str(row["model"]) == model]
        )
        for model in models
    }


def run_baseline_suite(args: argparse.Namespace) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    tanh_candidates: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        dataset = DATASETS[dataset_name]
        for seed in args.seeds:
            print(f"Dataset={dataset['label']} seed={seed}")
            X_tr, X_val, X_te, y_tr, y_val, y_te = load_dataset_with_validation(
                data_id=int(dataset["id"]),
                data_name=dataset["name"],
                version=args.data_version,
                validation_size=args.validation_size,
                test_size=args.test_size,
                seed=seed,
                data_home=args.data_home,
            )

            linear = fit_linear_ridge(X_tr, X_val, X_te, y_tr, y_val, y_te)
            rows.append({**linear, "dataset": dataset["label"], "seed": seed})

            random_tanh, candidates = fit_random_tanh_ridge(
                X_tr,
                X_val,
                X_te,
                y_tr,
                y_val,
                y_te,
                hidden_units_grid=args.hidden_units_grid,
                seed=seed,
            )
            rows.append({**random_tanh, "dataset": dataset["label"], "seed": seed})
            tanh_candidates.extend(
                {**candidate, "dataset": dataset["label"], "seed": seed}
                for candidate in candidates
            )

            es_cfs = fit_es_cfs(dataset, seed, args)
            rows.append({**es_cfs, "dataset": dataset["label"], "seed": seed})

            for name, model in make_sklearn_baselines(seed, args.mlp_max_iter):
                baseline = fit_sklearn_baseline(
                    name,
                    model,
                    X_tr,
                    X_val,
                    X_te,
                    y_tr,
                    y_val,
                    y_te,
                )
                rows.append({**baseline, "dataset": dataset["label"], "seed": seed})

    return {
        "rows": rows,
        "random_tanh_candidates": tanh_candidates,
        "summary": summarise_rows(rows),
        "test_accuracy_ranks": compute_test_accuracy_ranks(rows),
    }


def run_lambda_sensitivity(args: argparse.Namespace) -> dict[str, object]:
    dataset = DATASETS[args.diagnostic_dataset]
    seed = args.diagnostic_seed
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_dataset_with_validation(
        data_id=int(dataset["id"]),
        data_name=dataset["name"],
        version=args.data_version,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=seed,
        data_home=args.data_home,
    )
    Phi_tr, Phi_val, Phi_te = _random_tanh_designs(
        X_tr,
        X_val,
        X_te,
        hidden_units=args.diagnostic_hidden_units,
        seed=seed + 1000,
    )
    gcv_lam, gcv, train_scores, val_scores, test_scores, fit_seconds = _fit_ridge_scores(
        Phi_tr,
        Phi_val,
        Phi_te,
        y_tr,
    )
    gcv_row = {
        "lambda": gcv_lam,
        "gcv": gcv,
        "fit_seconds": fit_seconds,
        "kind": "gcv_selected",
    }
    gcv_row.update(_score_metrics(y_tr, y_val, y_te, train_scores, val_scores, test_scores))

    rows: list[dict[str, object]] = []
    for lam in args.lambda_grid:
        _, _, train_s, val_s, test_s, fit_s = _fit_ridge_scores(
            Phi_tr,
            Phi_val,
            Phi_te,
            y_tr,
            lambda_override=float(lam),
        )
        row = {"lambda": float(lam), "kind": "fixed", "fit_seconds": fit_s}
        row.update(_score_metrics(y_tr, y_val, y_te, train_s, val_s, test_s))
        rows.append(row)
    return {
        "dataset": dataset["label"],
        "seed": seed,
        "hidden_units": args.diagnostic_hidden_units,
        "gcv_selected": gcv_row,
        "fixed_lambda_rows": rows,
    }


def save_baseline_accuracy_plot(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]["by_dataset"]
    datasets = list(summary.keys())
    models = list(payload["summary"]["overall"].keys())
    x = np.arange(len(models))
    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(13.5, max(4.2, 3.2 * len(datasets))),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, dataset in zip(axes.ravel(), datasets):
        means = [
            summary[dataset][model].get("test_acc", {"mean": np.nan})["mean"]
            for model in models
        ]
        stds = [
            summary[dataset][model].get("test_acc", {"std": 0.0})["std"]
            for model in models
        ]
        colors = ["#2f6f9f" if model in RIDGE_MODELS else "#b36b2c" for model in models]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors)
        ax.set_title(f"{dataset}: test accuracy across seeds")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=18, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Ridge prototype against standard tabular baselines")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_accuracy_time_plot(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overall = payload["summary"]["overall"]
    models = list(overall.keys())
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for model in models:
        x = overall[model]["fit_seconds"]["mean"]
        y = overall[model]["test_acc"]["mean"]
        xerr = overall[model]["fit_seconds"]["std"]
        yerr = overall[model]["test_acc"]["std"]
        color = "#2f6f9f" if model in RIDGE_MODELS else "#b36b2c"
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            marker="o",
            markersize=7,
            capsize=3,
            color=color,
            label=model,
        )
        ax.annotate(model, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Mean fit time (seconds, log scale)")
    ax.set_ylabel("Mean test accuracy")
    ax.set_title("Accuracy/cost trade-off")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_ridge_ablation_plot(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overall = payload["summary"]["overall"]
    models = [model for model in RIDGE_MODELS if model in overall]
    x = np.arange(len(models))
    width = 0.36
    val_means = [overall[model]["validation_acc"]["mean"] for model in models]
    val_stds = [overall[model]["validation_acc"]["std"] for model in models]
    test_means = [overall[model]["test_acc"]["mean"] for model in models]
    test_stds = [overall[model]["test_acc"]["std"] for model in models]
    auc_means = [overall[model]["test_roc_auc"]["mean"] for model in models]
    auc_stds = [overall[model]["test_roc_auc"]["std"] for model in models]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    axes[0].bar(x - width / 2, val_means, width, yerr=val_stds, capsize=3, label="Validation")
    axes[0].bar(x + width / 2, test_means, width, yerr=test_stds, capsize=3, label="Test")
    axes[0].set_title("Accuracy contribution")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x, auc_means, yerr=auc_stds, capsize=3, color="#2f6f9f")
    axes[1].set_title("Ranking quality")
    axes[1].set_ylabel("Test ROC-AUC")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Ablation: linear output vs random features vs evolved features")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_lambda_sensitivity_plot(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(payload["fixed_lambda_rows"], key=lambda row: float(row["lambda"]))
    lambdas = np.asarray([float(row["lambda"]) for row in rows])
    val_acc = np.asarray([float(row["validation_acc"]) for row in rows])
    test_acc = np.asarray([float(row["test_acc"]) for row in rows])
    val_mse = np.asarray([float(row["validation_mse"]) for row in rows])
    test_mse = np.asarray([float(row["test_mse"]) for row in rows])
    gcv_lam = float(payload["gcv_selected"]["lambda"])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    axes[0].plot(lambdas, val_acc, marker="o", label="Validation")
    axes[0].plot(lambdas, test_acc, marker="o", label="Test")
    axes[0].axvline(gcv_lam, color="black", linestyle="--", linewidth=1.3, label="GCV lambda")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Accuracy sensitivity")
    axes[0].set_xlabel("Fixed lambda")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(lambdas, val_mse, marker="o", label="Validation")
    axes[1].plot(lambdas, test_mse, marker="o", label="Test")
    axes[1].axvline(gcv_lam, color="black", linestyle="--", linewidth=1.3, label="GCV lambda")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Score-MSE sensitivity")
    axes[1].set_xlabel("Fixed lambda")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle(
        f"Lambda sensitivity on {payload['dataset']} "
        f"(H={payload['hidden_units']}, seed={payload['seed']})"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_ranking_plot(ranks: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = sorted(ranks, key=lambda model: ranks[model]["mean"])
    means = [ranks[model]["mean"] for model in models]
    stds = [ranks[model]["std"] for model in models]
    colors = ["#2f6f9f" if model in RIDGE_MODELS else "#b36b2c" for model in models]
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.bar(np.arange(len(models)), means, yerr=stds, capsize=3, color=colors)
    ax.set_title("Mean test-accuracy rank across dataset/seed pairs")
    ax.set_ylabel("Mean rank (lower is better)")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=18, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_latex_notes(paths: dict[str, Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by ridge_defense_suite.py.",
        "% Copy the snippets you want into report.txt and adjust captions if needed.",
        "",
    ]
    captions = {
        "baseline_accuracy": "Comparison between the Ridge prototype and standard tabular baselines.",
        "accuracy_time": "Accuracy/cost trade-off across Ridge and non-Ridge models.",
        "ridge_ablation": "Ablation of the Ridge architecture: linear features, random tanh features, and evolved CFS features.",
        "lambda_sensitivity": "Sensitivity of Ridge scores to the regularisation parameter lambda.",
        "ranking": "Average test-accuracy rank across dataset/seed pairs.",
    }
    for key, path in paths.items():
        rel = path.as_posix()
        lines.extend(
            [
                f"% Figure: {key}",
                "% \\begin{figure}[H]",
                "% \\centering",
                f"% \\includegraphics[width=0.95\\textwidth]{{{rel}}}",
                f"% \\caption{{{captions[key]}}}",
                f"% \\label{{fig:{key}}}",
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


def print_summary(payload: dict[str, object]) -> None:
    print("\n## Overall Test Accuracy")
    print("| Model | Runs | Val Acc | Test Acc | Test AUC | Fit seconds |")
    print("|---|---:|---:|---:|---:|---:|")
    for model, stats in sorted(
        payload["baseline_suite"]["summary"]["overall"].items(),
        key=lambda item: item[1]["test_acc"]["mean"],
        reverse=True,
    ):
        print(
            f"| {model} | {int(stats['n_runs'])} | "
            f"{_fmt(stats['validation_acc'])} | "
            f"{_fmt(stats['test_acc'])} | "
            f"{_fmt(stats['test_roc_auc'])} | "
            f"{_fmt(stats['fit_seconds'])} |"
        )

    print("\n## Mean Test-Accuracy Rank")
    print("| Model | Mean rank |")
    print("|---|---:|")
    for model, stats in sorted(
        payload["baseline_suite"]["test_accuracy_ranks"].items(),
        key=lambda item: item[1]["mean"],
    ):
        print(f"| {model} | {_fmt(stats)} |")

    lam = payload["lambda_sensitivity"]["gcv_selected"]
    print("\n## Lambda Sensitivity Diagnostic")
    print(
        f"GCV selected lambda={float(lam['lambda']):.4g}, "
        f"validation acc={float(lam['validation_acc']):.4f}, "
        f"test acc={float(lam['test_acc']):.4f}, "
        f"test MSE={float(lam['test_mse']):.4f}."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run thesis-oriented Ridge/GCV defence experiments."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["ionosphere", "sonar", "breast-w"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data-version", type=int, default=1)
    parser.add_argument("--data-home", default=".openml_cache")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--hidden-units-grid", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument(
        "--lambda-grid",
        nargs="+",
        type=float,
        default=[1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--diagnostic-dataset", choices=sorted(DATASETS), default="ionosphere")
    parser.add_argument("--diagnostic-seed", type=int, default=42)
    parser.add_argument("--diagnostic-hidden-units", type=int, default=64)
    parser.add_argument("--es-epochs", type=int, default=60)
    parser.add_argument("--es-mu", type=int, default=50)
    parser.add_argument("--es-offspring", type=int, default=250)
    parser.add_argument("--mlp-max-iter", type=int, default=120)
    parser.add_argument("--output-dir", default="runs/ridge_defense")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)

    baseline_suite = run_baseline_suite(args)
    lambda_sensitivity = run_lambda_sensitivity(args)

    payload = {
        "config": vars(args),
        "baseline_suite": baseline_suite,
        "lambda_sensitivity": lambda_sensitivity,
    }

    paths = {
        "baseline_accuracy": output_dir / "baseline_test_accuracy.png",
        "accuracy_time": output_dir / "accuracy_time_tradeoff.png",
        "ridge_ablation": output_dir / "ridge_ablation.png",
        "lambda_sensitivity": output_dir / "lambda_sensitivity.png",
        "ranking": output_dir / "test_accuracy_ranking.png",
    }
    save_baseline_accuracy_plot(baseline_suite, paths["baseline_accuracy"])
    save_accuracy_time_plot(baseline_suite, paths["accuracy_time"])
    save_ridge_ablation_plot(baseline_suite, paths["ridge_ablation"])
    save_lambda_sensitivity_plot(lambda_sensitivity, paths["lambda_sensitivity"])
    save_ranking_plot(baseline_suite["test_accuracy_ranks"], paths["ranking"])

    summary_path = output_dir / "summary.json"
    latex_notes_path = output_dir / "latex_notes.tex"
    save_json(payload, summary_path)
    write_latex_notes(paths, latex_notes_path)

    print_summary(payload)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved LaTeX notes: {latex_notes_path}")
    for name, path in paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
