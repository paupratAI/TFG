"""
lasso_filter_benchmark.py — Sparse logistic output-layer filtering experiment.

This experiment keeps the current Michigan ES hidden representation fixed and
compares three output-layer choices on the same selected neurons:

  1) Full ES-CFS Logistic CE: all evolved hidden neurons are kept.
  2) ES-CFS L1 Logistic: L1-logistic is used directly as the sparse output layer.
  3) L1-screened ES-CFS Logistic CE: L1-logistic selects active neurons, then
     unregularised logistic output is refit only on those neurons.

The goal is to test whether an L1 penalty can act as a second filtering stage
after the Michigan fitness step. The evolutionary stage already selects mu
neurons from a parent-plus-offspring pool; L1-logistic can then remove
redundant survivors at the output layer while keeping the classification
objective probabilistic.
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from core_es import compute_activations
from data import load_dataset_with_validation
from logistic_output import (
    binary_brier_score,
    binary_ece,
    binary_log_loss_from_proba,
    compute_logistic_output,
    labels_to_binary,
    predict_proba as predict_logistic_proba,
)
from train import train


DATASETS = {
    "ionosphere": {"id": 59, "name": None, "label": "Ionosphere"},
    "sonar": {"id": 40, "name": None, "label": "Sonar"},
    "breast-w": {"id": 15, "name": None, "label": "Breast-W"},
}


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


def _mean_std_sem(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr, ddof=0))
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "sem": float(std / np.sqrt(max(1, arr.size))),
    }


def _bias_design(Phi: np.ndarray) -> np.ndarray:
    return np.c_[np.asarray(Phi, dtype=np.float64), np.ones(Phi.shape[0])]


def _probability_metrics(
    y_tr: np.ndarray,
    y_val: np.ndarray,
    y_te: np.ndarray,
    train_proba: np.ndarray,
    val_proba: np.ndarray,
    test_proba: np.ndarray,
) -> dict[str, float]:
    y_tr_v = np.asarray(y_tr, dtype=np.float64).reshape(-1)
    y_val_v = np.asarray(y_val, dtype=np.float64).reshape(-1)
    y_te_v = np.asarray(y_te, dtype=np.float64).reshape(-1)
    y_tr01 = labels_to_binary(y_tr_v).reshape(-1)
    y_val01 = labels_to_binary(y_val_v).reshape(-1)
    y_te01 = labels_to_binary(y_te_v).reshape(-1)
    train_proba = np.asarray(train_proba, dtype=np.float64).reshape(-1)
    val_proba = np.asarray(val_proba, dtype=np.float64).reshape(-1)
    test_proba = np.asarray(test_proba, dtype=np.float64).reshape(-1)
    train_pred = np.where(train_proba >= 0.5, 1.0, -1.0)
    val_pred = np.where(val_proba >= 0.5, 1.0, -1.0)
    test_pred = np.where(test_proba >= 0.5, 1.0, -1.0)
    return {
        "train_acc": float(accuracy_score(y_tr_v, train_pred)),
        "validation_acc": float(accuracy_score(y_val_v, val_pred)),
        "test_acc": float(accuracy_score(y_te_v, test_pred)),
        "train_ce": binary_log_loss_from_proba(y_tr01, train_proba),
        "validation_ce": binary_log_loss_from_proba(y_val01, val_proba),
        "test_ce": binary_log_loss_from_proba(y_te01, test_proba),
        "train_brier": binary_brier_score(y_tr01, train_proba),
        "validation_brier": binary_brier_score(y_val01, val_proba),
        "test_brier": binary_brier_score(y_te01, test_proba),
        "train_ece": binary_ece(y_tr01, train_proba),
        "validation_ece": binary_ece(y_val01, val_proba),
        "test_ece": binary_ece(y_te01, test_proba),
        "validation_roc_auc": float(roc_auc_score(y_val01, val_proba)),
        "test_roc_auc": float(roc_auc_score(y_te01, test_proba)),
    }


def _fit_full_logistic(
    Phi_tr: np.ndarray,
    Phi_val: np.ndarray,
    Phi_te: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    y_te: np.ndarray,
) -> dict[str, object]:
    t0 = time.perf_counter()
    beta, train_ce, n_iter = compute_logistic_output(_bias_design(Phi_tr), y_tr)
    fit_seconds = time.perf_counter() - t0
    row: dict[str, object] = {
        "model": "Full ES-CFS Logistic CE",
        "family": "logistic",
        "C": None,
        "train_ce_objective": float(train_ce),
        "output_n_iter": int(n_iter),
        "active_neurons": int(Phi_tr.shape[1]),
        "active_fraction": 1.0,
        "fit_seconds": float(fit_seconds),
        "selected_by": "cross_entropy",
    }
    row.update(
        _probability_metrics(
            y_tr,
            y_val,
            y_te,
            predict_logistic_proba(_bias_design(Phi_tr), beta),
            predict_logistic_proba(_bias_design(Phi_val), beta),
            predict_logistic_proba(_bias_design(Phi_te), beta),
        )
    )
    return row


def _choose_sparse_candidate(
    candidates: list[dict[str, object]],
    validation_tolerance: float,
    selection_rule: str,
    sparsity_penalty: float,
) -> dict[str, object]:
    if not candidates:
        raise RuntimeError("No L1-logistic candidates were evaluated.")
    if selection_rule == "sparsity_penalty":
        selected = max(
            candidates,
            key=lambda row: (
                float(row["validation_acc"])
                - sparsity_penalty * float(row["active_fraction"]),
                float(row["validation_acc"]),
                -int(row["active_neurons"]),
                -float(row["validation_ce"]),
            ),
        )
        selected = dict(selected)
        selected["selection_score"] = float(selected["validation_acc"]) - (
            sparsity_penalty * float(selected["active_fraction"])
        )
        selected["selected_by"] = (
            f"validation_acc_minus_{sparsity_penalty:g}_active_fraction"
        )
        return selected
    if selection_rule != "tolerance":
        raise ValueError(f"Unknown selection_rule={selection_rule!r}.")
    best_val = max(float(row["validation_acc"]) for row in candidates)
    eligible = [
        row
        for row in candidates
        if float(row["validation_acc"]) >= best_val - validation_tolerance
    ]
    return dict(
        sorted(
            eligible,
            key=lambda row: (
                int(row["active_neurons"]),
                -float(row["validation_acc"]),
                float(row["validation_ce"]),
            ),
        )[0]
    )
    selected["selection_score"] = float(selected["validation_acc"])
    selected["selected_by"] = (
        f"validation_acc_with_{validation_tolerance:g}_tolerance_then_sparsity"
    )
    return selected


def _fit_l1_logistic_path(
    Phi_tr: np.ndarray,
    Phi_val: np.ndarray,
    Phi_te: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    y_te: np.ndarray,
    c_grid: np.ndarray,
    active_threshold: float,
    max_iter: int,
    tol: float,
    validation_tolerance: float,
    selection_rule: str,
    sparsity_penalty: float,
) -> tuple[dict[str, object], list[dict[str, object]], dict[float, np.ndarray]]:
    candidates: list[dict[str, object]] = []
    active_by_c: dict[float, np.ndarray] = {}
    y_tr01 = labels_to_binary(y_tr).reshape(-1).astype(int)
    for c_value in c_grid:
        t0 = time.perf_counter()
        model = LogisticRegression(
            penalty="l1",
            C=float(c_value),
            solver="liblinear",
            fit_intercept=True,
            max_iter=max_iter,
            tol=tol,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(Phi_tr, y_tr01)
        fit_seconds = time.perf_counter() - t0
        coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        active = np.flatnonzero(np.abs(coef) > active_threshold)
        active_by_c[float(c_value)] = active
        train_proba = model.predict_proba(Phi_tr)[:, 1]
        val_proba = model.predict_proba(Phi_val)[:, 1]
        test_proba = model.predict_proba(Phi_te)[:, 1]
        row: dict[str, object] = {
            "model": "ES-CFS L1 Logistic",
            "family": "l1_logistic",
            "C": float(c_value),
            "active_neurons": int(active.size),
            "active_fraction": float(active.size / max(1, Phi_tr.shape[1])),
            "fit_seconds": float(fit_seconds),
            "selected_by": f"validation_acc_with_{validation_tolerance:g}_tolerance_then_sparsity",
        }
        row.update(_probability_metrics(y_tr, y_val, y_te, train_proba, val_proba, test_proba))
        candidates.append(row)
    selected = _choose_sparse_candidate(
        candidates,
        validation_tolerance=validation_tolerance,
        selection_rule=selection_rule,
        sparsity_penalty=sparsity_penalty,
    )
    return selected, candidates, active_by_c


def _fit_screened_logistic_path(
    Phi_tr: np.ndarray,
    Phi_val: np.ndarray,
    Phi_te: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    y_te: np.ndarray,
    l1_candidates: list[dict[str, object]],
    active_by_c: dict[float, np.ndarray],
    validation_tolerance: float,
    selection_rule: str,
    sparsity_penalty: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for l1_row in l1_candidates:
        c_value = float(l1_row["C"])
        active = active_by_c[c_value]
        if active.size == 0:
            continue
        t0 = time.perf_counter()
        beta, train_ce, n_iter = compute_logistic_output(_bias_design(Phi_tr[:, active]), y_tr)
        fit_seconds = time.perf_counter() - t0
        row: dict[str, object] = {
            "model": "L1-screened ES-CFS Logistic CE",
            "family": "l1_screened_logistic",
            "C": c_value,
            "train_ce_objective": float(train_ce),
            "output_n_iter": int(n_iter),
            "active_neurons": int(active.size),
            "active_fraction": float(active.size / max(1, Phi_tr.shape[1])),
            "fit_seconds": float(fit_seconds + float(l1_row["fit_seconds"])),
            "selected_by": f"validation_acc_with_{validation_tolerance:g}_tolerance_then_sparsity",
        }
        row.update(
            _probability_metrics(
                y_tr,
                y_val,
                y_te,
                predict_logistic_proba(_bias_design(Phi_tr[:, active]), beta),
                predict_logistic_proba(_bias_design(Phi_val[:, active]), beta),
                predict_logistic_proba(_bias_design(Phi_te[:, active]), beta),
            )
        )
        candidates.append(row)
    selected = _choose_sparse_candidate(
        candidates,
        validation_tolerance=validation_tolerance,
        selection_rule=selection_rule,
        sparsity_penalty=sparsity_penalty,
    )
    return selected, candidates


def _hidden_designs_for_state(
    alpha: np.ndarray,
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Phi_tr_raw = compute_activations(alpha, X_tr)
    Phi_val_raw = compute_activations(alpha, X_val)
    Phi_te_raw = compute_activations(alpha, X_te)
    scaler = StandardScaler()
    Phi_tr = scaler.fit_transform(Phi_tr_raw).astype(np.float64)
    Phi_val = scaler.transform(Phi_val_raw).astype(np.float64)
    Phi_te = scaler.transform(Phi_te_raw).astype(np.float64)
    return Phi_tr, Phi_val, Phi_te


def run_one(dataset_key: str, seed: int, args: argparse.Namespace) -> dict[str, object]:
    dataset = DATASETS[dataset_key]
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

    es_t0 = time.perf_counter()
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
            output_layer="logistic",
            save_dir=None,
        )
    es_seconds = time.perf_counter() - es_t0

    Phi_tr, Phi_val, Phi_te = _hidden_designs_for_state(
        state.alpha,
        X_tr,
        X_val,
        X_te,
    )
    c_grid = np.asarray(args.c_grid, dtype=np.float64)
    full_row = _fit_full_logistic(Phi_tr, Phi_val, Phi_te, y_tr, y_val, y_te)
    l1_row, l1_candidates, active_by_c = _fit_l1_logistic_path(
        Phi_tr,
        Phi_val,
        Phi_te,
        y_tr,
        y_val,
        y_te,
        c_grid=c_grid,
        active_threshold=args.active_threshold,
        max_iter=args.logistic_max_iter,
        tol=args.logistic_tol,
        validation_tolerance=args.validation_tolerance,
        selection_rule=args.selection_rule,
        sparsity_penalty=args.sparsity_penalty,
    )
    screened_row, screened_candidates = _fit_screened_logistic_path(
        Phi_tr,
        Phi_val,
        Phi_te,
        y_tr,
        y_val,
        y_te,
        l1_candidates=l1_candidates,
        active_by_c=active_by_c,
        validation_tolerance=args.validation_tolerance,
        selection_rule=args.selection_rule,
        sparsity_penalty=args.sparsity_penalty,
    )

    rows = []
    for row in (full_row, l1_row, screened_row):
        enriched = {
            **row,
            "dataset": dataset["label"],
            "seed": int(seed),
            "hidden_units": int(args.es_mu),
            "es_seconds": float(es_seconds),
            "selected_epoch": state.metrics.get("selected_epoch"),
        }
        rows.append(enriched)

    return {
        "dataset": dataset["label"],
        "seed": int(seed),
        "rows": rows,
        "l1_candidates": [
            {**row, "dataset": dataset["label"], "seed": int(seed)}
            for row in l1_candidates
        ],
        "screened_candidates": [
            {**row, "dataset": dataset["label"], "seed": int(seed)}
            for row in screened_candidates
        ],
    }


def summarise(rows: list[dict[str, object]]) -> dict[str, object]:
    models = sorted({str(row["model"]) for row in rows})
    datasets = sorted({str(row["dataset"]) for row in rows})
    metrics = (
        "validation_acc",
        "test_acc",
        "test_roc_auc",
        "test_ce",
        "test_brier",
        "test_ece",
        "active_neurons",
        "active_fraction",
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
                summary["overall"][model][metric] = _mean_std_sem(values)
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
                    summary["by_dataset"][dataset][model][metric] = _mean_std_sem(values)
    return summary


def save_accuracy_sparsity_plot(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = [
        "Full ES-CFS Logistic CE",
        "ES-CFS L1 Logistic",
        "L1-screened ES-CFS Logistic CE",
    ]
    labels = ["Full Logistic", "L1 Logistic", "L1-screened Logistic"]
    overall = summary["overall"]
    x = np.arange(len(models))
    val_means = [overall[model]["validation_acc"]["mean"] for model in models]
    val_sem = [overall[model]["validation_acc"]["sem"] for model in models]
    test_means = [overall[model]["test_acc"]["mean"] for model in models]
    test_sem = [overall[model]["test_acc"]["sem"] for model in models]
    active_means = [overall[model]["active_neurons"]["mean"] for model in models]
    active_sem = [overall[model]["active_neurons"]["sem"] for model in models]
    colors = ["#7F7F7F", "#0072B2", "#009E73"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    axes[0].bar(x, val_means, yerr=val_sem, capsize=4, color=colors)
    axes[0].set_title("Validation performance")
    axes[0].set_ylabel("Mean validation accuracy")
    axes[0].set_ylim(max(0.0, min(val_means) - 0.08), 1.02)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, test_means, yerr=test_sem, capsize=4, color=colors)
    axes[1].set_title("Test performance")
    axes[1].set_ylabel("Mean test accuracy")
    axes[1].set_ylim(max(0.0, min(test_means) - 0.08), 1.02)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(x, active_means, yerr=active_sem, capsize=4, color=colors)
    axes[2].set_title("Output-layer sparsity")
    axes[2].set_ylabel("Active hidden neurons")
    axes[2].set_ylim(0, max(active_means) * 1.2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25)
    fig.suptitle("L1-logistic as a second filter after Michigan ES selection")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_dataset_sparsity_plot(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = list(summary["by_dataset"].keys())
    models = [
        "Full ES-CFS Logistic CE",
        "ES-CFS L1 Logistic",
        "L1-screened ES-CFS Logistic CE",
    ]
    labels = ["Full Logistic", "L1 Logistic", "Screened Logistic"]
    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.0, 5.0), constrained_layout=True)
    for i, model in enumerate(models):
        means = [
            summary["by_dataset"][dataset][model]["active_neurons"]["mean"]
            for dataset in datasets
        ]
        ax.bar(
            x + (i - 1) * width,
            means,
            width=width,
            label=labels[i],
            color=["#7F7F7F", "#0072B2", "#009E73"][i],
            edgecolor="white",
        )
    ax.set_title("Selected active neurons by dataset")
    ax.set_ylabel("Active hidden neurons")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_latex_notes(summary: dict[str, object], output_path: Path) -> None:
    overall = summary["overall"]
    lines = [
        "% Auto-generated notes for the L1-logistic filtering experiment.",
        "\\begin{figure}[H]",
        "\\centering",
        "\\includegraphics[width=0.95\\textwidth]{runs/lasso_filter/lasso_accuracy_sparsity.png}",
        "\\caption{L1-logistic output filtering after Michigan ES selection. The panels compare validation accuracy, test accuracy, and the number of active hidden neurons.}",
        "\\label{fig:lasso_accuracy_sparsity}",
        "\\end{figure}",
        "",
        "% Overall aggregate:",
    ]
    for model, stats in overall.items():
        lines.append(
            "% "
            f"{model}: test_acc={stats['test_acc']['mean']:.4f} +/- {stats['test_acc']['std']:.4f}, "
            f"test_auc={stats['test_roc_auc']['mean']:.4f} +/- {stats['test_roc_auc']['std']:.4f}, "
            f"active={stats['active_neurons']['mean']:.2f} +/- {stats['active_neurons']['std']:.2f}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the L1-logistic filtering benchmark.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["ionosphere", "sonar", "breast-w"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data-version", type=int, default=1)
    parser.add_argument("--data-home", default=".openml_cache")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--es-epochs", type=int, default=60)
    parser.add_argument("--es-mu", type=int, default=50)
    parser.add_argument("--es-offspring", type=int, default=350)
    parser.add_argument("--c-grid", nargs="+", type=float, default=None)
    parser.add_argument("--c-min-exp", type=float, default=-2.0)
    parser.add_argument("--c-max-exp", type=float, default=3.0)
    parser.add_argument("--c-points", type=int, default=36)
    parser.add_argument("--validation-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--selection-rule",
        choices=["sparsity_penalty", "tolerance"],
        default="sparsity_penalty",
        help=(
            "How to choose the final L1-logistic C value on validation data. "
            "'sparsity_penalty' maximises validation_acc - sparsity_penalty * active_fraction; "
            "'tolerance' chooses the sparsest candidate within validation_tolerance of the best validation accuracy."
        ),
    )
    parser.add_argument(
        "--sparsity-penalty",
        type=float,
        default=0.05,
        help="Penalty applied to active_fraction when selection_rule='sparsity_penalty'.",
    )
    parser.add_argument("--active-threshold", type=float, default=1e-8)
    parser.add_argument("--logistic-max-iter", type=int, default=5000)
    parser.add_argument("--logistic-tol", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="runs/lasso_filter")
    args = parser.parse_args()
    if args.c_grid is None:
        args.c_grid = np.logspace(
            args.c_min_exp,
            args.c_max_exp,
            args.c_points,
        ).tolist()
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    l1_candidates: list[dict[str, object]] = []
    screened_candidates: list[dict[str, object]] = []
    for dataset in args.datasets:
        for seed in args.seeds:
            result = run_one(dataset, seed, args)
            runs.append(result)
            rows.extend(result["rows"])
            l1_candidates.extend(result["l1_candidates"])
            screened_candidates.extend(result["screened_candidates"])

    summary = summarise(rows)
    payload = {
        "config": vars(args),
        "rows": rows,
        "runs": runs,
        "l1_candidates": l1_candidates,
        "screened_candidates": screened_candidates,
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    save_accuracy_sparsity_plot(summary, output_dir / "lasso_accuracy_sparsity.png")
    save_dataset_sparsity_plot(summary, output_dir / "lasso_dataset_sparsity.png")
    write_latex_notes(summary, output_dir / "latex_notes.tex")

    print("\nOverall summary")
    for model, stats in summary["overall"].items():
        print(
            f"{model}: "
            f"test_acc={stats['test_acc']['mean']:.4f}+/-{stats['test_acc']['std']:.4f}, "
            f"test_auc={stats['test_roc_auc']['mean']:.4f}+/-{stats['test_roc_auc']['std']:.4f}, "
            f"test_ce={stats['test_ce']['mean']:.4f}+/-{stats['test_ce']['std']:.4f}, "
            f"active={stats['active_neurons']['mean']:.2f}+/-{stats['active_neurons']['std']:.2f}"
        )
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
