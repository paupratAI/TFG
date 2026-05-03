"""
benchmark_level1.py

Automated Level-1 benchmark for:
1) Michigan-style Evolution Strategy (train.py)
2) Conventional Genetic Algorithm (train_ga_conventional.py)

Datasets:
- 59  (Ionosphere)
- 40  (Sonar)
- 15  (Breast-W)
"""

from __future__ import annotations

import gc
import io
import time
from contextlib import redirect_stdout

import numpy as np
from sklearn.metrics import accuracy_score

from core_es import compute_activations
from data import load_dataset
from logistic_output import binary_log_loss_from_proba, labels_to_binary, predict_pm1, predict_proba
from train import train
from train_ga_conventional import train_conventional_ga


DATASETS = [
    {"id": 59, "name": "ionosphere", "label": "Ionosphere"},
    {"id": 40, "name": "sonar", "label": "Sonar"},
    {"id": 15, "name": "breast-w", "label": "Breast-W"},
]

SEEDS = (42, 43, 44)
EPOCHS = 100
MU = 50
HIDDEN_UNITS = 50


def evaluate_es_state(
    state,
    data_id: int,
    data_name: str,
    seed: int,
) -> tuple[float, float, float]:
    """
    Compute ES final test accuracy, train accuracy, and logistic log-loss.
    """
    X_tr, X_te, y_tr, y_te = load_dataset(
        data_id=data_id,
        data_name=data_name,
        seed=seed,
    )

    Phi_tr = compute_activations(state.alpha, X_tr)
    Phi_tr_bias = np.c_[Phi_tr, np.ones(len(y_tr))]
    train_proba = predict_proba(Phi_tr_bias, state.beta)
    train_pred = predict_pm1(Phi_tr_bias, state.beta)
    train_acc = float(accuracy_score(y_tr.ravel(), train_pred))
    log_loss = binary_log_loss_from_proba(labels_to_binary(y_tr), train_proba)

    Phi_te = compute_activations(state.alpha, X_te)
    Phi_te_bias = np.c_[Phi_te, np.ones(len(y_te))]
    test_pred = predict_pm1(Phi_te_bias, state.beta)
    test_acc = float(accuracy_score(y_te.ravel(), test_pred))

    return test_acc, train_acc, log_loss


def _mean_std(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return f"{arr.mean():.4f} +/- {arr.std(ddof=0):.4f}"


def run_benchmark() -> None:
    rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        data_id = int(dataset["id"])
        data_name = str(dataset["name"])
        data_label = str(dataset["label"])
        print(f"\nRunning dataset {data_id} ({data_label})...")
        dataset_runs: list[dict[str, float | int]] = []

        for seed in SEEDS:
            print(f"  seed={seed}")

            # ---------------------------
            # ES benchmark (run first)
            # ---------------------------
            es_t0 = time.perf_counter()
            with redirect_stdout(io.StringIO()):
                es_state = train(
                    fitness_strategy="omp",
                    epochs=EPOCHS,
                    mu=MU,
                    data_id=data_id,
                    data_name=data_name,
                    verbose=999,  # minimises per-generation logging
                    seed=seed,
                )
            es_elapsed = time.perf_counter() - es_t0
            es_test_acc, es_best_train_acc, es_best_log_loss = evaluate_es_state(
                es_state, data_id=data_id, data_name=data_name, seed=seed
            )

            # Memory cleanup before GA.
            del es_state
            gc.collect()

            # ---------------------------
            # GA benchmark
            # ---------------------------
            ga_t0 = time.perf_counter()
            with redirect_stdout(io.StringIO()):
                ga_out = train_conventional_ga(
                    generations=EPOCHS,
                    hidden_units=HIDDEN_UNITS,
                    data_id=data_id,
                    data_name=data_name,
                    verbose=999,  # minimises per-generation logging
                    seed=seed,
                )
            ga_elapsed = time.perf_counter() - ga_t0

            dataset_runs.append(
                {
                    "seed": seed,
                    "es_time": es_elapsed,
                    "es_test_acc": es_test_acc,
                    "es_best_train_acc": es_best_train_acc,
                    "es_best_log_loss": es_best_log_loss,
                    "ga_time": ga_elapsed,
                    "ga_test_acc": float(ga_out["test_acc"]),
                    "ga_best_train_acc": float(ga_out["best_train_acc"]),
                    "ga_best_log_loss": float(ga_out["best_log_loss"]),
                }
            )

            # Extra cleanup between runs.
            del ga_out
            gc.collect()

        rows.append(
            {
                "dataset_id": data_id,
                "dataset_name": data_label,
                "runs": dataset_runs,
            }
        )

    # ------------------------------------------------------------
    # Main comparative table requested: Time and Test Accuracy
    # ------------------------------------------------------------
    print("\n## Level 1 Benchmark Results")
    print()
    print("| Dataset (OpenML ID) | Seeds | ES Time (s) | ES Test Acc | GA Time (s) | GA Test Acc |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        runs = r["runs"]
        print(
            f"| {r['dataset_name']} ({r['dataset_id']}) | "
            f"{len(runs)} | "
            f"{_mean_std([float(run['es_time']) for run in runs])} | "
            f"{_mean_std([float(run['es_test_acc']) for run in runs])} | "
            f"{_mean_std([float(run['ga_time']) for run in runs])} | "
            f"{_mean_std([float(run['ga_test_acc']) for run in runs])} |"
        )

    # ------------------------------------------------------------
    # Secondary table: Best Train Accuracy and Best LogLoss
    # ------------------------------------------------------------
    print()
    print("| Dataset (OpenML ID) | ES Best Train Acc | ES Best LogLoss | GA Best Train Acc | GA Best LogLoss |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        runs = r["runs"]
        print(
            f"| {r['dataset_name']} ({r['dataset_id']}) | "
            f"{_mean_std([float(run['es_best_train_acc']) for run in runs])} | "
            f"{_mean_std([float(run['es_best_log_loss']) for run in runs])} | "
            f"{_mean_std([float(run['ga_best_train_acc']) for run in runs])} | "
            f"{_mean_std([float(run['ga_best_log_loss']) for run in runs])} |"
        )


if __name__ == "__main__":
    run_benchmark()
