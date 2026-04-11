"""
benchmark_level1.py

Automated Level-1 benchmark for:
1) Michigan-style Evolution Strategy (train.py)
2) Conventional Genetic Algorithm (train_ga_conventional.py)

Datasets:
- 59  (Ionosphere)
- 40  (Sonar)
- 150 (Breast-W)
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
from gcv import compute_ridge_gcv
from train import train
from train_ga_conventional import train_conventional_ga


DATASETS = {
    59: "Ionosphere",
    40: "Sonar",
    150: "Breast-W",
}

SEED = 42
EPOCHS = 100
MU = 50
HIDDEN_UNITS = 50


def evaluate_es_state(state, data_id: int, seed: int) -> tuple[float, float, float]:
    """
    Compute ES final test accuracy, train accuracy, and GCV from final state.
    """
    X_tr, X_te, y_tr, y_te = load_dataset(data_id=data_id, seed=seed)

    Phi_tr = compute_activations(state.alpha, X_tr)
    Phi_tr_bias = np.c_[Phi_tr, np.ones(len(y_tr))]
    y_hat_tr = Phi_tr_bias @ state.beta
    train_pred = np.where(y_hat_tr.ravel() > 0, 1.0, -1.0)
    train_acc = float(accuracy_score(y_tr.ravel(), train_pred))

    # Recompute best lambda/GCV on final hidden representation for reporting.
    _, _, best_gcv = compute_ridge_gcv(Phi_tr_bias, y_tr)
    best_gcv = float(best_gcv)

    Phi_te = compute_activations(state.alpha, X_te)
    Phi_te_bias = np.c_[Phi_te, np.ones(len(y_te))]
    y_hat_te = Phi_te_bias @ state.beta
    test_pred = np.where(y_hat_te.ravel() > 0, 1.0, -1.0)
    test_acc = float(accuracy_score(y_te.ravel(), test_pred))

    return test_acc, train_acc, best_gcv


def run_benchmark() -> None:
    rows: list[dict[str, float | int | str]] = []

    for data_id, data_name in DATASETS.items():
        print(f"\nRunning dataset {data_id} ({data_name})...")

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
                verbose=999,  # minimises per-generation logging
                seed=SEED,
            )
        es_elapsed = time.perf_counter() - es_t0
        es_test_acc, es_best_train_acc, es_best_gcv = evaluate_es_state(
            es_state, data_id=data_id, seed=SEED
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
                verbose=999,  # minimises per-generation logging
                seed=SEED,
            )
        ga_elapsed = time.perf_counter() - ga_t0

        rows.append(
            {
                "dataset_id": data_id,
                "dataset_name": data_name,
                "es_time": es_elapsed,
                "es_test_acc": es_test_acc,
                "es_best_train_acc": es_best_train_acc,
                "es_best_gcv": es_best_gcv,
                "ga_time": ga_elapsed,
                "ga_test_acc": float(ga_out["test_acc"]),
                "ga_best_train_acc": float(ga_out["best_train_acc"]),
                "ga_best_gcv": float(ga_out["best_gcv"]),
            }
        )

        # Extra cleanup between datasets.
        del ga_out
        gc.collect()

    # ------------------------------------------------------------
    # Main comparative table requested: Time and Test Accuracy
    # ------------------------------------------------------------
    print("\n## Level 1 Benchmark Results")
    print()
    print("| Dataset (OpenML ID) | ES Time (s) | ES Test Acc | GA Time (s) | GA Test Acc |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['dataset_name']} ({r['dataset_id']}) | "
            f"{r['es_time']:.2f} | {r['es_test_acc']:.4f} | "
            f"{r['ga_time']:.2f} | {r['ga_test_acc']:.4f} |"
        )

    # ------------------------------------------------------------
    # Secondary table: Best Train Accuracy and Best GCV
    # ------------------------------------------------------------
    print()
    print("| Dataset (OpenML ID) | ES Best Train Acc | ES Best GCV | GA Best Train Acc | GA Best GCV |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['dataset_name']} ({r['dataset_id']}) | "
            f"{r['es_best_train_acc']:.4f} | {r['es_best_gcv']:.6f} | "
            f"{r['ga_best_train_acc']:.4f} | {r['ga_best_gcv']:.6f} |"
        )


if __name__ == "__main__":
    run_benchmark()
