"""
train.py — Main training loop for the Ridge/GCV Michigan-style ES MLP.

The hidden layer is trained with a Michigan-style Evolution Strategy: each
individual is one tanh hidden neuron. Once a subset of neurons survives each
generation, the hidden-to-output (H2O) layer is solved analytically with Ridge
regression and Generalised Cross-Validation (GCV).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from data import load_dataset, load_dataset_with_validation
from core_es import MichiganESState, compute_activations, mutate_population
from gcv import compute_ridge_gcv
from selection import (
    cfs_auc_selection,
    cfs_selection,
    diversity_penalised_selection,
    omp_residual_selection,
    r2_fitness,
)


def predict_scores(Phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return Ridge output scores for a design matrix that includes bias."""
    return np.asarray(Phi, dtype=np.float64) @ np.asarray(beta, dtype=np.float64)


def predict_pm1(Phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Threshold Ridge scores at zero to obtain {-1,+1} predictions."""
    scores = predict_scores(Phi, beta).reshape(-1)
    return np.where(scores >= 0.0, 1.0, -1.0)


def _binary_score_metrics(
    y_pm1: np.ndarray,
    scores: np.ndarray,
    pred_pm1: np.ndarray,
) -> dict[str, float | None]:
    """Metrics for Ridge scores on {-1,+1} binary classification labels."""
    y = np.asarray(y_pm1, dtype=np.float64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_pm1, dtype=np.float64).reshape(-1)
    metrics: dict[str, float | None] = {
        "acc": float(accuracy_score(y, pred)),
        "mse": float(mean_squared_error(y, s)),
        "score_mean": float(np.mean(s)),
        "score_std": float(np.std(s)),
    }
    if len(np.unique(y)) == 2:
        y01 = (y > 0.0).astype(np.float64)
        metrics["roc_auc"] = float(roc_auc_score(y01, s))
    else:
        metrics["roc_auc"] = None
    return metrics


def train(
    fitness_strategy: str = "omp",
    epochs: int = 100,
    mu: int = 50,
    n_offspring: int = 350,
    sigma_init: float = 0.3,
    gamma: float = 0.3,
    data_id: int = 59,
    data_name: str | None = None,
    data_version: int = 1,
    test_size: float = 0.2,
    validation_size: float = 0.0,
    data_home: str | None = ".openml_cache",
    verbose: int = 10,
    seed: int = 42,
    overfit: bool = False,
    overfit_n_samples: int = 60,
    overfit_mu: int = 50,
    overfit_lambda: float = 1e-8,
    select_best_validation: bool = True,
    save_dir: str | Path | None = None,
) -> MichiganESState:
    """
    Run the Michigan-style ES with an analytic Ridge+GCV output layer.

    If a validation split is present, the final test metrics are computed from
    the generation with highest validation accuracy, using validation MSE as
    tie-breaker.
    """
    valid_strategies = {"omp", "r2", "cfs", "cfs_auc"}
    if fitness_strategy not in valid_strategies:
        raise ValueError(
            f"Unknown fitness_strategy={fitness_strategy!r}. "
            f"Choose one of {sorted(valid_strategies)}."
        )
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if mu < 1:
        raise ValueError("mu must be >= 1.")
    if n_offspring < 1:
        raise ValueError("n_offspring must be >= 1.")
    if verbose < 1:
        raise ValueError("verbose must be >= 1.")

    np.random.seed(seed)
    total_t0 = time.perf_counter()

    X_val = None
    y_val = None
    if validation_size > 0.0:
        X_tr, X_val, X_te, y_tr, y_val, y_te = load_dataset_with_validation(
            data_id=data_id,
            data_name=data_name,
            version=data_version,
            validation_size=validation_size,
            test_size=test_size,
            seed=seed,
            data_home=data_home,
        )
    else:
        X_tr, X_te, y_tr, y_te = load_dataset(
            data_id=data_id,
            data_name=data_name,
            version=data_version,
            test_size=test_size,
            seed=seed,
            data_home=data_home,
        )

    if overfit:
        rng_ov = np.random.default_rng(seed + 1)
        n_available = X_tr.shape[0]
        n_subset = min(overfit_n_samples, n_available)
        subset_idx = rng_ov.choice(n_available, size=n_subset, replace=False)
        X_tr = X_tr[subset_idx]
        y_tr = y_tr[subset_idx]
        print(
            f"[OVERFIT MODE] Training restricted to {n_subset} samples "
            f"(from {n_available}). mu={overfit_mu}, ridge_lambda={overfit_lambda}"
        )

    effective_mu = overfit_mu if overfit else mu
    state = MichiganESState(
        mu=effective_mu, input_size=X_tr.shape[1], sigma_init=sigma_init
    )

    print(
        f"\nMichigan ES Ridge | strategy={fitness_strategy!r} | "
        f"mu={effective_mu} | lambda={n_offspring} | sigma_0={sigma_init}"
        f" | overfit={'ON' if overfit else 'off'}\n"
        + "-" * 65
    )

    gcv_history: list[float] = []
    select_ms_history: list[float] = []
    epoch_ms_history: list[float] = []
    history: list[dict[str, float | int | str | None]] = []
    best_checkpoint: dict[str, object] | None = None

    STAGNATION_WINDOW = 15
    STAGNATION_THRESHOLD = 0.005
    SIGMA_RESET_FRACTION = 0.25

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()

        off_alpha, off_L = mutate_population(state, n_offspring=n_offspring)
        pool_alpha = np.vstack([state.alpha, off_alpha])
        pool_L = np.vstack([state.L, off_L])
        Phi_pool = compute_activations(pool_alpha, X_tr)
        pool_r2 = r2_fitness(Phi_pool, y_tr)

        effective_strategy = "r2" if overfit else fitness_strategy
        effective_gamma = 0.0 if overfit else gamma

        select_t0 = time.perf_counter()
        if effective_strategy == "omp":
            best_idx = omp_residual_selection(Phi_pool, y_tr, mu=effective_mu)
        elif effective_strategy == "r2":
            best_idx = diversity_penalised_selection(
                pool_r2, Phi_pool, mu=effective_mu, gamma=effective_gamma
            )
        elif effective_strategy == "cfs":
            best_idx = cfs_selection(Phi_pool, y_tr, mu=effective_mu)
        elif effective_strategy == "cfs_auc":
            best_idx = cfs_auc_selection(Phi_pool, y_tr, mu=effective_mu)
        else:
            raise ValueError(f"Unknown fitness_strategy={fitness_strategy!r}.")
        select_ms = (time.perf_counter() - select_t0) * 1e3
        select_ms_history.append(select_ms)

        if not overfit:
            ELITE_K = max(1, effective_mu // 10)
            if epoch > 1 and state.beta is not None:
                elite_pool_idx = np.argsort(pool_r2[:effective_mu])[-ELITE_K:]
                best_idx = best_idx.tolist()
                best_set = set(best_idx)
                elite_set = set(elite_pool_idx.tolist())
                surv_scores = pool_r2[np.array(best_idx, dtype=np.int64)]
                replace_positions = [
                    pos for pos, sidx in enumerate(best_idx) if sidx not in elite_set
                ]
                replace_positions.sort(key=lambda pos: surv_scores[pos])
                replace_ptr = 0
                for eidx in elite_pool_idx:
                    if eidx not in best_set and replace_ptr < len(replace_positions):
                        rep_pos = replace_positions[replace_ptr]
                        best_idx[rep_pos] = eidx
                        best_set.add(eidx)
                        replace_ptr += 1
                best_idx = np.array(best_idx, dtype=np.int64)

        state.alpha = pool_alpha[best_idx].copy()
        state.L = pool_L[best_idx].copy()

        Phi_surv = Phi_pool[:, best_idx]
        Phi_surv_bias = np.c_[Phi_surv, np.ones(len(y_tr))]
        best_lam, best_beta, best_gcv = compute_ridge_gcv(
            Phi_surv_bias,
            y_tr,
            lambda_override=(overfit_lambda if overfit else None),
        )
        state.beta = best_beta
        gcv_metric = float(best_gcv) if np.isfinite(best_gcv) else float(best_lam)
        gcv_history.append(gcv_metric)

        train_scores = predict_scores(Phi_surv_bias, state.beta)
        train_pred = predict_pm1(Phi_surv_bias, state.beta)
        train_epoch_metrics = _binary_score_metrics(y_tr, train_scores, train_pred)
        train_acc = float(train_epoch_metrics["acc"])

        val_epoch_metrics = None
        val_acc_epoch = None
        val_mse_epoch = None
        if X_val is not None and y_val is not None:
            Phi_val_epoch = compute_activations(state.alpha, X_val)
            Phi_val_bias_epoch = np.c_[Phi_val_epoch, np.ones(len(y_val))]
            val_scores_epoch = predict_scores(Phi_val_bias_epoch, state.beta)
            val_pred_epoch = predict_pm1(Phi_val_bias_epoch, state.beta)
            val_epoch_metrics = _binary_score_metrics(
                y_val, val_scores_epoch, val_pred_epoch
            )
            val_acc_epoch = float(val_epoch_metrics["acc"])
            val_mse_epoch = float(val_epoch_metrics["mse"])

            if select_best_validation:
                is_better = (
                    best_checkpoint is None
                    or val_acc_epoch > float(best_checkpoint["validation_acc"]) + 1e-12
                    or (
                        abs(val_acc_epoch - float(best_checkpoint["validation_acc"]))
                        <= 1e-12
                        and val_mse_epoch < float(best_checkpoint["validation_mse"])
                    )
                )
                if is_better:
                    best_checkpoint = {
                        "epoch": epoch,
                        "alpha": state.alpha.copy(),
                        "L": state.L.copy(),
                        "beta": state.beta.copy(),
                        "lambda": float(best_lam),
                        "gcv": float(best_gcv),
                        "validation_acc": val_acc_epoch,
                        "validation_mse": val_mse_epoch,
                    }

        mean_sigma = float(state.L[:, np.arange(state.D), np.arange(state.D)].mean())

        reset_count = 0
        if not overfit and len(gcv_history) >= STAGNATION_WINDOW:
            recent_improvement = gcv_history[-STAGNATION_WINDOW] - gcv_history[-1]
            relative_improvement = recent_improvement / (
                abs(gcv_history[-STAGNATION_WINDOW]) + 1e-12
            )
            if relative_improvement < STAGNATION_THRESHOLD:
                scores_surv = pool_r2[best_idx]
                n_reset = max(1, int(effective_mu * SIGMA_RESET_FRACTION))
                worst_idx = np.argsort(scores_surv)[:n_reset]
                state.L[worst_idx] = np.stack(
                    [
                        np.eye(state.D, dtype=np.float64) * sigma_init
                        for _ in range(n_reset)
                    ]
                )
                reset_count = n_reset
                if epoch % verbose == 0:
                    print(
                        f"  [stagnation reset: {n_reset} neurons re-initialised "
                        f"| relative_improvement={relative_improvement:.6f}]"
                    )

        epoch_ms = (time.perf_counter() - epoch_t0) * 1e3
        epoch_ms_history.append(epoch_ms)
        row: dict[str, float | int | str | None] = {
            "epoch": epoch,
            "strategy": effective_strategy,
            "gcv": float(best_gcv),
            "lambda": float(best_lam),
            "train_acc": train_acc,
            "train_mse": train_epoch_metrics["mse"],
            "train_roc_auc": train_epoch_metrics["roc_auc"],
            "train_score_mean": train_epoch_metrics["score_mean"],
            "train_score_std": train_epoch_metrics["score_std"],
            "sigma_mean": mean_sigma,
            "select_ms": float(select_ms),
            "epoch_ms": float(epoch_ms),
            "reset_count": reset_count,
        }
        if val_epoch_metrics is not None:
            row["validation_acc"] = val_acc_epoch
            row["validation_mse"] = val_mse_epoch
            row["validation_roc_auc"] = val_epoch_metrics["roc_auc"]
            row["validation_score_mean"] = val_epoch_metrics["score_mean"]
            row["validation_score_std"] = val_epoch_metrics["score_std"]
        history.append(row)

        if epoch % verbose == 0 or epoch == 1:
            val_msg = "  |  "
            if val_acc_epoch is not None:
                val_msg = f"  |  Val Acc={val_acc_epoch:.4f}  |  "
            print(
                f"Gen {epoch:4d}/{epochs}  |  "
                f"GCV={best_gcv:.5f}  |  "
                f"ridge_lambda={best_lam:.2e}  |  "
                f"Train Acc={train_acc:.4f}"
                f"{val_msg}"
                f"sigma_mean={mean_sigma:.5f}  |  "
                f"select_ms={select_ms:.1f}  |  "
                f"epoch_ms={epoch_ms:.1f}"
            )

    selected_epoch = int(history[-1]["epoch"])
    selected_by_validation = False
    if best_checkpoint is not None and select_best_validation:
        state.alpha = np.asarray(best_checkpoint["alpha"]).copy()
        state.L = np.asarray(best_checkpoint["L"]).copy()
        state.beta = np.asarray(best_checkpoint["beta"]).copy()
        selected_epoch = int(best_checkpoint["epoch"])
        selected_by_validation = True

    Phi_train_final = compute_activations(state.alpha, X_tr)
    Phi_train_bias_final = np.c_[Phi_train_final, np.ones(len(y_tr))]
    train_scores_final = predict_scores(Phi_train_bias_final, state.beta)
    train_pred_final = predict_pm1(Phi_train_bias_final, state.beta)
    train_final_metrics = _binary_score_metrics(
        y_tr, train_scores_final, train_pred_final
    )

    Phi_test = compute_activations(state.alpha, X_te)
    Phi_test_bias = np.c_[Phi_test, np.ones(len(y_te))]
    test_scores = predict_scores(Phi_test_bias, state.beta)
    test_pred = predict_pm1(Phi_test_bias, state.beta)
    test_metrics = _binary_score_metrics(y_te, test_scores, test_pred)

    val_acc = None
    val_metrics = None
    if X_val is not None and y_val is not None:
        Phi_val = compute_activations(state.alpha, X_val)
        Phi_val_bias = np.c_[Phi_val, np.ones(len(y_val))]
        val_scores = predict_scores(Phi_val_bias, state.beta)
        val_pred = predict_pm1(Phi_val_bias, state.beta)
        val_metrics = _binary_score_metrics(y_val, val_scores, val_pred)
        val_acc = float(val_metrics["acc"])

    selected_row = history[selected_epoch - 1]
    total_elapsed = time.perf_counter() - total_t0
    final_metrics = {
        "architecture": "ridge_gcv",
        "strategy": fitness_strategy,
        "effective_mu": effective_mu,
        "data_id": data_id,
        "data_name": data_name,
        "data_version": data_version,
        "data_home": data_home,
        "validation_size": validation_size,
        "seed": seed,
        "epochs": epochs,
        "n_offspring": n_offspring,
        "overfit": overfit,
        "selected_epoch": selected_epoch,
        "selected_by_validation": selected_by_validation,
        "final_train_acc": train_final_metrics["acc"],
        "train_mse": train_final_metrics["mse"],
        "train_roc_auc": train_final_metrics["roc_auc"],
        "validation_acc": val_acc,
        "validation_mse": None if val_metrics is None else val_metrics["mse"],
        "validation_roc_auc": None if val_metrics is None else val_metrics["roc_auc"],
        "test_acc": test_metrics["acc"],
        "test_mse": test_metrics["mse"],
        "test_roc_auc": test_metrics["roc_auc"],
        "final_gcv": selected_row["gcv"],
        "final_lambda": selected_row["lambda"],
        "test_score_mean": test_metrics["score_mean"],
        "test_score_std": test_metrics["score_std"],
        "elapsed_seconds": float(total_elapsed),
        "select_ms_mean": float(np.mean(select_ms_history)),
        "select_ms_median": float(np.median(select_ms_history)),
        "epoch_ms_mean": float(np.mean(epoch_ms_history)),
        "epoch_ms_median": float(np.median(epoch_ms_history)),
    }
    state.history = history
    state.metrics = final_metrics

    print(f"\n{'=' * 65}")
    print(f"  Strategy     : {fitness_strategy!r}")
    if selected_by_validation:
        print(f"  Selected Gen : {selected_epoch} (best validation checkpoint)")
    if val_acc is not None:
        print(f"  Val Accuracy : {val_acc:.4f}")
    print(f"  Test Accuracy: {float(test_metrics['acc']):.4f}")
    print(f"  Test MSE     : {float(test_metrics['mse']):.6f}")
    print(
        f"  Timing (ms)  : select mean={np.mean(select_ms_history):.1f}, "
        f"median={np.median(select_ms_history):.1f} | "
        f"epoch mean={np.mean(epoch_ms_history):.1f}, "
        f"median={np.median(epoch_ms_history):.1f}"
    )
    print(f"{'=' * 65}\n")

    if save_dir is not None:
        _save_run_artifacts(Path(save_dir), state, final_metrics, history)

    return state


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


def _save_run_artifacts(
    save_dir: Path,
    state: MichiganESState,
    metrics: dict[str, object],
    history: list[dict[str, object]],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"ridge_es_{metrics['strategy']}_"
        f"data{metrics['data_id']}_"
        f"seed{metrics['seed']}_"
        f"epochs{metrics['epochs']}"
    )
    npz_path = save_dir / f"{stem}.npz"
    json_path = save_dir / f"{stem}.json"

    np.savez_compressed(
        npz_path,
        alpha=state.alpha,
        L=state.L,
        beta=state.beta,
    )
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            _json_safe({"metrics": metrics, "history": history}),
            fh,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved run artifacts: {npz_path} | {json_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Ridge/GCV Michigan ES MLP.")
    parser.add_argument(
        "--strategy",
        choices=["omp", "r2", "cfs", "cfs_auc"],
        default="omp",
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
    parser.add_argument("--validation-size", type=float, default=0.0)
    parser.add_argument("--data-home", default=".openml_cache")
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit-n-samples", type=int, default=60)
    parser.add_argument("--overfit-mu", type=int, default=50)
    parser.add_argument("--overfit-lambda", type=float, default=1e-8)
    parser.add_argument(
        "--no-select-best-validation",
        action="store_true",
        help=(
            "When validation is enabled, keep the final generation instead of "
            "restoring the best validation checkpoint."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(
        fitness_strategy=args.strategy,
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
        overfit=args.overfit,
        overfit_n_samples=args.overfit_n_samples,
        overfit_mu=args.overfit_mu,
        overfit_lambda=args.overfit_lambda,
        select_best_validation=not args.no_select_best_validation,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
