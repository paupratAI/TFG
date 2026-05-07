"""
train.py — Main training loop for the Michigan-style Evolution Strategy MLP.

This script orchestrates the full training procedure:
  1. Load and preprocess data.
  2. Initialise the ES population (one neuron per individual).
  3. Each generation: mutate -> activate -> select -> solve output layer.
  4. Report training accuracy, logistic log-loss, and sigma at regular intervals.
  5. Report final test accuracy.

Usage
-----
    python train.py --strategy omp --data-id 59 --epochs 100 --save-dir runs

Performance notes
-----------------
- r2_fitness is called ONCE per generation over the full pool (pool_r2).
  The stagnation-reset block reuses pool_r2[best_idx] instead of making
  a second r2_fitness call over Phi_surv.
- The output layer is a logistic classifier.  Its linear score is a logit,
  so the model returns binary class probabilities instead of a raw
  regression score.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from data      import load_dataset, load_dataset_with_validation
from core_es   import MichiganESState, mutate_population, compute_activations
from selection import (
    omp_residual_selection,
    r2_fitness,
    diversity_penalised_selection,
    cfs_selection,
    cfs_auc_selection,
)
from logistic_output import (
    binary_log_loss_from_proba,
    compute_logistic_output,
    labels_to_binary,
    predict_pm1,
    predict_proba,
)


def expected_calibration_error(
    y01: np.ndarray,
    proba_pos: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute a binary Expected Calibration Error (ECE).

    The confidence of a binary prediction is max(p, 1-p). The prediction is
    correct when the thresholded probability agrees with the binary target.
    ECE then averages |accuracy(bin) - confidence(bin)| weighted by bin size.
    """
    y = np.asarray(y01, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(proba_pos, dtype=np.float64).reshape(-1), 0.0, 1.0)
    pred = (p >= 0.5).astype(np.float64)
    confidence = np.maximum(p, 1.0 - p)
    correct = (pred == y).astype(np.float64)

    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        if not np.any(mask):
            continue
        bin_weight = float(np.mean(mask))
        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(confidence[mask]))
        ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


def _binary_metrics(
    y_pm1: np.ndarray,
    proba_pos: np.ndarray,
    pred_pm1: np.ndarray,
) -> dict[str, float | None]:
    """Return accuracy and probability-quality metrics for {-1,+1} labels."""
    y01 = labels_to_binary(y_pm1).reshape(-1)
    p = np.asarray(proba_pos, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_pm1, dtype=np.float64).reshape(-1)

    metrics: dict[str, float | None] = {
        "acc": float(accuracy_score(np.asarray(y_pm1).reshape(-1), pred)),
        "log_loss": binary_log_loss_from_proba(y01, p),
        "brier": float(brier_score_loss(y01, p)),
        "ece": expected_calibration_error(y01, p),
        "mean_positive_proba": float(np.mean(p)),
    }
    if len(np.unique(y01)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y01, p))
    else:
        metrics["roc_auc"] = None
    return metrics


def train(
    fitness_strategy: str   = "omp",
    epochs          : int   = 100,
    mu              : int   = 50,
    n_offspring     : int   = 350,
    sigma_init      : float = 0.3,
    gamma           : float = 0.3,
    data_id         : int   = 59,
    data_name       : str | None = None,
    data_version    : int   = 1,
    test_size       : float = 0.2,
    validation_size : float = 0.0,
    data_home       : str | None = ".openml_cache",
    verbose         : int   = 10,
    seed            : int   = 42,
    overfit          : bool  = False,
    overfit_n_samples: int   = 60,
    overfit_mu       : int   = 50,
    select_best_validation: bool = True,
    save_dir         : str | Path | None = None,
) -> MichiganESState:
    """
    Run the Michigan-style (mu + lambda)-ES training loop.

    Each generation follows four steps:

        Mutate   ->  produce n_offspring candidate neurons from the current
                     population (mu parents + lambda offspring = pool of 400).

        Activate ->  compute tanh hidden activations for all pool neurons.

        Select   ->  pick the best mu neurons from the pool.
                    'omp' uses sequential residual fitting (recommended).
                    'r2'  uses marginal R² + diversity penalty (ablation).
                    'cfs' uses subset merit balancing relevance/redundancy.
                    'cfs_auc' uses AUC relevance + Spearman redundancy.

        Solve    ->  fit an unregularised logistic output layer on the mu
                     survivors.  The output score is interpreted as a logit.

    Note on the bias column and probabilities
    -----------------------------------------
    A column of ones is appended to the activation matrix ONLY in the Solve
    step, immediately before the logistic fit.  All prior steps (fitness
    scoring, selection) work on raw activations without the bias column.
    The fitted model returns p(y=+1|X); p(y=-1|X) is 1-p.

    Parameters
    ----------
    fitness_strategy : 'omp' (default), 'r2' (ablation), 'cfs', or 'cfs_auc'.
    epochs           : Number of ES generations to run.
    mu               : Population size = hidden layer width.
    n_offspring      : Offspring produced per generation (lambda).
    sigma_init       : Initial mutation step size (L diagonal value).
    gamma            : Diversity penalty weight, used only for 'r2' strategy.
    data_id          : OpenML dataset identifier.
    data_name        : Optional OpenML dataset name. If provided, overrides data_id.
    data_version     : OpenML version used when data_name is provided.
    test_size        : Fraction of data held out for testing.
    validation_size  : Fraction of the non-test data held out for validation.
                       Set to 0.0 to disable validation.
    data_home        : Local OpenML cache directory.
    verbose          : Print a log line every this many generations.
    seed             : Global random seed.
    overfit          : If True, enable memorisation demo on a small subset.
    overfit_n_samples: Training rows when overfit=True (capped by available).
    overfit_mu       : Population size when overfit=True (replaces mu).
    select_best_validation:
                       If validation is enabled, restore the generation with the
                       best validation accuracy before final test evaluation.
                       Validation log-loss is used as a tie-breaker.
    save_dir         : Optional directory where model arrays and JSON metadata are saved.

    Returns
    -------
    state : MichiganESState with the final evolved population and output weights.
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

    # --- Load data ---
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

    # --- Overfit mode: restrict training set to a tiny random subset ---
    if overfit:
        rng_ov = np.random.default_rng(seed + 1)
        n_available = X_tr.shape[0]
        n_subset = min(overfit_n_samples, n_available)
        subset_idx = rng_ov.choice(n_available, size=n_subset, replace=False)
        X_tr = X_tr[subset_idx]
        y_tr = y_tr[subset_idx]
        print(
            f"[OVERFIT MODE] Training restricted to {n_subset} samples "
            f"(from {n_available}). mu={overfit_mu}"
        )

    effective_mu = overfit_mu if overfit else mu

    # --- Initialise population ---
    state = MichiganESState(
        mu=effective_mu, input_size=X_tr.shape[1], sigma_init=sigma_init
    )

    print(
        f"\nMichigan ES | strategy={fitness_strategy!r} | "
        f"mu={effective_mu} | lambda={n_offspring} | sigma_0={sigma_init}"
        f" | overfit={'ON' if overfit else 'off'}\n"
        + "-" * 65
    )
    log_loss_history: list[float] = []
    select_ms_history: list[float] = []
    epoch_ms_history: list[float] = []
    history: list[dict[str, float | int | str]] = []
    best_checkpoint: dict[str, object] | None = None
    STAGNATION_WINDOW    = 15
    STAGNATION_THRESHOLD = 0.005
    SIGMA_RESET_FRACTION = 0.25

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()

        # --- Step 1: Mutate ---
        off_alpha, off_L = mutate_population(state, n_offspring=n_offspring)
        pool_alpha = np.vstack([state.alpha, off_alpha])   # (P, D)
        pool_L     = np.vstack([state.L,     off_L    ])   # (P, D, D)

        # --- Step 2: Activate ---
        # Phi_pool columns are neurons, rows are training samples.
        # No bias column here -- it is added only before the logistic output fit.
        Phi_pool = compute_activations(pool_alpha, X_tr)   # (m, P)

        # Compute r2 ONCE for the full pool; reused by selection and elitism.
        pool_r2 = r2_fitness(Phi_pool, y_tr)               # (P,)

        # In overfit mode, force pure R² selection (no diversity penalty)
        effective_strategy = "r2" if overfit else fitness_strategy
        effective_gamma = 0.0 if overfit else gamma

        # --- Step 3: Select ---
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
            raise ValueError(
                f"Unknown fitness_strategy={fitness_strategy!r}. "
                f"Choose one of {sorted(valid_strategies)}."
            )
        select_ms = (time.perf_counter() - select_t0) * 1e3
        select_ms_history.append(select_ms)

        # --- Weak elitism: guarantee top-k parents survive selection ---
        # Disabled in overfit mode: maximum plasticity needed for memorisation.
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
                    if eidx not in best_set:
                        if replace_ptr < len(replace_positions):
                            rep_pos = replace_positions[replace_ptr]
                            best_idx[rep_pos] = eidx
                            best_set.add(eidx)
                            replace_ptr += 1

                best_idx = np.array(best_idx, dtype=np.int64)

        # Survivors become the new population.
        state.alpha = pool_alpha[best_idx].copy()   # (mu, D)
        state.L     = pool_L[best_idx].copy()       # (mu, D, D)

        # --- Step 4: Solve output layer (unregularised Logistic) ---
        # Append the bias column here, right before fitting the logistic output.
        Phi_surv      = Phi_pool[:, best_idx]                   # (m, mu)
        Phi_surv_bias = np.c_[Phi_surv, np.ones(len(y_tr))]    # (m, mu+1)

        best_beta, best_log_loss, step_norm = compute_logistic_output(Phi_surv_bias, y_tr)
        state.beta = best_beta   # (mu+1, 1)
        stagnation_metric = -float(np.max(pool_r2[best_idx])) if overfit else best_log_loss
        log_loss_history.append(stagnation_metric)

        y_proba_tr = predict_proba(Phi_surv_bias, state.beta)
        y_pred_tr = predict_pm1(Phi_surv_bias, state.beta)
        train_epoch_metrics = _binary_metrics(y_tr, y_proba_tr, y_pred_tr)
        train_acc = float(train_epoch_metrics["acc"])
        val_acc_epoch = None
        val_log_loss_epoch = None
        val_epoch_metrics = None
        if X_val is not None and y_val is not None:
            Phi_val_epoch = compute_activations(state.alpha, X_val)
            Phi_val_bias_epoch = np.c_[Phi_val_epoch, np.ones(len(y_val))]
            val_proba_epoch = predict_proba(Phi_val_bias_epoch, state.beta)
            val_pred_epoch = predict_pm1(Phi_val_bias_epoch, state.beta)
            val_epoch_metrics = _binary_metrics(y_val, val_proba_epoch, val_pred_epoch)
            val_acc_epoch = float(val_epoch_metrics["acc"])
            val_log_loss_epoch = float(val_epoch_metrics["log_loss"])
            if select_best_validation:
                is_better = (
                    best_checkpoint is None
                    or val_acc_epoch > float(best_checkpoint["validation_acc"]) + 1e-12
                    or (
                        abs(val_acc_epoch - float(best_checkpoint["validation_acc"])) <= 1e-12
                        and val_log_loss_epoch < float(best_checkpoint["validation_log_loss"])
                    )
                )
                if is_better:
                    best_checkpoint = {
                        "epoch": epoch,
                        "alpha": state.alpha.copy(),
                        "L": state.L.copy(),
                        "beta": state.beta.copy(),
                        "validation_acc": val_acc_epoch,
                        "validation_log_loss": val_log_loss_epoch,
                        "train_acc": train_acc,
                        "train_log_loss": float(best_log_loss),
                    }
        mean_sigma = float(state.L[:, np.arange(state.D), np.arange(state.D)].mean())

        # --- Stagnation detection and sigma reset ---
        reset_count = 0
        if not overfit:
            if len(log_loss_history) >= STAGNATION_WINDOW:
                recent_improvement = log_loss_history[-STAGNATION_WINDOW] - log_loss_history[-1]
                relative_improvement = recent_improvement / (abs(log_loss_history[-STAGNATION_WINDOW]) + 1e-12)
                if relative_improvement < STAGNATION_THRESHOLD:
                    # Reuse pool_r2[best_idx] — same as r2 scores for survivors.
                    # No second r2_fitness call needed.
                    scores_surv = pool_r2[best_idx]
                    n_reset = max(1, int(effective_mu * SIGMA_RESET_FRACTION))
                    worst_idx = np.argsort(scores_surv)[:n_reset]
                    reset_sigma = sigma_init * 3.0 if overfit else sigma_init
                    state.L[worst_idx] = np.stack(
                        [np.eye(state.D, dtype=np.float64) * reset_sigma for _ in range(n_reset)]
                    )
                    reset_count = n_reset
                    if epoch % verbose == 0:
                        print(
                            f"  [stagnation reset: {n_reset} neurons re-initialised "
                            f"| relative_improvement={relative_improvement:.6f}]"
                        )

        # --- Logging ---
        epoch_ms = (time.perf_counter() - epoch_t0) * 1e3
        epoch_ms_history.append(epoch_ms)
        row: dict[str, float | int | str | None] = {
            "epoch": epoch,
            "strategy": effective_strategy,
            "log_loss": float(best_log_loss),
            "newton_step_norm": float(step_norm),
            "train_acc": train_acc,
            "train_brier": train_epoch_metrics["brier"],
            "train_roc_auc": train_epoch_metrics["roc_auc"],
            "train_ece": train_epoch_metrics["ece"],
            "train_mean_positive_proba": train_epoch_metrics["mean_positive_proba"],
            "sigma_mean": mean_sigma,
            "select_ms": float(select_ms),
            "epoch_ms": float(epoch_ms),
            "reset_count": reset_count,
        }
        if val_epoch_metrics is not None:
            row["validation_acc"] = val_acc_epoch
            row["validation_log_loss"] = val_log_loss_epoch
            row["validation_brier"] = val_epoch_metrics["brier"]
            row["validation_roc_auc"] = val_epoch_metrics["roc_auc"]
            row["validation_ece"] = val_epoch_metrics["ece"]
            row["validation_mean_positive_proba"] = val_epoch_metrics["mean_positive_proba"]
        history.append(row)
        if epoch % verbose == 0 or epoch == 1:
            val_msg = "  |  "
            if val_acc_epoch is not None:
                val_msg = f"  |  Val Acc={val_acc_epoch:.4f}  |  "
            print(
                f"Gen {epoch:4d}/{epochs}  |  "
                f"LogLoss={best_log_loss:.5f}  |  "
                f"step_norm={step_norm:.1e}  |  "
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

    # --- Final train/validation/test evaluation with the selected checkpoint ---
    Phi_train_final = compute_activations(state.alpha, X_tr)
    Phi_train_bias_final = np.c_[Phi_train_final, np.ones(len(y_tr))]
    train_proba_final = predict_proba(Phi_train_bias_final, state.beta)
    train_pred_final = predict_pm1(Phi_train_bias_final, state.beta)
    train_final_metrics = _binary_metrics(y_tr, train_proba_final, train_pred_final)

    # --- Final test evaluation ---
    Phi_test      = compute_activations(state.alpha, X_te)       # (m_te, mu)
    Phi_test_bias = np.c_[Phi_test, np.ones(len(y_te))]         # (m_te, mu+1)
    test_proba    = predict_proba(Phi_test_bias, state.beta)
    test_pred     = predict_pm1(Phi_test_bias, state.beta)
    test_metrics = _binary_metrics(y_te, test_proba, test_pred)
    val_acc = None
    val_metrics = None
    if X_val is not None and y_val is not None:
        Phi_val = compute_activations(state.alpha, X_val)
        Phi_val_bias = np.c_[Phi_val, np.ones(len(y_val))]
        val_proba = predict_proba(Phi_val_bias, state.beta)
        val_pred = predict_pm1(Phi_val_bias, state.beta)
        val_metrics = _binary_metrics(y_val, val_proba, val_pred)
        val_acc = float(val_metrics["acc"])
    total_elapsed = time.perf_counter() - total_t0
    final_metrics = {
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
        "train_log_loss": train_final_metrics["log_loss"],
        "train_brier": train_final_metrics["brier"],
        "train_roc_auc": train_final_metrics["roc_auc"],
        "train_ece": train_final_metrics["ece"],
        "validation_acc": val_acc,
        "validation_log_loss": None if val_metrics is None else val_metrics["log_loss"],
        "validation_brier": None if val_metrics is None else val_metrics["brier"],
        "validation_roc_auc": None if val_metrics is None else val_metrics["roc_auc"],
        "validation_ece": None if val_metrics is None else val_metrics["ece"],
        "test_acc": test_metrics["acc"],
        "test_log_loss": test_metrics["log_loss"],
        "test_brier": test_metrics["brier"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_ece": test_metrics["ece"],
        "final_log_loss": train_final_metrics["log_loss"],
        "final_newton_step_norm": history[selected_epoch - 1]["newton_step_norm"],
        "test_mean_positive_proba": test_metrics["mean_positive_proba"],
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
    if select_ms_history and epoch_ms_history:
        select_mean = float(np.mean(select_ms_history))
        select_med  = float(np.median(select_ms_history))
        epoch_mean  = float(np.mean(epoch_ms_history))
        epoch_med   = float(np.median(epoch_ms_history))
        print(
            f"  Timing (ms)  : select mean={select_mean:.1f}, median={select_med:.1f} "
            f"| epoch mean={epoch_mean:.1f}, median={epoch_med:.1f}"
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
        f"es_{metrics['strategy']}_"
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
    parser = argparse.ArgumentParser(description="Train the Michigan-style ES MLP.")
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
        select_best_validation=not args.no_select_best_validation,
        save_dir=args.save_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
