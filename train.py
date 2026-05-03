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
  so the model returns calibrated binary probabilities instead of a raw
  regression score.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import accuracy_score

from data      import load_dataset, load_dataset_with_validation
from core_es   import MichiganESState, mutate_population, compute_activations
from selection import (
    omp_residual_selection,
    r2_fitness,
    diversity_penalised_selection,
    cfs_selection,
    cfs_auc_selection,
)
from logistic_output import compute_logistic_l2, predict_pm1, predict_proba


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
    overfit_lambda   : float = 1e-8,
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

        Solve    ->  fit an L2-regularised logistic output layer on the mu
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
    overfit          : If True, enable memorisation demo (subset train, tiny L2 penalty).
    overfit_n_samples: Training rows when overfit=True (capped by available).
    overfit_mu       : Population size when overfit=True (replaces mu).
    overfit_lambda   : Fixed logistic L2 penalty when overfit=True.
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
            f"(from {n_available}). mu={overfit_mu}, lambda={overfit_lambda}"
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
    objective_history: list[float] = []
    select_ms_history: list[float] = []
    epoch_ms_history: list[float] = []
    history: list[dict[str, float | int | str]] = []
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

        # --- Step 4: Solve output layer (Logistic + L2) ---
        # Append the bias column here, right before fitting the logistic output.
        Phi_surv      = Phi_pool[:, best_idx]                   # (m, mu)
        Phi_surv_bias = np.c_[Phi_surv, np.ones(len(y_tr))]    # (m, mu+1)

        best_lam, best_beta, best_log_loss, best_objective = compute_logistic_l2(
            Phi_surv_bias, y_tr,
            lambda_override=(overfit_lambda if overfit else None),
        )
        state.beta = best_beta   # (mu+1, 1)
        stagnation_metric = -float(np.max(pool_r2[best_idx])) if overfit else best_objective
        objective_history.append(stagnation_metric)

        y_proba_tr = predict_proba(Phi_surv_bias, state.beta)
        y_pred_tr = predict_pm1(Phi_surv_bias, state.beta)
        train_acc = float(accuracy_score(
            y_tr.ravel(),
            y_pred_tr
        ))
        mean_sigma = float(state.L[:, np.arange(state.D), np.arange(state.D)].mean())

        # --- Stagnation detection and sigma reset ---
        reset_count = 0
        if not overfit:
            if len(objective_history) >= STAGNATION_WINDOW:
                recent_improvement = objective_history[-STAGNATION_WINDOW] - objective_history[-1]
                relative_improvement = recent_improvement / (abs(objective_history[-STAGNATION_WINDOW]) + 1e-12)
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
        history.append(
            {
                "epoch": epoch,
                "strategy": effective_strategy,
                "log_loss": float(best_log_loss),
                "objective": float(best_objective),
                "lambda": float(best_lam),
                "train_acc": train_acc,
                "train_mean_positive_proba": float(np.mean(y_proba_tr)),
                "sigma_mean": mean_sigma,
                "select_ms": float(select_ms),
                "epoch_ms": float(epoch_ms),
                "reset_count": reset_count,
            }
        )
        if epoch % verbose == 0 or epoch == 1:
            print(
                f"Gen {epoch:4d}/{epochs}  |  "
                f"LogLoss={best_log_loss:.5f}  |  "
                f"Obj={best_objective:.5f}  |  "
                f"lambda={best_lam:.1e}  |  "
                f"Train Acc={train_acc:.4f}  |  "
                f"sigma_mean={mean_sigma:.5f}  |  "
                f"select_ms={select_ms:.1f}  |  "
                f"epoch_ms={epoch_ms:.1f}"
            )

    # --- Final test evaluation ---
    Phi_test      = compute_activations(state.alpha, X_te)       # (m_te, mu)
    Phi_test_bias = np.c_[Phi_test, np.ones(len(y_te))]         # (m_te, mu+1)
    test_proba    = predict_proba(Phi_test_bias, state.beta)
    test_pred     = predict_pm1(Phi_test_bias, state.beta)
    test_acc      = float(accuracy_score(
        y_te.ravel(),
        test_pred
    ))
    val_acc = None
    if X_val is not None and y_val is not None:
        Phi_val = compute_activations(state.alpha, X_val)
        Phi_val_bias = np.c_[Phi_val, np.ones(len(y_val))]
        val_proba = predict_proba(Phi_val_bias, state.beta)
        val_pred = predict_pm1(Phi_val_bias, state.beta)
        val_acc = float(accuracy_score(
            y_val.ravel(),
            val_pred
        ))
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
        "final_train_acc": history[-1]["train_acc"],
        "validation_acc": val_acc,
        "test_acc": test_acc,
        "final_log_loss": history[-1]["log_loss"],
        "final_objective": history[-1]["objective"],
        "final_lambda": history[-1]["lambda"],
        "test_mean_positive_proba": float(np.mean(test_proba)),
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
    if val_acc is not None:
        print(f"  Val Accuracy : {val_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
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
    parser.add_argument("--overfit-lambda", type=float, default=1e-8)
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
        save_dir=args.save_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
