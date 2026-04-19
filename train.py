"""
train.py — Main training loop for the Michigan-style Evolution Strategy MLP.

This script orchestrates the full training procedure:
  1. Load and preprocess data.
  2. Initialise the ES population (one neuron per individual).
  3. Each generation: mutate -> activate -> select -> solve output layer.
  4. Report training accuracy, GCV score, and sigma at regular intervals.
  5. Report final test accuracy.

Usage
-----
    python train.py

To run the ablation (OMP vs R2), set fitness_strategy="r2" in the train()
call at the bottom, or import train() from another script.

Performance notes
-----------------
- r2_fitness is called ONCE per generation over the full pool (pool_r2).
  The stagnation-reset block reuses pool_r2[best_idx] instead of making
  a second r2_fitness call over Phi_surv.
- adaptive_lambda_grid has been removed: compute_ridge_gcv now builds the
  adaptive grid internally from the singular values it already computes,
  so no second SVD is ever issued per generation.
"""

import numpy as np
import time
from sklearn.metrics import accuracy_score

from data      import load_dataset
from core_es   import MichiganESState, mutate_population, compute_activations
from selection import (
    omp_residual_selection,
    r2_fitness,
    diversity_penalised_selection,
    cfs_selection,
)
from gcv       import compute_ridge_gcv


def train(
    fitness_strategy: str   = "omp",
    epochs          : int   = 100,
    mu              : int   = 50,
    n_offspring     : int   = 350,
    sigma_init      : float = 0.3,
    gamma           : float = 0.3,
    data_id         : int   = 150,
    verbose         : int   = 10,
    seed            : int   = 42,
    overfit          : bool  = False,
    overfit_n_samples: int   = 60,
    overfit_mu       : int   = 250,
    overfit_lambda   : float = 1e-8,
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

        Solve    ->  fit the output layer analytically using Ridge + GCV on
                     the mu survivors.  No gradient descent is needed here.

    Note on the bias column
    -----------------------
    A column of ones is appended to the activation matrix ONLY in the Solve
    step, immediately before the Ridge call.  All prior steps (fitness scoring,
    selection) work on raw activations without the bias column.

    Parameters
    ----------
    fitness_strategy : 'omp' (default), 'r2' (ablation), or 'cfs'.
    epochs           : Number of ES generations to run.
    mu               : Population size = hidden layer width.
    n_offspring      : Offspring produced per generation (lambda).
    sigma_init       : Initial mutation step size (L diagonal value).
    gamma            : Diversity penalty weight, used only for 'r2' strategy.
    data_id          : OpenML dataset identifier.
    verbose          : Print a log line every this many generations.
    seed             : Global random seed.
    overfit          : If True, enable memorisation demo (subset train, tiny ridge).
    overfit_n_samples: Training rows when overfit=True (capped by available).
    overfit_mu       : Population size when overfit=True (replaces mu).
    overfit_lambda   : Fixed Ridge lambda when overfit=True (bypasses GCV).

    Returns
    -------
    state : MichiganESState with the final evolved population and output weights.
    """
    np.random.seed(seed)

    # --- Load data ---
    X_tr, X_te, y_tr, y_te = load_dataset(data_id=data_id, seed=seed)

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
    gcv_history: list[float] = []
    select_ms_history: list[float] = []
    epoch_ms_history: list[float] = []
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
        # No bias column here -- it is added only before the Ridge solve.
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

        else:
            raise ValueError(
                f"Unknown fitness_strategy={fitness_strategy!r}. "
                "Choose 'omp', 'r2', or 'cfs'."
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

        # --- Step 4: Solve output layer (Ridge + GCV) ---
        # Append the bias column here, right before the Ridge solve.
        # compute_ridge_gcv builds the adaptive lambda grid internally from
        # the singular values it computes — no separate adaptive_lambda_grid call.
        Phi_surv      = Phi_pool[:, best_idx]                   # (m, mu)
        Phi_surv_bias = np.c_[Phi_surv, np.ones(len(y_tr))]    # (m, mu+1)

        best_lam, best_beta, best_gcv = compute_ridge_gcv(
            Phi_surv_bias, y_tr,
            lambda_override=(overfit_lambda if overfit else None),
        )
        state.beta = best_beta   # (mu+1, 1)
        stagnation_metric = -float(np.max(pool_r2[best_idx])) if overfit else best_gcv
        gcv_history.append(stagnation_metric)

        # --- Stagnation detection and sigma reset ---
        if not overfit:
            if len(gcv_history) >= STAGNATION_WINDOW:
                recent_improvement = gcv_history[-STAGNATION_WINDOW] - gcv_history[-1]
                relative_improvement = recent_improvement / (abs(gcv_history[-STAGNATION_WINDOW]) + 1e-12)
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
                    if epoch % verbose == 0:
                        print(
                            f"  [stagnation reset: {n_reset} neurons re-initialised "
                            f"| relative_improvement={relative_improvement:.6f}]"
                        )

        # --- Logging ---
        if epoch % verbose == 0 or epoch == 1:
            epoch_ms = (time.perf_counter() - epoch_t0) * 1e3
            epoch_ms_history.append(epoch_ms)
            y_hat_tr  = Phi_surv_bias @ state.beta
            train_acc = accuracy_score(
                y_tr.ravel(),
                np.where(y_hat_tr.ravel() > 0, 1.0, -1.0)
            )
            mean_sigma = state.L[:, np.arange(state.D), np.arange(state.D)].mean()
            print(
                f"Gen {epoch:4d}/{epochs}  |  "
                f"GCV={best_gcv:.5f}  |  "
                f"lambda={best_lam:.1e}  |  "
                f"Train Acc={train_acc:.4f}  |  "
                f"sigma_mean={mean_sigma:.5f}  |  "
                f"select_ms={select_ms:.1f}  |  "
                f"epoch_ms={epoch_ms:.1f}"
            )
        else:
            epoch_ms_history.append((time.perf_counter() - epoch_t0) * 1e3)

    # --- Final test evaluation ---
    Phi_test      = compute_activations(state.alpha, X_te)       # (m_te, mu)
    Phi_test_bias = np.c_[Phi_test, np.ones(len(y_te))]         # (m_te, mu+1)
    test_hat      = Phi_test_bias @ state.beta
    test_acc      = accuracy_score(
        y_te.ravel(),
        np.where(test_hat.ravel() > 0, 1.0, -1.0)
    )
    print(f"\n{'=' * 65}")
    print(f"  Strategy     : {fitness_strategy!r}")
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

    return state


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # R2 ablation run: compare against the marginal-R² baseline.
    train(
        fitness_strategy = "r2",
         epochs           = 100,
         mu               = 50,
         n_offspring      = 350,
         sigma_init       = 0.3,
         gamma            = 0.3,
         data_id          = 150,
         verbose          = 10,
    )

    # OMP primary run: sequential residual fitting (recommended).
    train(
        fitness_strategy = "omp",
        epochs           = 100,
        mu               = 50,
        n_offspring      = 350,
        sigma_init       = 0.3,
        verbose          = 10,
    )

    # CFS ablation run: explicit relevance/redundancy subset merit.
    train(
        fitness_strategy = "cfs",
        epochs           = 100,
        mu               = 50,
        n_offspring      = 350,
        sigma_init       = 0.3,
        verbose          = 10,
    )

    # ── Overfit benchmark (demonstrates memorisation capacity) ──────────
    # print("\n" + "=" * 65)
    # print("  OVERFIT BENCHMARK — forcing memorisation")
    # print("=" * 65)
    # train(
    #     fitness_strategy  = "r2",   # overridden internally, kept for clarity
    #     epochs            = 150,
    #     mu                = 50,     # overridden internally by overfit_mu
    #     n_offspring       = 350,
    #     sigma_init        = 0.3,
    #     verbose           = 10,
    #     seed              = 42,
    #     overfit           = True,
    #     overfit_n_samples = 25,
    #     overfit_mu        = 250,
    #     overfit_lambda    = 1e-8,
    # )