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
from sklearn.metrics import accuracy_score

from data      import load_dataset
from core_es   import MichiganESState, mutate_population, compute_activations
from selection import omp_residual_selection, r2_fitness, diversity_penalised_selection
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

        Solve    ->  fit the output layer analytically using Ridge + GCV on
                     the mu survivors.  No gradient descent is needed here.

    Note on the bias column
    -----------------------
    A column of ones is appended to the activation matrix ONLY in the Solve
    step, immediately before the Ridge call.  All prior steps (fitness scoring,
    selection) work on raw activations without the bias column.

    Parameters
    ----------
    fitness_strategy : 'omp' (default) or 'r2' (ablation).
    epochs           : Number of ES generations to run.
    mu               : Population size = hidden layer width.
    n_offspring      : Offspring produced per generation (lambda).
    sigma_init       : Initial mutation step size (L diagonal value).
    gamma            : Diversity penalty weight, used only for 'r2' strategy.
    data_id          : OpenML dataset identifier.
    verbose          : Print a log line every this many generations.
    seed             : Global random seed.

    Returns
    -------
    state : MichiganESState with the final evolved population and output weights.
    """
    np.random.seed(seed)

    # --- Load data ---
    X_tr, X_te, y_tr, y_te = load_dataset(data_id=data_id, seed=seed)

    # --- Initialise population ---
    state = MichiganESState(
        mu=mu, input_size=X_tr.shape[1], sigma_init=sigma_init
    )

    print(
        f"\nMichigan ES | strategy={fitness_strategy!r} | "
        f"mu={mu} | lambda={n_offspring} | sigma_0={sigma_init}\n"
        + "-" * 65
    )
    gcv_history: list[float] = []
    STAGNATION_WINDOW    = 15
    STAGNATION_THRESHOLD = 0.005
    SIGMA_RESET_FRACTION = 0.25

    for epoch in range(1, epochs + 1):

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

        # --- Step 3: Select ---
        if fitness_strategy == "omp":
            best_idx = omp_residual_selection(Phi_pool, y_tr, mu=mu)

        elif fitness_strategy == "r2":
            best_idx = diversity_penalised_selection(
                pool_r2, Phi_pool, mu=mu, gamma=gamma
            )

        else:
            raise ValueError(
                f"Unknown fitness_strategy={fitness_strategy!r}. "
                "Choose 'omp' or 'r2'."
            )

        # --- Weak elitism: guarantee top-k parents survive selection ---
        # Parents occupy pool indices [0 .. mu-1]. We identify the best ELITE_K
        # among them by R² and force them into best_idx, displacing the weakest
        # non-elite survivors. This preserves well-adapted L matrices.
        ELITE_K = max(1, mu // 10)
        if epoch > 1 and state.beta is not None:
            # Reuse pool_r2[:mu] — parents are always the first mu entries in pool.
            elite_pool_idx = np.argsort(pool_r2[:mu])[-ELITE_K:]

            best_idx = best_idx.tolist()
            best_set = set(best_idx)
            elite_set = set(elite_pool_idx.tolist())
            # Reuse pool_r2 for survivor scores — no second r2_fitness call.
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
            Phi_surv_bias, y_tr
        )
        state.beta = best_beta   # (mu+1, 1)
        gcv_history.append(best_gcv)

        # --- Stagnation detection and sigma reset ---
        if len(gcv_history) >= STAGNATION_WINDOW:
            recent_improvement = gcv_history[-STAGNATION_WINDOW] - gcv_history[-1]
            relative_improvement = recent_improvement / (abs(gcv_history[-STAGNATION_WINDOW]) + 1e-12)
            if relative_improvement < STAGNATION_THRESHOLD:
                # Reuse pool_r2[best_idx] — same as r2 scores for survivors.
                # No second r2_fitness call needed.
                scores_surv = pool_r2[best_idx]
                n_reset = max(1, int(mu * SIGMA_RESET_FRACTION))
                worst_idx = np.argsort(scores_surv)[:n_reset]
                state.L[worst_idx] = np.stack(
                    [np.eye(state.D, dtype=np.float64) * sigma_init for _ in range(n_reset)]
                )
                if epoch % verbose == 0:
                    print(
                        f"  [stagnation reset: {n_reset} neurons re-initialised "
                        f"| relative_improvement={relative_improvement:.6f}]"
                    )

        # --- Logging ---
        if epoch % verbose == 0 or epoch == 1:
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
                f"sigma_mean={mean_sigma:.5f}"
            )

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
    print(f"{'=' * 65}\n")

    return state


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Primary run: OMP strategy (recommended).
    train(
        fitness_strategy = "omp",
        epochs           = 100,
        mu               = 50,
        n_offspring      = 350,
        sigma_init       = 0.3,
        data_id          = 150,
        verbose          = 10,
    )

    # Ablation run: uncomment to compare against the marginal-R² baseline.
    # train(
    #     fitness_strategy = "r2",
    #     epochs           = 100,
    #     mu               = 50,
    #     n_offspring      = 350,
    #     sigma_init       = 0.3,
    #     gamma            = 0.3,
    #     data_id          = 150,
    #     verbose          = 10,
    # )