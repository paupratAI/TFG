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
"""

import numpy as np
from sklearn.metrics import accuracy_score

from data      import load_dataset
from core_es   import MichiganESState, mutate_population, compute_activations
from selection import omp_residual_selection, r2_fitness, diversity_penalised_selection
from gcv       import compute_ridge_gcv


# Lambda grid shared by all Ridge solves.
# 17 log-spaced values on [1e-6, 1e6] -- wide enough to avoid boundary effects.
LAMBDA_GRID: list[float] = list(np.logspace(-6, 6, 17))


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

    for epoch in range(1, epochs + 1):

        # --- Step 1: Mutate ---
        off_alpha, off_L = mutate_population(state, n_offspring=n_offspring)
        pool_alpha = np.vstack([state.alpha, off_alpha])   # (P, D)
        pool_L     = np.vstack([state.L,     off_L    ])   # (P, D, D)

        # --- Step 2: Activate ---
        # Phi_pool columns are neurons, rows are training samples.
        # No bias column here -- it is added only before the Ridge solve.
        Phi_pool = compute_activations(pool_alpha, X_tr)   # (m, P)

        # --- Step 3: Select ---
        if fitness_strategy == "omp":
            best_idx = omp_residual_selection(Phi_pool, y_tr, mu=mu)

        elif fitness_strategy == "r2":
            scores   = r2_fitness(Phi_pool, y_tr)
            best_idx = diversity_penalised_selection(
                scores, Phi_pool, mu=mu, gamma=gamma
            )

        else:
            raise ValueError(
                f"Unknown fitness_strategy={fitness_strategy!r}. "
                "Choose 'omp' or 'r2'."
            )

        # Survivors become the new population.
        state.alpha = pool_alpha[best_idx].copy()   # (mu, D)
        state.L     = pool_L[best_idx].copy()       # (mu, D, D)

        # --- Step 4: Solve output layer (Ridge + GCV) ---
        # Append the bias column here, right before the Ridge solve.
        Phi_surv      = Phi_pool[:, best_idx]                   # (m, mu)
        Phi_surv_bias = np.c_[Phi_surv, np.ones(len(y_tr))]    # (m, mu+1)

        best_lam, best_beta, best_gcv = compute_ridge_gcv(
            Phi_surv_bias, y_tr, lambda_candidates=LAMBDA_GRID
        )
        state.beta = best_beta   # (mu+1, 1)

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