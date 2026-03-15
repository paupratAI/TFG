"""
core_es.py — ES population state, mutation, and neuron activations.

This module defines what a "population" is in our Michigan-style ES
and how it evolves from one generation to the next.

Michigan-style framing
----------------------
Each individual in the population IS one hidden neuron of the MLP.
The full hidden layer is therefore represented by the whole population.
Each neuron carries:
  - a genotype alpha = [weights | bias]  (the neuron's input connections)
  - a strategy matrix L  (controls how it mutates)
"""

import numpy as np


class MichiganESState:
    """
    Holds the mutable state of the Michigan-style ES population.

    Attributes
    ----------
    alpha : (mu, D)     Each row is one neuron's genotype [weights | bias].
                        D = n_features + 1.
    L     : (mu, D, D)  Per-neuron lower-triangular Cholesky factor.
                        The actual mutation covariance of neuron i is L_i @ L_i.T.
    beta  : (mu+1, 1)   Analytic output weights, set after each Ridge solve.
                        None until the first generation completes.
    mu    : int         Population size = hidden layer width.
    D     : int         Genotype length = n_features + 1.
    """

    def __init__(self, mu: int, input_size: int, sigma_init: float = 0.3) -> None:
        """
        Initialise a random population of mu neurons.

        Parameters
        ----------
        mu         : Number of neurons (= hidden layer width).
        input_size : Number of input features (bias is added internally).
        sigma_init : Initial step size on all L diagonal entries.
        """
        D          = input_size + 1
        bound      = 1.0 / np.sqrt(input_size)   # keeps initial activations in tanh's linear region
        self.alpha = np.random.uniform(-bound, bound, (mu, D)).astype(np.float64)
        self.L     = np.stack(
            [np.eye(D, dtype=np.float64) * sigma_init for _ in range(mu)]
        )
        self.beta: np.ndarray | None = None
        self.mu    = mu
        self.D     = D


def mutate_population(
    state      : MichiganESState,
    n_offspring: int   = 350,
    beta_cov   : float = 0.05,
    sigma_min  : float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n_offspring new neurons by mutating randomly chosen parents.

    Each offspring inherits and mutates both its genotype (alpha) and its
    strategy parameters (L).  This is called self-adaptive mutation: the
    step sizes evolve alongside the solution.

    How the strategy matrix L is mutated
    -------------------------------------
    Diagonal entries (step sizes) are scaled by a log-normal factor so they
    stay positive.  A hard lower bound sigma_min prevents them from shrinking
    to zero, which would halt all exploration:

        L_ii <- max( L_ii * exp(tau_prime * xi_global + tau * xi_local),
                     sigma_min )

    Off-diagonal entries (rotation/correlation) are shifted by small additive
    Gaussian noise.

    How the genotype alpha is mutated
    ----------------------------------
        alpha <- alpha + L' @ z,    z ~ N(0, I)

    Because L is lower-triangular, L @ z draws from a correlated Gaussian
    distribution N(0, L L^T), allowing the search to follow elongated fitness
    landscapes.

    Parameters
    ----------
    state       : Current ES population.
    n_offspring : Number of offspring to produce (lambda in (mu+lambda)-ES).
    beta_cov    : Noise scale for off-diagonal L entries.
    sigma_min   : Hard lower bound on L diagonal entries (prevents collapse).

    Returns
    -------
    off_alpha : (n_offspring, D)     offspring genotypes.
    off_L     : (n_offspring, D, D)  offspring strategy matrices.
    """
    mu, D = state.mu, state.D

    # Learning rates from Beyer & Schwefel (2002).
    tau   = 1.0 / np.sqrt(2.0 * D)          # global learning rate
    tau_p = 1.0 / np.sqrt(2.0 * np.sqrt(D)) # local  learning rate

    # Each offspring copies a randomly chosen parent (with replacement).
    parent_idx = np.random.randint(0, mu, n_offspring)
    alpha_p    = state.alpha[parent_idx].copy()   # (n_off, D)
    L_p        = state.L[parent_idx].copy()       # (n_off, D, D)

    # --- Mutate diagonal (step sizes) ---
    xi_global = np.random.randn(n_offspring)        # one shared draw per offspring
    xi_local  = np.random.randn(n_offspring, D)     # one draw per weight dimension
    diag_idx  = np.arange(D)
    diag_vals = L_p[:, diag_idx, diag_idx]
    new_diag  = diag_vals * np.exp(tau_p * xi_global[:, None] + tau * xi_local)
    L_p[:, diag_idx, diag_idx] = np.maximum(new_diag, sigma_min)

    # --- Mutate off-diagonal (rotation) ---
    i_idx, j_idx = np.tril_indices(D, k=-1)
    L_p[:, i_idx, j_idx] += beta_cov * np.random.randn(n_offspring, len(i_idx))

    # --- Mutate genotype ---
    z           = np.random.randn(n_offspring, D, 1)
    delta_alpha = np.matmul(L_p, z).squeeze(-1)    # (n_off, D)
    off_alpha   = (alpha_p + delta_alpha).astype(np.float64)

    return off_alpha, L_p


def compute_activations(alpha: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute the hidden layer activation matrix Phi = tanh(X @ W.T + b).

    Each column of the output is one neuron's response across all samples.

    Parameters
    ----------
    alpha : (P, D)  genotype matrix; last column holds biases.
    X     : (m, n)  input feature matrix.

    Returns
    -------
    Phi : (m, P)  float64  hidden activations.
    """
    W   = alpha[:, :-1]     # (P, n)  input weights
    b   = alpha[:, -1]      # (P,)    biases
    return np.tanh(X @ W.T + b).astype(np.float64)