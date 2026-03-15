"""
selection.py — Neuron selection strategies for the Michigan-style ES.

The central challenge in a Michigan-style ES is credit assignment:
because all neurons work together to produce a single output, it is
hard to tell which individual neurons are "good."

Two strategies are implemented here:

  omp_residual_selection  (default, recommended)
      Selects neurons one at a time to explain the variance in y that
      the previously selected neurons could NOT explain.  Solves the
      cooperative credit assignment problem by evaluating each neuron
      against the *residual*, not against y directly.

  r2_fitness + diversity_penalised_selection  (ablation baseline)
      Scores each neuron independently by its squared correlation with y,
      then penalises redundant neurons.  Simpler but blind to synergies.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Primary strategy: OMP-style sequential residual selection
# ─────────────────────────────────────────────────────────────────────────────

def omp_residual_selection(
    Phi: np.ndarray,
    y  : np.ndarray,
    mu : int,
) -> np.ndarray:
    """
    Select mu neurons from the candidate pool using an OMP-inspired greedy
    algorithm that updates a residual after each selection.

    Why this is better than scoring neurons independently
    ------------------------------------------------------
    If we score every neuron against the original target y, we miss two
    important cases:

    1. A neuron that is useless alone can be essential in combination
       with others (the "XOR / synergy" problem).
    2. Two neurons that explain the same feature both get high scores,
       wasting a selection slot on a redundant neuron.

    The residual-update trick fixes both: after selecting neuron i*, we
    subtract the part of y that i* already explains, producing a residual
    r that contains only the unexplained variance.  Every subsequent
    selection targets what is STILL missing.

    Algorithm (one iteration)
    -------------------------
    1. Score each candidate:   score(i) = (phi_i . r)^2 / ||phi_i||^2
       This is the fraction of the current residual that neuron i would explain.

    2. Pick the best:          i* = argmax score(i)

    3. Orthogonalise phi_i* against all previously selected directions using
       Modified Gram-Schmidt (MGS).  This builds an orthonormal basis Q.

    4. Update the residual:    r <- r - (q . r) * q
       After this step, r is perpendicular to the direction i* just captured.

    Cost per generation: O(mu * m * P)  --  no Ridge solve inside the loop.

    Parameters
    ----------
    Phi : (m, P)       activation matrix for all P candidate neurons.
                       Must NOT include a bias column (added later before Ridge).
    y   : (m,) or (m,1) target labels.
    mu  : number of neurons to select.

    Returns
    -------
    selected_idx : (mu,)  integer column indices into Phi, in selection order.
    """
    y_v   = y.ravel().astype(np.float64)
    Phi64 = Phi.astype(np.float64)
    m, P  = Phi64.shape

    if mu > P:
        raise ValueError(f"Cannot select {mu} neurons from a pool of {P}.")

    # Initialise the residual as mean-centred y.
    # The output bias beta_0 will absorb the mean of y, so we remove it here
    # to prevent the first neuron from trivially correlating with a constant.
    r = y_v - y_v.mean()                         # (m,)

    # Pre-compute column norms once for efficient scoring.
    phi_norms = np.linalg.norm(Phi64, axis=0)    # (P,)

    selected : list[int]        = []
    Q        : list[np.ndarray] = []              # growing orthonormal basis
    available = np.ones(P, dtype=bool)            # False once a neuron is selected

    for _ in range(mu):
        # --- Step 1: Score all remaining candidates against the residual ---
        dots   = Phi64.T @ r                                     # (P,)
        safe_n = np.where(phi_norms > 1e-10, phi_norms, 1.0)    # avoid division by zero
        scores = (dots ** 2) / (safe_n ** 2)                     # (P,)
        scores = np.where(phi_norms > 1e-10, scores, 0.0)        # zero-out degenerate neurons
        scores[~available] = -np.inf

        # --- Step 2: Pick best candidate ---
        i_star = int(np.argmax(scores))

        # --- Step 3: Modified Gram-Schmidt orthogonalisation ---
        # We need a vector q that points in the direction of phi_i*
        # but is orthogonal to all previously selected directions.
        q = Phi64[:, i_star].copy()
        for q_j in Q:
            q = q - np.dot(q_j, q) * q_j   # remove component along each existing basis vector

        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            # phi_i* is already spanned by the selected set (near-duplicate).
            # Skip it and try the next best candidate.
            available[i_star] = False
            continue

        q = q / q_norm                      # normalise to unit length
        Q.append(q)

        # --- Step 4: Update residual ---
        r = r - np.dot(q, r) * q           # project out the captured direction

        selected.append(i_star)
        available[i_star] = False

    # Safety fallback: if MGS skips caused fewer than mu selections,
    # fill remaining slots with the highest marginal-R2 candidates.
    if len(selected) < mu:
        fallback = r2_fitness(Phi64, y_v)
        fallback[np.array(selected, dtype=np.int64)] = -np.inf
        while len(selected) < mu:
            fb_idx = int(np.argmax(fallback))
            selected.append(fb_idx)
            fallback[fb_idx] = -np.inf

    return np.array(selected[:mu], dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation baseline: marginal R² fitness + diversity-penalised selection
# ─────────────────────────────────────────────────────────────────────────────

def r2_fitness(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Score each neuron independently as the squared Pearson correlation
    between its activation vector and the target y.

    score(i) = Cor(phi_i, y)^2

    This equals the R^2 of the simple linear regression y ~ phi_i.
    A score of 1.0 means the neuron perfectly predicts y on its own;
    0.0 means it carries no linear information about y.

    Limitation: neurons are evaluated in isolation, so synergistic
    interactions between neurons are completely ignored.

    Parameters
    ----------
    Phi : (m, P)  activation matrix.
    y   : (m,) or (m, 1)  target vector.

    Returns
    -------
    scores : (P,)  float64, values in [0, 1].
    """
    y_v    = y.ravel().astype(np.float64)
    Phi64  = Phi.astype(np.float64)
    phi_c  = Phi64 - Phi64.mean(axis=0, keepdims=True)   # centre columns
    y_c    = y_v   - y_v.mean()                           # centre target
    cov    = (phi_c * y_c[:, None]).mean(axis=0)
    std_ph = phi_c.std(axis=0)
    std_y  = y_c.std()
    denom  = np.maximum(std_ph * std_y, 1e-12)
    return ((cov / denom) ** 2).astype(np.float64)


def diversity_penalised_selection(
    scores: np.ndarray,
    Phi   : np.ndarray,
    mu    : int,
    gamma : float = 0.3,
) -> np.ndarray:
    """
    Greedy selection that balances individual R^2 scores with diversity.

    At each step, a neuron loses points for being too similar (correlated)
    with neurons already selected:

        effective_score(i) = score(i) - gamma * max_j |Cor(phi_i, phi_j)|

    This prevents all mu survivors from collapsing onto a single dominant
    feature direction.  gamma=0 gives pure R^2 ranking; gamma=1 forces
    maximum diversity.

    Parameters
    ----------
    scores : (P,)   marginal R^2 per neuron.
    Phi    : (m, P) activation matrix.
    mu     : number of neurons to select.
    gamma  : diversity penalty weight (default 0.3).

    Returns
    -------
    selected_idx : (mu,)  integer indices into the pool.
    """
    P     = len(scores)
    Phi64 = Phi.astype(np.float64)

    # Normalise columns so that a dot product equals the Pearson correlation.
    phi_c = Phi64 - Phi64.mean(axis=0, keepdims=True)
    norms = np.maximum(np.linalg.norm(phi_c, axis=0, keepdims=True), 1e-12)
    phi_n = phi_c / norms                             # (m, P) unit-norm columns

    selected  = []
    remaining = np.ones(P, dtype=bool)
    max_sim   = np.zeros(P, dtype=np.float64)         # tracks max |corr| with selected set

    for _ in range(mu):
        eff             = scores.astype(np.float64) - gamma * max_sim
        eff[~remaining] = -np.inf
        i_star          = int(np.argmax(eff))
        selected.append(i_star)
        remaining[i_star] = False
        # Update similarity scores now that i_star is in the selected set.
        new_sims = np.abs(phi_n.T @ phi_n[:, i_star])
        max_sim  = np.maximum(max_sim, new_sims)

    return np.array(selected, dtype=np.int64)