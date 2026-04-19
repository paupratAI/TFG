"""
selection.py — Neuron selection strategies for the Michigan-style ES.

The central challenge in a Michigan-style ES is credit assignment:
because all neurons work together to produce a single output, it is
hard to tell which individual neurons are "good."

Three strategies are implemented here:

  omp_residual_selection  (default, recommended)
      Selects neurons one at a time to explain the variance in y that
      the previously selected neurons could NOT explain.  Solves the
      cooperative credit assignment problem by evaluating each neuron
      against the *residual*, not against y directly.

  r2_fitness + diversity_penalised_selection  (ablation baseline)
      Scores each neuron independently by its squared correlation with y,
      then penalises redundant neurons.  Simpler but blind to synergies.

  cfs_selection (correlation-based subset merit)
      Uses Hall's CFS heuristic adapted to continuous targets: maximise
      mean neuron-target relevance while penalising mean neuron-neuron
      redundancy within the selected subset.

Performance note
----------------
omp_residual_selection pre-normalises the columns of Phi once before
the selection loop.  This means each iteration computes dots = Phi_n.T @ r
and gets the cosine scores directly without a per-iteration division,
saving one (P,) vector operation per OMP step.
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
    1. Score each candidate:   score(i) = (phi_n_i . r)^2
       where phi_n_i is the unit-norm version of phi_i (pre-computed once).
       This is the squared cosine similarity between phi_i and the residual.

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
    Phi64 = np.asfortranarray(Phi, dtype=np.float64)
    m, P  = Phi64.shape

    if mu > P:
        raise ValueError(f"Cannot select {mu} neurons from a pool of {P}.")

    # Initialise the residual as mean-centred y.
    # The output bias beta_0 will absorb the mean of y, so we remove it here
    # to prevent the first neuron from trivially correlating with a constant.
    r = y_v - y_v.mean()                         # (m,)

    # --- Pre-normalise columns ONCE before the loop ---
    # phi_norms(i) = ||phi_i||.  Degenerate columns get norm=1 to avoid
    # division by zero; they will score 0 because their dot with r is tiny.
    phi_norms = np.linalg.norm(Phi64, axis=0)                    # (P,)
    safe_norms = np.where(phi_norms > 1e-10, phi_norms, 1.0)    # (P,)
    Phi_n = Phi64 / safe_norms[np.newaxis, :]                    # (m, P)  unit-norm columns
    degenerate = phi_norms <= 1e-10                              # (P,) bool mask

    selected : list[int] = []
    Q_mat    = np.empty((m, mu), dtype=np.float64, order='F')  # growing orthonormal basis
    k        = 0                                     # valid basis columns in Q_mat
    available = np.ones(P, dtype=bool)               # False once a neuron is selected

    for _ in range(mu):
        # --- Step 1: Score all remaining candidates against the residual ---
        # Because columns of Phi_n are unit-norm, dots are cosine similarities.
        # scores(i) = (phi_n_i . r)^2  — no per-iteration division needed.
        dots   = Phi_n.T @ r                                     # (P,)
        scores = dots ** 2                                       # (P,)
        scores[degenerate]  = 0.0
        scores[~available] = -np.inf

        # --- Step 2: Pick best candidate ---
        i_star = int(np.argmax(scores))

        # --- Step 3: Modified Gram-Schmidt orthogonalisation ---
        # We work with the ORIGINAL (unnormalised) column of phi_i* here,
        # because the orthonormal basis Q must be built from true directions.
        q = Phi64[:, i_star].copy()
        if k > 0:
            Q_k = Q_mat[:, :k]                         # (m, k) view — no copy
            q = q - Q_k @ (Q_k.T @ q)                 # first pass
            q = q - Q_k @ (Q_k.T @ q)                 # second pass (re-orthogonalise)

        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            # phi_i* is already spanned by the selected set (near-duplicate).
            # Skip it and try the next best candidate.
            available[i_star] = False
            continue

        q = q / q_norm                      # normalise to unit length
        Q_mat[:, k] = q
        k += 1

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


def _omp_residual_selection_legacy(
    Phi: np.ndarray,
    y  : np.ndarray,
    mu : int,
) -> np.ndarray:
    """Reference implementation used only for equivalence testing."""
    y_v   = y.ravel().astype(np.float64)
    Phi64 = Phi.astype(np.float64)
    m, P  = Phi64.shape

    if mu > P:
        raise ValueError(f"Cannot select {mu} neurons from a pool of {P}.")

    r = y_v - y_v.mean()
    phi_norms = np.linalg.norm(Phi64, axis=0)

    selected: list[int] = []
    Q: list[np.ndarray] = []
    available = np.ones(P, dtype=bool)

    for _ in range(mu):
        dots   = Phi64.T @ r
        safe_n = np.where(phi_norms > 1e-10, phi_norms, 1.0)
        scores = (dots ** 2) / (safe_n ** 2)
        scores = np.where(phi_norms > 1e-10, scores, 0.0)
        scores[~available] = -np.inf
        i_star = int(np.argmax(scores))

        q = Phi64[:, i_star].copy()
        if Q:
            Q_mat = np.column_stack(Q)
            q = q - Q_mat @ (Q_mat.T @ q)
            q = q - Q_mat @ (Q_mat.T @ q)

        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            available[i_star] = False
            continue

        q = q / q_norm
        Q.append(q)
        r = r - np.dot(q, r) * q
        selected.append(i_star)
        available[i_star] = False

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
    phi_c  = Phi64 - Phi64.mean(axis=0, keepdims=True)
    y_c    = y_v   - y_v.mean()
    
    # PRODUCTE ESCALAR: No col·lapsa la memòria i usa el maquinari BLAS
    cov    = (phi_c.T @ y_c) / len(y_c)
    
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


def cfs_selection(
    Phi: np.ndarray,
    y: np.ndarray,
    mu: int,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Select mu neurons using greedy forward search that maximises an adapted
    CFS merit for continuous targets:

        Merit(S) = k * mean_rcf / sqrt(k + k*(k-1)*mean_rff)

    where mean_rcf is mean squared neuron-target correlation and mean_rff is
    mean squared neuron-neuron correlation within S.
    """
    y_v = y.ravel().astype(np.float64)
    Phi64 = Phi.astype(np.float64)
    _m, P = Phi64.shape

    if mu > P:
        raise ValueError(f"Cannot select {mu} neurons from a pool of {P}.")

    phi_c = Phi64 - Phi64.mean(axis=0, keepdims=True)
    y_c = y_v - y_v.mean()

    col_norms = np.linalg.norm(phi_c, axis=0)
    safe_norms = np.where(col_norms > epsilon, col_norms, 1.0)
    degenerate = col_norms <= epsilon

    y_norm = np.linalg.norm(y_c)
    safe_y_norm = y_norm if y_norm > epsilon else 1.0
    phi_n = phi_c / safe_norms[np.newaxis, :]

    # Relevance term: squared neuron-target correlations.
    r_cf = (phi_n.T @ y_c / safe_y_norm) ** 2
    r_cf[degenerate] = 0.0

    # Redundancy term matrix: squared neuron-neuron correlations.
    corr_matrix = phi_n.T @ phi_n
    C = corr_matrix ** 2
    C[degenerate, :] = 0.0
    C[:, degenerate] = 0.0

    selected: list[int] = []
    available = np.ones(P, dtype=bool)
    available[degenerate] = False

    sum_rcf = 0.0
    sum_rff = 0.0

    for step in range(mu):
        k = step
        new_k = k + 1
        new_rcf = sum_rcf + r_cf

        if not selected:
            new_rff_vec = np.zeros(P, dtype=np.float64)
        else:
            new_rff_vec = sum_rff + C[:, selected].sum(axis=1)

        mean_rcf_vec = new_rcf / new_k
        if new_k > 1:
            n_pairs = new_k * (new_k - 1) / 2.0
            mean_rff_vec = new_rff_vec / n_pairs
        else:
            mean_rff_vec = np.zeros(P, dtype=np.float64)

        denom_sq_vec = new_k + new_k * (new_k - 1) * mean_rff_vec
        denom_sq_vec = np.maximum(denom_sq_vec, epsilon)

        merit_vec = new_k * mean_rcf_vec / np.sqrt(denom_sq_vec)
        merit_vec[~available] = -np.inf
        merit_vec[degenerate] = -np.inf

        i_star = int(np.argmax(merit_vec))

        if selected:
            sum_rff += float(C[i_star, selected].sum())
        sum_rcf += float(r_cf[i_star])

        selected.append(i_star)
        available[i_star] = False

    return np.array(selected, dtype=np.int64)


if __name__ == "__main__":
    # Deterministic check: vectorised MGS must match legacy selection order.
    rng = np.random.default_rng(42)
    m, P, mu = 200, 90, 25
    Phi_test = rng.standard_normal((m, P))
    y_test   = rng.choice([-1.0, 1.0], size=m)
    idx_legacy = _omp_residual_selection_legacy(Phi_test, y_test, mu)
    idx_vec    = omp_residual_selection(Phi_test, y_test, mu)
    if not np.array_equal(idx_legacy, idx_vec):
        raise AssertionError(
            "Vectorised omp_residual_selection changed selected indices."
        )
    print("OMP vectorisation test passed: selected indices are identical.")