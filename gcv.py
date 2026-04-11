"""
gcv.py — Analytic Ridge Regression with Generalised Cross-Validation (GCV).

Solves the output (hidden-to-output) layer of the MLP analytically.
Rather than training the output weights by gradient descent, we find
the mathematically optimal weights in one closed-form step.

What is Ridge Regression?
--------------------------
Given hidden activations Phi (m x H) and targets Y (m x 1), Ridge
regression finds output weights beta that minimise:

    ||Y - Phi @ beta||^2  +  lambda * ||beta||^2

The first term minimises prediction error; the second term (scaled by
lambda) prevents overfitting by penalising large weights.  The solution
is exact:

    beta(lambda) = (Phi.T @ Phi + lambda * I)^-1 @ Phi.T @ Y

What is GCV and why do we need it?
------------------------------------
Lambda controls how much regularisation to apply.  Too small -> overfit;
too large -> underfit.  GCV (Golub et al. 1979) efficiently estimates
which lambda will generalise best, without needing a separate validation
set.  It approximates Leave-One-Out Cross-Validation (LOOCV) in one pass.

Why SVD instead of direct matrix inversion?
--------------------------------------------
Direct inversion of (Phi.T @ Phi + lambda * I) is numerically unstable
when Phi has near-zero singular values.  The SVD decomposes Phi = U D V.T
and expresses the solution using element-wise division by (d^2 + lambda),
which is always numerically safe.  It also lets us evaluate all lambda
candidates cheaply after a single decomposition.

Performance note
----------------
The previous design called adaptive_lambda_grid (which ran a full SVD)
and then compute_ridge_gcv (which ran a second full SVD on the same
matrix).  This module fuses both into a single function so the SVD is
computed exactly once per generation.
"""

import numpy as np


def compute_ridge_gcv(
    Phi              : np.ndarray,
    Y                : np.ndarray,
    lambda_candidates: list[float] | None = None,
    n_lambda_points  : int = 30,
) -> tuple[float, np.ndarray, float]:
    """
    Find the best Ridge penalty lambda via GCV, then return the
    corresponding weights beta.

    The adaptive lambda grid is built internally from the singular values
    of Phi computed during the SVD, so no second SVD is ever needed.

    Parameters
    ----------
    Phi               : (m, H)  feature matrix (hidden activations + bias column).
    Y                 : (m,) or (m, 1)  target vector.
    lambda_candidates : Explicit list of positive regularisation values to try.
                        If None (default), an adaptive grid of n_lambda_points
                        values is built automatically from the singular values of
                        Phi.  Providing an explicit list overrides this behaviour.
    n_lambda_points   : Number of points in the auto-built adaptive grid.
                        Ignored when lambda_candidates is provided explicitly.

    Returns
    -------
    best_lambda : float        Lambda that minimised the GCV score.
    best_beta   : (H, 1)       Ridge weights at best_lambda.
    best_gcv    : float        The minimum GCV score achieved.

    Raises
    ------
    ValueError  If shapes are incompatible or no positive lambda is provided.
    """
    # --- Input checks ---
    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2-D, got shape {Phi.shape}")
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    m, H = Phi.shape
    if Y.shape[0] != m:
        raise ValueError(
            f"Row mismatch: Phi has {m} rows but Y has {Y.shape[0]} rows."
        )

    # Promote to float64 to avoid cancellation errors in the GCV denominator.
    Phi64 = Phi.astype(np.float64)
    Y64   = Y.astype(np.float64)

    # --- Economy SVD: Phi = U @ diag(d) @ Vt ---
    # U  : (m, r)   left  singular vectors
    # d  : (r,)     singular values
    # Vt : (r, H)   right singular vectors (transposed)
    # r = min(m, H)
    # NOTE: this is the ONE AND ONLY SVD per generation.
    U, d, Vt = np.linalg.svd(Phi64, full_matrices=False)
    d2  = d ** 2
    Uty = U.T @ Y64    # (r, 1)  — project targets once, reuse across all lambdas

    # --- Build lambda grid from singular values if not provided explicitly ---
    # This replaces the old adaptive_lambda_grid call, reusing d already computed.
    if lambda_candidates is None:
        lam_lo = max(float(d[-1] ** 2) / 100.0, 1e-6)
        lam_hi = min(float(d[0]  ** 2) * 10.0,  1e6)
        if lam_lo >= lam_hi:
            lam_lo = 1e-6
            lam_hi = 1e6
        lambda_candidates = list(
            np.logspace(np.log10(lam_lo), np.log10(lam_hi), n_lambda_points)
        )

    positives = [lam for lam in lambda_candidates if lam > 0]
    if not positives:
        raise ValueError("lambda_candidates must contain at least one positive value.")

    best_gcv    = np.inf
    best_lambda = positives[0]
    best_beta   : np.ndarray | None = None

    for lam in positives:
        # Filter: d^2 / (d^2 + lambda) are the diagonal entries of the hat matrix.
        d2_scale = d2 / (d2 + lam)                          # (r,)

        # Predictions: Y_hat = U @ diag(d2_scale) @ U.T @ Y
        Y_hat     = U @ (d2_scale[:, None] * Uty)           # (m, 1)
        residuals = Y64 - Y_hat
        mse       = np.mean(residuals ** 2)

        # Effective degrees of freedom = trace of the hat matrix.
        trace_S = np.sum(d2_scale)

        # GCV score = MSE / (1 - trace_S/m)^2
        denom = (1.0 - trace_S / m) ** 2
        if denom < 1e-12:
            continue

        gcv_score = mse / denom

        if gcv_score < best_gcv:
            best_gcv    = gcv_score
            best_lambda = lam
            # Recover full weights: beta = V @ diag(d / (d^2 + lambda)) @ U.T @ Y
            scale     = d / (d2 + lam)
            best_beta = Vt.T @ (scale[:, None] * Uty)       # (H, 1)

    # Fallback: if every candidate tripped the denom guard (shouldn't happen
    # in practice when m >> H), use the most-regularised lambda.
    if best_beta is None:
        lam         = max(positives)
        scale       = d / (d2 + lam)
        best_beta   = Vt.T @ (scale[:, None] * Uty)
        best_lambda = lam
        best_gcv    = np.inf

    return best_lambda, best_beta, best_gcv