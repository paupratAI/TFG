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
"""

import numpy as np


def compute_ridge_gcv(
    Phi              : np.ndarray,
    Y                : np.ndarray,
    lambda_candidates: list[float] | None = None,
) -> tuple[float, np.ndarray, float]:
    """
    Find the best Ridge penalty lambda via GCV, then return the
    corresponding weights beta.

    Parameters
    ----------
    Phi               : (m, H)  feature matrix (hidden activations + bias column).
    Y                 : (m,) or (m, 1)  target vector.
    lambda_candidates : List of positive regularisation values to try.
                        If None, 17 log-spaced values on [1e-6, 1e6] are used.
                        Providing an explicit list is recommended so the grid
                        boundaries are known.

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

    if lambda_candidates is None:
        # 17 values spanning [1e-6, 1e6].  The wide range reduces the chance
        # of the optimal lambda sitting at a grid boundary.
        lambda_candidates = list(np.logspace(-6, 6, 17))

    positives = [lam for lam in lambda_candidates if lam > 0]
    if not positives:
        raise ValueError("lambda_candidates must contain at least one positive value.")

    # --- Economy SVD: Phi = U @ diag(d) @ Vt ---
    # U  : (m, r)   left  singular vectors
    # d  : (r,)     singular values
    # Vt : (r, H)   right singular vectors (transposed)
    # r = min(m, H)
    U, d, Vt = np.linalg.svd(Phi64, full_matrices=False)
    d2  = d ** 2
    Uty = U.T @ Y64    # (r, 1)  — project targets once, reuse across all lambdas

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
        # The denominator is the GCV "self-influence" correction.
        # It approaches zero only when lambda -> 0 AND m <= H (more features
        # than samples).  In our regime m >> H, so this guard is precautionary.
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