"""
gcv.py — Analytic Ridge Regression with Generalised Cross-Validation (GCV)
==========================================================================
Solves the H2O (hidden-to-output) layer analytically.

Mathematical background
-----------------------
Given the feature matrix  Φ ∈ ℝ^{m×H}  and targets  Y ∈ ℝ^{m×1}  the
Ridge solution is:

    β̂(λ) = (ΦᵀΦ + λI)⁻¹ ΦᵀY                              (1)

The hat (smoother) matrix is:

    S(λ) = Φ (ΦᵀΦ + λI)⁻¹ Φᵀ ∈ ℝ^{m×m}                    (2)

GCV score (Allen 1974 / Golub et al. 1979):

    GCV(λ) = (1/m) ‖Y - Ŷ‖² / (1 - tr(S)/m)²              (3)

Efficient trace identity (avoids forming the m×m hat matrix):

    tr(S) = tr(Φ (ΦᵀΦ + λI)⁻¹ Φᵀ)
           = tr((ΦᵀΦ + λI)⁻¹ ΦᵀΦ)      [cyclic property]
           = tr(K (K + λI)⁻¹)            where K = ΦᵀΦ     (4)

This keeps the cost O(H³) per lambda instead of O(m³).

SVD-based implementation
------------------------
Decompose  Φ = U D Vᵀ  (economy SVD, d = min(m,H) singular values).
Then:

    K = ΦᵀΦ = V D² Vᵀ
    β̂(λ) = V diag(d²/(d²+λ)) U D / ... [see code]          (5)
    tr(S) = Σ_k  d_k² / (d_k² + λ)                          (6)
    Ŷ     = U diag(d²/(d²+λ)) Uᵀ Y                          (7)

Using the SVD avoids explicit matrix inversion (numerically superior)
and lets us sweep over all λ candidates in O(d) operations each after
a single O(m·H·min(m,H)) decomposition.
"""

import numpy as np


def compute_ridge_gcv(
    Phi: np.ndarray,
    Y: np.ndarray,
    lambda_candidates: list[float] = None,
) -> tuple[float, np.ndarray, float]:
    """
    Select λ via GCV and return the corresponding Ridge weights β.

    Parameters
    ----------
    Phi : ndarray of shape (m, H)
        Design / feature matrix (hidden activations + optional bias column).
    Y   : ndarray of shape (m, 1)
        Target vector.  For binary classification: {-1, +1}.
    lambda_candidates : list[float] | None
        Regularisation values to evaluate.  A logarithmically-spaced default
        is used when None.

    Returns
    -------
    best_lambda : float
        λ that minimises GCV(λ).
    best_beta   : ndarray of shape (H, 1)
        Ridge weights β̂(best_lambda).
    best_gcv    : float
        Minimum GCV score achieved.

    Raises
    ------
    ValueError
        If Phi or Y have incompatible shapes.
    """
    # ------------------------------------------------------------------
    # 0.  Input validation
    # ------------------------------------------------------------------
    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2-D, got shape {Phi.shape}")
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    m, H = Phi.shape
    if Y.shape[0] != m:
        raise ValueError(
            f"Row mismatch: Phi has {m} rows but Y has {Y.shape[0]} rows."
        )

    if lambda_candidates is None:
        # 13 values from 1e-4 to 1e4 on a log scale
        lambda_candidates = list(np.logspace(-4, 4, 13))

    # ------------------------------------------------------------------
    # 1.  Economy SVD  —  Φ = U · diag(d) · Vᵀ
    #     U  ∈ ℝ^{m×r},  d  ∈ ℝ^r,  V  ∈ ℝ^{H×r}   where r = min(m,H)
    # ------------------------------------------------------------------
    U, d, Vt = np.linalg.svd(Phi, full_matrices=False)   # economy SVD
    d2 = d ** 2                                           # squared sing. vals
    Uty = U.T @ Y                                         # (r, 1) — precomputed

    # ------------------------------------------------------------------
    # 2.  Sweep λ candidates (O(r) per candidate after the SVD)
    # ------------------------------------------------------------------
    best_gcv    = np.inf
    best_lambda = lambda_candidates[0]
    best_beta   = None

    for lam in lambda_candidates:
        if lam <= 0:
            continue

        # Ridge coefficients in SVD coordinates  (r,)
        # β̂ = V diag(d/(d²+λ)) Uᵀ Y
        scale = d / (d2 + lam)          # (r,)   — element-wise

        # Predictions  Ŷ = U diag(d²/(d²+λ)) Uᵀ Y
        d2_scale = d2 / (d2 + lam)      # (r,)
        Y_hat    = U @ (d2_scale[:, None] * Uty)   # (m, 1)

        # Residual MSE
        residuals = Y - Y_hat
        mse       = np.mean(residuals ** 2)

        # Effective degrees of freedom / trace of hat matrix  — eq. (6)
        trace_S = np.sum(d2_scale)

        # GCV denominator  (1 - tr(S)/m)²
        denom = (1.0 - trace_S / m) ** 2
        if denom < 1e-12:
            # Perfectly interpolating model; skip this λ
            continue

        gcv_score = mse / denom

        if gcv_score < best_gcv:
            best_gcv    = gcv_score
            best_lambda = lam
            # Recover full β̂ ∈ ℝ^{H×1}
            best_beta   = Vt.T @ (scale[:, None] * Uty)  # (H, 1)

    # ------------------------------------------------------------------
    # 3.  Fallback: if somehow every candidate failed, use largest λ
    # ------------------------------------------------------------------
    if best_beta is None:
        lam       = max(lambda_candidates)
        scale     = d / (d2 + lam)
        best_beta = Vt.T @ (scale[:, None] * Uty)
        best_lambda = lam
        best_gcv    = np.inf

    return best_lambda, best_beta, best_gcv