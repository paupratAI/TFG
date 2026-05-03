"""
logistic_output.py — L2-regularised logistic output layer.

This module replaces the Ridge+GCV output solve used by the original prototype.
Hidden activations are still produced by the evolved tanh neurons, but the final
linear score is interpreted as a logit:

    p(y=+1 | X) = sigmoid(Phi @ beta)
    p(y=-1 | X) = 1 - p(y=+1 | X)

The output weights are fitted with Newton/IRLS on binary cross-entropy plus an
L2 penalty.  The bias column is not regularised.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    z64 = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z64)
    pos = z64 >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z64[pos]))
    exp_z = np.exp(z64[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def labels_to_binary(Y: np.ndarray) -> np.ndarray:
    """Map {-1,+1} labels to {0,1}, preserving +1 as the positive class."""
    y = np.asarray(Y).reshape(-1, 1)
    unique = np.unique(y)
    if not set(unique.tolist()).issubset({-1.0, 1.0}):
        raise ValueError(f"Expected labels in {{-1,+1}}, found {unique}.")
    return (y > 0.0).astype(np.float64)


def binary_log_loss_from_proba(y01: np.ndarray, proba: np.ndarray) -> float:
    """Mean binary cross-entropy from probabilities."""
    y = np.asarray(y01, dtype=np.float64).reshape(-1, 1)
    p = np.clip(np.asarray(proba, dtype=np.float64).reshape(-1, 1), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def predict_proba(Phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return p(y=+1|X) for a design matrix that already includes the bias column."""
    return sigmoid(np.asarray(Phi, dtype=np.float64) @ np.asarray(beta, dtype=np.float64))


def predict_class_proba(Phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return two columns: [p(y=-1|X), p(y=+1|X)]."""
    p_pos = predict_proba(Phi, beta).reshape(-1, 1)
    return np.c_[1.0 - p_pos, p_pos]


def predict_pm1(Phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return {-1,+1} predictions from logistic probabilities."""
    proba = predict_proba(Phi, beta)
    return np.where(proba.ravel() >= 0.5, 1.0, -1.0)


def _fit_logistic_newton(
    Phi: np.ndarray,
    y01: np.ndarray,
    lam: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float, float]:
    """
    Fit one L2-regularised logistic model with Newton updates.

    Returns beta, unregularised mean log-loss, and penalised mean objective.
    """
    Phi64 = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y01, dtype=np.float64).reshape(-1, 1)
    m, H = Phi64.shape

    beta = np.zeros((H, 1), dtype=np.float64)
    penalty_mask = np.ones((H, 1), dtype=np.float64)
    penalty_mask[-1, 0] = 0.0  # do not regularise the output bias
    penalty_diag = penalty_mask.ravel()

    for _ in range(max_iter):
        logits = Phi64 @ beta
        proba = sigmoid(logits)
        weights = np.maximum((proba * (1.0 - proba)).ravel(), 1e-9)

        grad = Phi64.T @ (proba - y) + lam * penalty_mask * beta
        hessian = (Phi64.T * weights) @ Phi64
        hessian += np.diag(lam * penalty_diag + 1e-8)

        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]

        beta -= step
        if np.linalg.norm(step) <= tol * (1.0 + np.linalg.norm(beta)):
            break

    proba = predict_proba(Phi64, beta)
    log_loss = binary_log_loss_from_proba(y, proba)
    penalty = 0.5 * lam * float(np.sum((penalty_mask * beta) ** 2)) / m
    objective = log_loss + penalty
    return beta, log_loss, objective


def compute_logistic_l2(
    Phi              : np.ndarray,
    Y                : np.ndarray,
    lambda_candidates: list[float] | None = None,
    n_lambda_points  : int = 20,
    lambda_override  : float | None = None,
    max_iter         : int = 50,
    tol              : float = 1e-8,
) -> tuple[float, np.ndarray, float, float]:
    """
    Fit the logistic output layer and choose an L2 penalty.

    Parameters
    ----------
    Phi : (m, H) design matrix, including the output bias column.
    Y   : (m,) or (m,1) labels in {-1,+1}.

    Returns
    -------
    best_lambda    : selected L2 penalty.
    best_beta      : (H,1) output weights.
    best_log_loss  : mean binary cross-entropy without the penalty term.
    best_objective : mean binary cross-entropy plus L2 penalty.
    """
    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2-D, got shape {Phi.shape}.")
    y01 = labels_to_binary(Y)
    if y01.shape[0] != Phi.shape[0]:
        raise ValueError(
            f"Row mismatch: Phi has {Phi.shape[0]} rows but Y has {y01.shape[0]} rows."
        )

    if lambda_override is not None:
        candidates = [float(lambda_override)]
    elif lambda_candidates is not None:
        candidates = [float(lam) for lam in lambda_candidates if lam >= 0.0]
    else:
        candidates = list(np.logspace(-6, 2, n_lambda_points))

    if not candidates:
        raise ValueError("lambda_candidates must contain at least one non-negative value.")

    best_lambda = candidates[0]
    best_beta: np.ndarray | None = None
    best_log_loss = np.inf
    best_objective = np.inf

    for lam in candidates:
        beta, log_loss, objective = _fit_logistic_newton(
            Phi,
            y01,
            lam=lam,
            max_iter=max_iter,
            tol=tol,
        )
        if objective < best_objective:
            best_lambda = lam
            best_beta = beta
            best_log_loss = log_loss
            best_objective = objective

    if best_beta is None:
        raise RuntimeError("Failed to fit logistic output layer.")

    return best_lambda, best_beta, best_log_loss, best_objective
