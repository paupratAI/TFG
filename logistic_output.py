"""
logistic_output.py — Unregularised logistic output layer.

This module replaces the Ridge+GCV output solve used by the original prototype.
Hidden activations are still produced by the evolved tanh neurons, but the final
linear score is interpreted as a logit:

    p(y=+1 | X) = sigmoid(Phi @ beta)
    p(y=-1 | X) = 1 - p(y=+1 | X)

The output weights are fitted with Newton/IRLS on binary cross-entropy.  There
is no regularisation parameter and no GCV-style lambda selection in this
formulation.
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
    max_iter: int,
    tol: float,
    damping: float,
) -> tuple[np.ndarray, float, float]:
    """
    Fit one unregularised logistic model with Newton updates.

    Returns beta, mean log-loss, and final Newton step norm.
    """
    Phi64 = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y01, dtype=np.float64).reshape(-1, 1)
    _, H = Phi64.shape

    beta = np.zeros((H, 1), dtype=np.float64)

    for _ in range(max_iter):
        logits = Phi64 @ beta
        proba = sigmoid(logits)
        weights = np.maximum((proba * (1.0 - proba)).ravel(), 1e-9)

        grad = Phi64.T @ (proba - y)
        hessian = (Phi64.T * weights) @ Phi64
        hessian += np.eye(H, dtype=np.float64) * damping

        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]

        beta -= step
        if np.linalg.norm(step) <= tol * (1.0 + np.linalg.norm(beta)):
            break

    proba = predict_proba(Phi64, beta)
    log_loss = binary_log_loss_from_proba(y, proba)
    return beta, log_loss, float(np.linalg.norm(step))


def compute_logistic_output(
    Phi     : np.ndarray,
    Y       : np.ndarray,
    max_iter: int = 50,
    tol     : float = 1e-8,
    damping : float = 1e-8,
) -> tuple[np.ndarray, float, float]:
    """
    Fit the unregularised logistic output layer.

    Parameters
    ----------
    Phi : (m, H) design matrix, including the output bias column.
    Y   : (m,) or (m,1) labels in {-1,+1}.

    Returns
    -------
    beta      : (H,1) output weights.
    log_loss  : mean binary cross-entropy.
    step_norm : norm of the final Newton step, useful as a convergence trace.
    """
    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2-D, got shape {Phi.shape}.")
    y01 = labels_to_binary(Y)
    if y01.shape[0] != Phi.shape[0]:
        raise ValueError(
            f"Row mismatch: Phi has {Phi.shape[0]} rows but Y has {y01.shape[0]} rows."
        )

    beta, log_loss, step_norm = _fit_logistic_newton(
        Phi,
        y01,
        max_iter=max_iter,
        tol=tol,
        damping=damping,
    )
    return beta, log_loss, step_norm
