"""
logistic_output.py — Logistic and softmax output layers.

The hidden representation is produced elsewhere by the evolutionary prototype.
This module only fits the hidden-to-output layer with cross-entropy objectives:

  - binary logistic regression for two classes;
  - multinomial softmax regression for K classes.

No Ridge/GCV regularisation is used here.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    z64 = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z64)
    pos = z64 >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z64[pos]))
    exp_z = np.exp(z64[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


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
    """Return p(y=+1|X) for a binary design matrix that includes the bias column."""
    return sigmoid(np.asarray(Phi, dtype=np.float64) @ np.asarray(beta, dtype=np.float64))


def predict_class_proba(Phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return two columns: [p(y=-1|X), p(y=+1|X)]."""
    p_pos = predict_proba(Phi, beta).reshape(-1, 1)
    return np.c_[1.0 - p_pos, p_pos]


def predict_pm1(Phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return {-1,+1} predictions from logistic probabilities."""
    proba = predict_proba(Phi, beta)
    return np.where(proba.ravel() >= 0.5, 1.0, -1.0)


def _fit_sklearn_logistic(
    Phi: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    tol: float,
) -> LogisticRegression:
    """Fit an unregularised logistic/softmax model with no intercept."""
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        fit_intercept=False,
        max_iter=max_iter,
        tol=tol,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(np.asarray(Phi, dtype=np.float64), np.asarray(y).reshape(-1))
    return model


def compute_logistic_output(
    Phi: np.ndarray,
    Y: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float, int]:
    """
    Fit an unregularised binary logistic output layer.

    Phi must already include the output bias column.  Labels are expected in
    {-1,+1}.  Returns beta, training log-loss, and iteration count.
    """
    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2-D, got shape {Phi.shape}.")
    y01 = labels_to_binary(Y).reshape(-1)
    if y01.shape[0] != Phi.shape[0]:
        raise ValueError(f"Row mismatch: Phi has {Phi.shape[0]} rows but Y has {y01.shape[0]}.")

    model = _fit_sklearn_logistic(Phi, y01, max_iter=max_iter, tol=tol)
    coef = model.coef_.reshape(1, -1)
    beta = coef.T
    proba = model.predict_proba(Phi)[:, 1]
    return beta, binary_log_loss_from_proba(y01, proba), int(np.max(model.n_iter_))


def compute_softmax_output(
    Phi: np.ndarray,
    y: np.ndarray,
    n_classes: int | None = None,
    max_iter: int = 150,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float, int]:
    """
    Fit an unregularised multinomial softmax output layer.

    Phi must already include the output bias column.  Labels are integer class
    indices.  Returns beta with shape (n_features, n_classes), training
    cross-entropy, and iteration count.
    """
    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2-D, got shape {Phi.shape}.")
    y_idx = np.asarray(y).reshape(-1).astype(int)
    if y_idx.shape[0] != Phi.shape[0]:
        raise ValueError(f"Row mismatch: Phi has {Phi.shape[0]} rows but y has {y_idx.shape[0]}.")
    classes = np.arange(n_classes if n_classes is not None else int(np.max(y_idx)) + 1)
    if not set(np.unique(y_idx).tolist()).issubset(set(classes.tolist())):
        raise ValueError("y contains class indices outside the expected class range.")

    model = _fit_sklearn_logistic(Phi, y_idx, max_iter=max_iter, tol=tol)
    beta = model.coef_.T.astype(np.float64)
    proba = model.predict_proba(Phi)
    ce = float(log_loss(y_idx, proba, labels=classes))
    return beta, ce, int(np.max(model.n_iter_))


def predict_softmax_proba(Phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Predict class probabilities from a design matrix and softmax weights."""
    return softmax(np.asarray(Phi, dtype=np.float64) @ np.asarray(beta, dtype=np.float64))


def predict_softmax_class(Phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Predict class indices with argmax softmax probability."""
    return np.argmax(predict_softmax_proba(Phi, beta), axis=1)
