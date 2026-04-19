"""
data.py — Dataset loading and preprocessing.

Loads a binary classification dataset from OpenML, applies a
leak-free train/test split, and standardises the features.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(
    data_id  : int        = 150,
    data_name: str | None = "ionosphere",
    version  : int        = 1,
    test_size: float      = 0.2,
    seed     : int        = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a binary classification dataset from OpenML and return
    leak-free train/test splits with standardised features.

    The pipeline follows the strict order:
        1. Split first  →  fit scaler on train only  →  transform both.
    Fitting the scaler before splitting would leak test-set statistics
    into the training process (data leakage).

    Labels are mapped to {-1, +1} for compatibility with the Ridge
    regression output layer.

    Parameters
    ----------
    data_id   : OpenML numeric dataset ID. Used only when data_name is None.
    data_name : OpenML dataset name (default "breast-w"). When not None,
                this takes priority over data_id. Fetching by name+version
                is stable across OpenML library versions.
    version   : Dataset version to fetch when data_name is not None.
                Default 1 pins the canonical first release.
    test_size : Fraction of data held out for testing.
    seed      : Random seed for reproducibility.

    Returns
    -------
    X_tr : (m_train, n_features)  float64  standardised training features.
    X_te : (m_test,  n_features)  float64  standardised test features.
    y_tr : (m_train, 1)           float64  labels in {-1, +1}.
    y_te : (m_test,  1)           float64  labels in {-1, +1}.
    """
    if data_name is not None:
        print(f"Loading OpenML dataset name='{data_name}' version={version} ...")
        data = fetch_openml(
            name=data_name, version=version, as_frame=False, parser="auto"
        )
    else:
        print(f"Loading OpenML dataset id={data_id} ...")
        data = fetch_openml(data_id=data_id, as_frame=False, parser="auto")
    X, y = data.data, data.target

    # Map the two class labels to {-1, +1}.
    unique = np.unique(y)
    if len(unique) != 2:
        raise ValueError(
            f"load_dataset expects a binary classification dataset, "
            f"but found {len(unique)} classes: {unique[:5]}..."
        )
    y_bin  = np.where(y == unique[0], 1.0, -1.0).astype(np.float64)

    # Step 1: Split.
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y_bin, test_size=test_size, random_state=seed
    )

    # Steps 2-3: Fit on train, transform both.
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw).astype(np.float64)
    X_te   = scaler.transform(X_te_raw).astype(np.float64)

    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)

    print(
        f"  -> {X_tr.shape[0]} train / {X_te.shape[0]} test, "
        f"{X_tr.shape[1]} features."
    )
    return X_tr, X_te, y_tr, y_te