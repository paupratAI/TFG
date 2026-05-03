"""
data.py — Dataset loading and preprocessing.

Loads a binary classification dataset from OpenML, applies a
leak-free stratified train/test split, imputes missing values, and
standardises the features.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_dataset(
    data_id  : int        = 59,
    data_name: str | None = None,
    version  : int        = 1,
    test_size: float      = 0.2,
    seed     : int        = 42,
    data_home: str | None = ".openml_cache",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a binary classification dataset from OpenML and return
    leak-free train/test splits with standardised features.

    The pipeline follows the strict order:
        1. Split first  →  fit imputer/scaler on train only  →  transform both.
    Fitting preprocessing on the full dataset before splitting would leak
    test-set statistics into the training process.

    Labels are mapped to {-1, +1}.  The logistic output layer internally maps
    them to {0, 1} with +1 as the positive class.

    Parameters
    ----------
    data_id   : OpenML numeric dataset ID. Used by default.
    data_name : Optional OpenML dataset name. When provided, it takes priority
                over data_id. The default is None so benchmark scripts that
                pass different data_id values really load different datasets.
    version   : Dataset version to fetch when data_name is not None.
                Default 1 pins the canonical first release.
    test_size : Fraction of data held out for testing. The split is stratified.
    seed      : Random seed for reproducibility.
    data_home : Local OpenML cache directory. The default keeps cache writes
                inside the repository workspace instead of the user home.

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
            name=data_name,
            version=version,
            as_frame=False,
            parser="auto",
            data_home=data_home,
        )
    else:
        print(f"Loading OpenML dataset id={data_id} ...")
        data = fetch_openml(
            data_id=data_id,
            as_frame=False,
            parser="auto",
            data_home=data_home,
        )
    try:
        X = np.asarray(data.data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "load_dataset expects numeric OpenML features. "
            "Categorical datasets need an explicit encoding step before this prototype."
        ) from exc
    y = np.asarray(data.target)

    # Map the two class labels to {-1, +1}.
    unique = np.unique(y)
    if len(unique) != 2:
        raise ValueError(
            f"load_dataset expects a binary classification dataset, "
            f"but found {len(unique)} classes: {unique[:5]}..."
        )
    y_bin  = np.where(y == unique[0], 1.0, -1.0).astype(np.float64)

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    # Step 1: Stratified split.
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X,
        y_bin,
        test_size=test_size,
        random_state=seed,
        stratify=y_bin,
    )

    # Steps 2-3: Fit preprocessing on train, transform both.
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr_raw)
    X_te_imp = imputer.transform(X_te_raw)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_imp).astype(np.float64)
    X_te   = scaler.transform(X_te_imp).astype(np.float64)

    if not (np.isfinite(X_tr).all() and np.isfinite(X_te).all()):
        raise ValueError("Non-finite values remain after imputation/scaling.")

    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)

    print(
        f"  -> {X_tr.shape[0]} train / {X_te.shape[0]} test, "
        f"{X_tr.shape[1]} features."
    )
    return X_tr, X_te, y_tr, y_te


def load_dataset_with_validation(
    data_id        : int        = 59,
    data_name      : str | None = None,
    version        : int        = 1,
    validation_size: float      = 0.2,
    test_size      : float      = 0.2,
    seed           : int        = 42,
    data_home      : str | None = ".openml_cache",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a binary OpenML dataset and return leak-free train/validation/test splits.

    The split order is:
        1. Hold out test from the full dataset.
        2. Hold out validation from the remaining training candidate set.
        3. Fit imputer and scaler on the final training split only.
        4. Transform validation and test with the training preprocessing.

    This keeps both validation and test outside preprocessing fitting, which is
    important when validation is used for model/strategy selection.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")
    if not 0.0 < validation_size < 1.0:
        raise ValueError(
            f"validation_size must be in (0, 1), got {validation_size}."
        )

    if data_name is not None:
        print(f"Loading OpenML dataset name='{data_name}' version={version} ...")
        data = fetch_openml(
            name=data_name,
            version=version,
            as_frame=False,
            parser="auto",
            data_home=data_home,
        )
    else:
        print(f"Loading OpenML dataset id={data_id} ...")
        data = fetch_openml(
            data_id=data_id,
            as_frame=False,
            parser="auto",
            data_home=data_home,
        )

    try:
        X = np.asarray(data.data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "load_dataset_with_validation expects numeric OpenML features. "
            "Categorical datasets need an explicit encoding step before this prototype."
        ) from exc
    y = np.asarray(data.target)

    unique = np.unique(y)
    if len(unique) != 2:
        raise ValueError(
            f"load_dataset_with_validation expects a binary classification dataset, "
            f"but found {len(unique)} classes: {unique[:5]}..."
        )
    y_bin = np.where(y == unique[0], 1.0, -1.0).astype(np.float64)

    X_trainval_raw, X_te_raw, y_trainval, y_te = train_test_split(
        X,
        y_bin,
        test_size=test_size,
        random_state=seed,
        stratify=y_bin,
    )
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_trainval_raw,
        y_trainval,
        test_size=validation_size,
        random_state=seed + 1,
        stratify=y_trainval,
    )

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_te_imp = imputer.transform(X_te_raw)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_imp).astype(np.float64)
    X_val = scaler.transform(X_val_imp).astype(np.float64)
    X_te = scaler.transform(X_te_imp).astype(np.float64)

    if not (
        np.isfinite(X_tr).all()
        and np.isfinite(X_val).all()
        and np.isfinite(X_te).all()
    ):
        raise ValueError("Non-finite values remain after imputation/scaling.")

    y_tr = y_tr.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)

    print(
        f"  -> {X_tr.shape[0]} train / {X_val.shape[0]} validation / "
        f"{X_te.shape[0]} test, {X_tr.shape[1]} features."
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te
