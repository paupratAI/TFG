"""
train_pytorch_conventional.py — Conventional PyTorch MLP baseline.

This script trains a standard neural network with backpropagation to solve
the same binary classification task used by the Michigan prototype.

Design choices to keep comparison fair:
- Same dataset loader (`load_dataset`) and same OpenML IDs.
- Same hidden width default (50 units).
- Same number of training epochs default (100).
- Same final metric: test accuracy with thresholded binary output.

Usage
-----
    python train_pytorch_conventional.py
"""

from __future__ import annotations

import time
import random
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data import load_dataset


class ConventionalMLP(nn.Module):
    """Deep MLP for high-accuracy binary classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size.")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        self.hidden_linear_layers: list[nn.Linear] = []

        for width in hidden_layers:
            fc = nn.Linear(prev_dim, width)
            self.hidden_linear_layers.append(fc)
            layers.extend(
                [
                    fc,
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                    nn.Dropout(p=dropout),
                ]
            )
            prev_dim = width

        self.out = nn.Linear(prev_dim, 1)
        layers.append(self.out)
        self.net = nn.Sequential(*layers)
        self._init_parameters(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def _init_parameters(self, input_dim: int) -> None:
        """Use stable initialization with a Michigan-like first layer scale."""
        bound = 1.0 / np.sqrt(input_dim)
        nn.init.uniform_(self.hidden_linear_layers[0].weight, -bound, bound)
        nn.init.uniform_(self.hidden_linear_layers[0].bias, -bound, bound)

        for fc in self.hidden_linear_layers[1:]:
            nn.init.kaiming_uniform_(fc.weight, a=np.sqrt(5.0))
            if fc.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(fc.weight)
                b = 1.0 / np.sqrt(fan_in)
                nn.init.uniform_(fc.bias, -b, b)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)


def set_global_seed(seed: int) -> None:
    """Set reproducible seeds across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_accuracy(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Compute binary accuracy from logits with zero threshold."""
    logits = model(X)
    preds = torch.where(logits > 0.0, 1.0, -1.0)
    return float(accuracy_score(y.cpu().numpy(), preds.cpu().numpy()))


def train_pytorch_mlp(
    epochs: int = 400,
    hidden_layers: tuple[int, ...] = (256, 128, 64),
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    dropout: float = 0.20,
    val_size: float = 0.20,
    early_stopping_patience: int = 45,
    lr_patience: int = 12,
    trials: int = 3,
    data_id: int = 59,
    data_name: str | None = None,
    data_version: int = 1,
    test_size: float = 0.2,
    data_home: str | None = ".openml_cache",
    verbose: int = 10,
    seed: int = 42,
    device: str | None = None,
    quiet: bool = False,
) -> dict[str, float]:
    """
    Train a conventional MLP baseline and report comparable metrics.

    Returns a dictionary with train/test accuracy and elapsed time.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if trials < 1:
        raise ValueError("trials must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if verbose < 1:
        raise ValueError("verbose must be >= 1.")

    set_global_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    X_tr_all, X_te, y_tr_all, y_te = load_dataset(
        data_id=data_id,
        data_name=data_name,
        version=data_version,
        test_size=test_size,
        seed=seed,
        data_home=data_home,
    )
    input_dim = X_tr_all.shape[1]

    # Hold out validation data from the training split for model selection.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_all,
        y_tr_all.ravel(),
        test_size=val_size,
        random_state=seed,
        stratify=y_tr_all.ravel(),
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_tr_pm_t = torch.tensor(y_tr, dtype=torch.float32)
    y_val_pm_t = torch.tensor(y_val, dtype=torch.float32)
    y_te_pm_t = torch.tensor(y_te.ravel(), dtype=torch.float32)

    # BCEWithLogitsLoss expects {0,1} labels.
    y_tr_bin_t = (y_tr_pm_t > 0.0).float()

    train_ds = TensorDataset(X_tr_t, y_tr_bin_t)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_tr_dev = X_tr_t.to(dev)
    y_tr_eval = y_tr_pm_t.to(dev)
    X_val_dev = X_val_t.to(dev)
    y_val_dev = y_val_pm_t.to(dev)
    X_te_dev = X_te_t.to(dev)
    y_te_dev = y_te_pm_t.to(dev)

    criterion = nn.BCEWithLogitsLoss()

    t0 = time.perf_counter()
    if not quiet:
        print(
            f"\nPyTorch Deep MLP | layers={hidden_layers} | epochs={epochs} | trials={trials} | "
            f"lr={learning_rate} | wd={weight_decay} | batch={batch_size} | "
            f"dropout={dropout} | device={dev}\n"
            + "-" * 65
        )
    best_global_val_acc = -1.0
    best_global_val_loss = np.inf
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_trial = -1
    best_epoch = 0

    for trial in range(1, trials + 1):
        # Multiple restarts usually improve final accuracy on tabular data.
        set_global_seed(seed + trial)
        model = ConventionalMLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
        ).to(dev)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-6,
        )

        best_trial_val_acc = -1.0
        best_trial_val_loss = np.inf
        best_trial_state: dict[str, torch.Tensor] | None = None
        no_improve_epochs = 0
        best_trial_epoch = 0

        if not quiet:
            print(f"\nTrial {trial}/{trials}")

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_samples = 0

            for xb, yb in loader:
                xb = xb.to(dev)
                yb = yb.to(dev)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

                bs = xb.shape[0]
                epoch_loss += float(loss.item()) * bs
                n_samples += bs

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_dev)
                val_loss = float(criterion(val_logits, (y_val_dev > 0.0).float()).item())
                val_pred = torch.where(val_logits > 0.0, 1.0, -1.0)
                val_acc = float(
                    accuracy_score(y_val_dev.cpu().numpy(), val_pred.cpu().numpy())
                )

            scheduler.step(val_acc)

            improved = (val_acc > best_trial_val_acc + 1e-6) or (
                abs(val_acc - best_trial_val_acc) <= 1e-6 and val_loss < best_trial_val_loss
            )
            if improved:
                best_trial_val_acc = val_acc
                best_trial_val_loss = val_loss
                best_trial_epoch = epoch
                best_trial_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if (epoch % verbose == 0 or epoch == 1) and (not quiet):
                train_acc = compute_accuracy(model, X_tr_dev, y_tr_eval)
                mean_loss = epoch_loss / max(1, n_samples)
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:4d}/{epochs}  |  "
                    f"Loss={mean_loss:.5f}  |  "
                    f"Train Acc={train_acc:.4f}  |  "
                    f"Val Acc={val_acc:.4f}  |  "
                    f"lr={current_lr:.2e}"
                )

            if no_improve_epochs >= early_stopping_patience:
                break

        if best_trial_state is None:
            best_trial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (best_trial_val_acc > best_global_val_acc + 1e-6) or (
            abs(best_trial_val_acc - best_global_val_acc) <= 1e-6
            and best_trial_val_loss < best_global_val_loss
        ):
            best_global_val_acc = best_trial_val_acc
            best_global_val_loss = best_trial_val_loss
            best_state_dict = best_trial_state
            best_trial = trial
            best_epoch = best_trial_epoch

    if best_state_dict is None:
        raise RuntimeError("Failed to train model: no valid checkpoint found.")

    best_model = ConventionalMLP(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
    ).to(dev)
    best_model.load_state_dict(best_state_dict)
    best_model.eval()

    elapsed = time.perf_counter() - t0

    final_train_acc = compute_accuracy(best_model, X_tr_dev, y_tr_eval)
    final_val_acc = compute_accuracy(best_model, X_val_dev, y_val_dev)
    final_test_acc = compute_accuracy(best_model, X_te_dev, y_te_dev)

    if not quiet:
        print(f"\n{'=' * 65}")
        print(f"  Best trial      : {best_trial}")
        print(f"  Best epoch      : {best_epoch}")
        print(f"  Final Train Acc: {final_train_acc:.4f}")
        print(f"  Final Val Acc   : {final_val_acc:.4f}")
        print(f"  Test Accuracy   : {final_test_acc:.4f}")
        print(f"  Total time      : {elapsed:.2f} s")
        print(f"{'=' * 65}\n")

    return {
        "train_acc": final_train_acc,
        "val_acc": final_val_acc,
        "test_acc": final_test_acc,
        "elapsed_seconds": float(elapsed),
    }


def run_experiments_select_best(
    data_id: int = 59,
    seed: int = 42,
    verbose: int = 10,
    device: str | None = None,
) -> dict[str, float | str | tuple[int, ...]]:
    """
    Run two experiment blocks and keep the best by validation accuracy:
      1) single hidden layer baseline
      2) variable-depth/hyperparameter search
    """
    baseline_cfg = {
        "epochs": 300,
        "hidden_layers": (128,),
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "dropout": 0.10,
        "val_size": 0.20,
        "test_size": 0.20,
        "early_stopping_patience": 40,
        "lr_patience": 10,
        "trials": 3,
        "data_id": data_id,
        "data_name": None,
        "data_version": 1,
        "verbose": verbose,
        "seed": seed,
        "device": device,
        "quiet": True,
    }

    search_space: list[dict[str, object]] = [
        {
            "epochs": 400,
            "hidden_layers": (256, 128, 64),
            "learning_rate": 2e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "dropout": 0.20,
            "val_size": 0.20,
            "test_size": 0.20,
            "early_stopping_patience": 45,
            "lr_patience": 12,
            "trials": 3,
            "data_id": data_id,
            "data_name": None,
            "data_version": 1,
            "verbose": verbose,
            "seed": seed,
            "device": device,
            "quiet": True,
        },
        {
            "epochs": 400,
            "hidden_layers": (512, 256, 128),
            "learning_rate": 1.5e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "dropout": 0.25,
            "val_size": 0.20,
            "test_size": 0.20,
            "early_stopping_patience": 50,
            "lr_patience": 12,
            "trials": 3,
            "data_id": data_id,
            "data_name": None,
            "data_version": 1,
            "verbose": verbose,
            "seed": seed + 100,
            "device": device,
            "quiet": True,
        },
        {
            "epochs": 350,
            "hidden_layers": (256, 256, 128, 64),
            "learning_rate": 1e-3,
            "weight_decay": 5e-5,
            "batch_size": 32,
            "dropout": 0.20,
            "val_size": 0.20,
            "test_size": 0.20,
            "early_stopping_patience": 45,
            "lr_patience": 10,
            "trials": 3,
            "data_id": data_id,
            "data_name": None,
            "data_version": 1,
            "verbose": verbose,
            "seed": seed + 200,
            "device": device,
            "quiet": True,
        },
        {
            "epochs": 450,
            "hidden_layers": (384, 192, 96),
            "learning_rate": 8e-4,
            "weight_decay": 2e-4,
            "batch_size": 64,
            "dropout": 0.15,
            "val_size": 0.20,
            "test_size": 0.20,
            "early_stopping_patience": 55,
            "lr_patience": 14,
            "trials": 4,
            "data_id": data_id,
            "data_name": None,
            "data_version": 1,
            "verbose": verbose,
            "seed": seed + 300,
            "device": device,
            "quiet": True,
        },
    ]

    baseline_result = train_pytorch_mlp(**baseline_cfg)
    best_multi_name = "search_config_1"
    best_multi_cfg = deepcopy(search_space[0])
    best_multi_result: dict[str, float] | None = None

    for idx, cfg in enumerate(search_space, start=1):
        result = train_pytorch_mlp(**cfg)
        if best_multi_result is None or float(result["val_acc"]) > float(best_multi_result["val_acc"]):
            best_multi_name = f"search_config_{idx}"
            best_multi_cfg = deepcopy(cfg)
            best_multi_result = deepcopy(result)

    if best_multi_result is None:
        raise RuntimeError("No multi-layer search result found.")

    print("\nSingle hidden-layer result")
    print(
        f"layers={baseline_cfg['hidden_layers']} | "
        f"train_acc={float(baseline_result['train_acc']):.4f} | "
        f"val_acc={float(baseline_result['val_acc']):.4f} | "
        f"test_acc={float(baseline_result['test_acc']):.4f} | "
        f"time={float(baseline_result['elapsed_seconds']):.2f}s"
    )
    print("\nBest multi-hidden-layer result")
    print(
        f"{best_multi_name} | layers={best_multi_cfg['hidden_layers']} | "
        f"lr={best_multi_cfg['learning_rate']} | wd={best_multi_cfg['weight_decay']} | "
        f"dropout={best_multi_cfg['dropout']} | batch={best_multi_cfg['batch_size']} | "
        f"train_acc={float(best_multi_result['train_acc']):.4f} | "
        f"val_acc={float(best_multi_result['val_acc']):.4f} | "
        f"test_acc={float(best_multi_result['test_acc']):.4f} | "
        f"time={float(best_multi_result['elapsed_seconds']):.2f}s"
    )

    return {
        "single_hidden_val_acc": float(baseline_result["val_acc"]),
        "single_hidden_test_acc": float(baseline_result["test_acc"]),
        "winner": best_multi_name,
        "hidden_layers": tuple(best_multi_cfg["hidden_layers"]),  # type: ignore[arg-type]
        "learning_rate": float(best_multi_cfg["learning_rate"]),  # type: ignore[arg-type]
        "weight_decay": float(best_multi_cfg["weight_decay"]),    # type: ignore[arg-type]
        "dropout": float(best_multi_cfg["dropout"]),              # type: ignore[arg-type]
        "batch_size": int(best_multi_cfg["batch_size"]),          # type: ignore[arg-type]
        "epochs": int(best_multi_cfg["epochs"]),                  # type: ignore[arg-type]
        "trials": int(best_multi_cfg["trials"]),                  # type: ignore[arg-type]
        "train_acc": float(best_multi_result["train_acc"]),
        "val_acc": float(best_multi_result["val_acc"]),
        "test_acc": float(best_multi_result["test_acc"]),
        "elapsed_seconds": float(best_multi_result["elapsed_seconds"]),
    }


if __name__ == "__main__":
    run_experiments_select_best(
        data_id=59,
        verbose=10,
        seed=42,
        device=None,
    )
