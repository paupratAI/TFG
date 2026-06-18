"""
benchmark_level1.py

Matched-budget Pittsburgh variant for the binary logistic architecture.

The goal of this script is to isolate the hidden-representation difference
while keeping the logistic output solver fixed:

* Michigan: each individual is one hidden neuron. The population itself is the
  hidden layer. Neuron-level fitness is handled by CFS neuron selection.
* Pittsburgh matched-budget variant: each individual is a complete hidden
  layer. A whole candidate layer is evaluated with the same logistic output
  layer.

Both variants use:

* the same train/validation/test split;
* the same tanh hidden units;
* the same hidden-layer width;
* the same unregularised logistic output solver;
* the same validation-based final checkpoint selection.

The script writes all artefacts under runs/michigan_pittsburgh/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".font_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from core_es import MichiganESState, compute_activations, mutate_population
from data import load_dataset_with_validation
from logistic_output import (
    binary_brier_score,
    binary_ece,
    binary_log_loss_from_proba,
    compute_logistic_output,
    labels_to_binary,
    predict_proba as predict_logistic_proba,
)
from selection import cfs_selection, r2_fitness


DATASET_PRESETS = {
    "breast-w": {"id": 15, "name": None, "label": "Breast-W"},
    "ionosphere": {"id": 59, "name": None, "label": "Ionosphere"},
    "sonar": {"id": 40, "name": None, "label": "Sonar"},
}


@dataclass(frozen=True)
class DatasetSplit:
    X_tr: np.ndarray
    X_val: np.ndarray
    X_te: np.ndarray
    y_tr: np.ndarray
    y_val: np.ndarray
    y_te: np.ndarray


@dataclass(frozen=True)
class ArchitectureConfig:
    name: str
    label: str
    kind: str
    hidden_units: int
    epochs: int
    sigma_init: float
    michigan_offspring: int = 0
    pittsburgh_population: int = 0
    pittsburgh_offspring: int = 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def _mean_sem(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr, ddof=0))
    sem = float(std / np.sqrt(max(len(arr), 1)))
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "sem": sem,
    }


def _format_pm(summary: dict[str, float], error: str = "std") -> str:
    return f"{summary['mean']:.4f}\\pm{summary[error]:.4f}"


def _binary_metrics(
    y_pm1: np.ndarray,
    proba_pos: np.ndarray,
    pred_pm1: np.ndarray,
) -> dict[str, float | None]:
    y = np.asarray(y_pm1, dtype=np.float64).reshape(-1)
    y01 = labels_to_binary(y).reshape(-1)
    p_pos = np.asarray(proba_pos, dtype=np.float64).reshape(-1)
    p = np.asarray(pred_pm1, dtype=np.float64).reshape(-1)
    out: dict[str, float | None] = {
        "acc": float(accuracy_score(y, p)),
        "ce": binary_log_loss_from_proba(y01, p_pos),
        "brier": binary_brier_score(y01, p_pos),
        "ece": binary_ece(y01, p_pos),
    }
    if len(np.unique(y01)) == 2:
        out["roc_auc"] = float(roc_auc_score(y01, p_pos))
    else:
        out["roc_auc"] = None
    return out


def _fit_and_score_layer(
    alpha: np.ndarray,
    split: DatasetSplit,
) -> dict[str, Any]:
    Phi_tr = compute_activations(alpha, split.X_tr)
    Phi_tr_bias = np.c_[Phi_tr, np.ones(len(split.y_tr))]
    beta, train_ce, n_iter = compute_logistic_output(Phi_tr_bias, split.y_tr)

    train_proba = predict_logistic_proba(Phi_tr_bias, beta)
    train_pred = np.where(train_proba.reshape(-1) >= 0.5, 1.0, -1.0)
    train = _binary_metrics(split.y_tr, train_proba, train_pred)

    Phi_val = compute_activations(alpha, split.X_val)
    Phi_val_bias = np.c_[Phi_val, np.ones(len(split.y_val))]
    val_proba = predict_logistic_proba(Phi_val_bias, beta)
    val_pred = np.where(val_proba.reshape(-1) >= 0.5, 1.0, -1.0)
    validation = _binary_metrics(split.y_val, val_proba, val_pred)

    return {
        "beta": beta,
        "train_ce_objective": float(train_ce),
        "output_n_iter": int(n_iter),
        "train": train,
        "validation": validation,
    }


def _evaluate_final(
    alpha: np.ndarray,
    beta: np.ndarray,
    split: DatasetSplit,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, X, y in (
        ("train", split.X_tr, split.y_tr),
        ("validation", split.X_val, split.y_val),
        ("test", split.X_te, split.y_te),
    ):
        Phi = compute_activations(alpha, X)
        Phi_bias = np.c_[Phi, np.ones(len(y))]
        proba = predict_logistic_proba(Phi_bias, beta)
        pred = np.where(proba.reshape(-1) >= 0.5, 1.0, -1.0)
        out[name] = _binary_metrics(y, proba, pred)
    return out


def _is_better_checkpoint(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
) -> bool:
    if best is None:
        return True
    cand_val = float(candidate["validation"]["acc"])
    best_val = float(best["validation"]["acc"])
    cand_ce = float(candidate["validation"]["ce"])
    best_ce = float(best["validation"]["ce"])
    return cand_val > best_val + 1e-12 or (
        abs(cand_val - best_val) <= 1e-12 and cand_ce < best_ce
    )


def train_michigan_cfs(
    split: DatasetSplit,
    config: ArchitectureConfig,
    seed: int,
) -> dict[str, Any]:
    np.random.seed(seed)
    t0 = time.perf_counter()

    state = MichiganESState(
        mu=config.hidden_units,
        input_size=split.X_tr.shape[1],
        sigma_init=config.sigma_init,
    )

    history: list[dict[str, float | int]] = []
    best_checkpoint: dict[str, Any] | None = None
    ce_history: list[float] = []
    select_ms_history: list[float] = []

    stagnation_window = 15
    stagnation_threshold = 0.005
    sigma_reset_fraction = 0.25

    for epoch in range(1, config.epochs + 1):
        epoch_t0 = time.perf_counter()
        off_alpha, off_L = mutate_population(
            state,
            n_offspring=config.michigan_offspring,
        )
        pool_alpha = np.vstack([state.alpha, off_alpha])
        pool_L = np.vstack([state.L, off_L])
        Phi_pool = compute_activations(pool_alpha, split.X_tr)
        pool_r2 = r2_fitness(Phi_pool, split.y_tr)

        select_t0 = time.perf_counter()
        best_idx = cfs_selection(Phi_pool, split.y_tr, mu=config.hidden_units)
        select_ms_history.append((time.perf_counter() - select_t0) * 1e3)

        if epoch > 1 and state.beta is not None:
            elite_k = max(1, config.hidden_units // 10)
            elite_pool_idx = np.argsort(pool_r2[: config.hidden_units])[-elite_k:]
            best_list = best_idx.tolist()
            best_set = set(best_list)
            elite_set = set(elite_pool_idx.tolist())
            surv_scores = pool_r2[np.array(best_list, dtype=np.int64)]
            replace_positions = [
                pos for pos, sidx in enumerate(best_list) if sidx not in elite_set
            ]
            replace_positions.sort(key=lambda pos: surv_scores[pos])
            replace_ptr = 0
            for elite_idx in elite_pool_idx:
                if elite_idx not in best_set and replace_ptr < len(replace_positions):
                    rep_pos = replace_positions[replace_ptr]
                    best_list[rep_pos] = int(elite_idx)
                    best_set.add(int(elite_idx))
                    replace_ptr += 1
            best_idx = np.array(best_list, dtype=np.int64)

        state.alpha = pool_alpha[best_idx].copy()
        state.L = pool_L[best_idx].copy()

        scored = _fit_and_score_layer(state.alpha, split)
        state.beta = np.asarray(scored["beta"], dtype=np.float64)
        ce_history.append(float(scored["validation"]["ce"]))

        candidate_checkpoint = {
            "epoch": epoch,
            "alpha": state.alpha.copy(),
            "L": state.L.copy(),
            "beta": state.beta.copy(),
            "train_ce_objective": float(scored["train_ce_objective"]),
            "output_n_iter": int(scored["output_n_iter"]),
            "train": scored["train"],
            "validation": scored["validation"],
        }
        if _is_better_checkpoint(candidate_checkpoint, best_checkpoint):
            best_checkpoint = candidate_checkpoint

        reset_count = 0
        if len(ce_history) >= stagnation_window:
            recent_improvement = ce_history[-stagnation_window] - ce_history[-1]
            relative_improvement = recent_improvement / (
                abs(ce_history[-stagnation_window]) + 1e-12
            )
            if relative_improvement < stagnation_threshold:
                scores_surv = pool_r2[best_idx]
                n_reset = max(1, int(config.hidden_units * sigma_reset_fraction))
                worst_idx = np.argsort(scores_surv)[:n_reset]
                state.L[worst_idx] = np.stack(
                    [
                        np.eye(state.D, dtype=np.float64) * config.sigma_init
                        for _ in range(n_reset)
                    ]
                )
                reset_count = n_reset

        history.append(
            {
                "epoch": epoch,
                "train_acc": float(scored["train"]["acc"]),
                "validation_acc": float(scored["validation"]["acc"]),
                "train_ce": float(scored["train"]["ce"]),
                "validation_ce": float(scored["validation"]["ce"]),
                "train_brier": float(scored["train"]["brier"]),
                "validation_brier": float(scored["validation"]["brier"]),
                "train_ece": float(scored["train"]["ece"]),
                "validation_ece": float(scored["validation"]["ece"]),
                "select_ms": float(select_ms_history[-1]),
                "epoch_ms": float((time.perf_counter() - epoch_t0) * 1e3),
                "reset_count": reset_count,
            }
        )

    if best_checkpoint is None:
        raise RuntimeError("Michigan training produced no checkpoint.")

    final = _evaluate_final(
        np.asarray(best_checkpoint["alpha"], dtype=np.float64),
        np.asarray(best_checkpoint["beta"], dtype=np.float64),
        split,
    )
    elapsed = time.perf_counter() - t0
    metrics = {
        "architecture": config.name,
        "label": config.label,
        "kind": config.kind,
        "hidden_units": config.hidden_units,
        "epochs": config.epochs,
        "selected_epoch": int(best_checkpoint["epoch"]),
        "train_acc": final["train"]["acc"],
        "validation_acc": final["validation"]["acc"],
        "test_acc": final["test"]["acc"],
        "validation_test_gap": float(final["validation"]["acc"] - final["test"]["acc"]),
        "train_ce": final["train"]["ce"],
        "validation_ce": final["validation"]["ce"],
        "test_ce": final["test"]["ce"],
        "train_brier": final["train"]["brier"],
        "validation_brier": final["validation"]["brier"],
        "test_brier": final["test"]["brier"],
        "train_ece": final["train"]["ece"],
        "validation_ece": final["validation"]["ece"],
        "test_ece": final["test"]["ece"],
        "train_roc_auc": final["train"]["roc_auc"],
        "validation_roc_auc": final["validation"]["roc_auc"],
        "test_roc_auc": final["test"]["roc_auc"],
        "output_n_iter": int(best_checkpoint["output_n_iter"]),
        "elapsed_seconds": float(elapsed),
        "candidate_hidden_units_per_generation": int(
            config.hidden_units + config.michigan_offspring
        ),
        "new_hidden_units_per_generation": int(config.michigan_offspring),
        "output_fits_per_generation": 1,
        "select_ms_mean": float(np.mean(select_ms_history)),
    }
    return {"metrics": metrics, "history": history}


def _initial_pittsburgh_population(
    rng: np.random.Generator,
    population_size: int,
    hidden_units: int,
    input_size: int,
    sigma_init: float,
) -> tuple[np.ndarray, np.ndarray]:
    D = input_size + 1
    bound = 1.0 / np.sqrt(input_size)
    alpha = rng.uniform(
        low=-bound,
        high=bound,
        size=(population_size, hidden_units, D),
    ).astype(np.float64)
    base_L = np.eye(D, dtype=np.float64) * sigma_init
    L = np.broadcast_to(
        base_L,
        (population_size, hidden_units, D, D),
    ).copy()
    return alpha, L


def _mutate_pittsburgh_layers(
    rng: np.random.Generator,
    alpha_pop: np.ndarray,
    L_pop: np.ndarray,
    n_offspring_layers: int,
    beta_cov: float = 0.05,
    sigma_min: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    pop_size, hidden_units, D = alpha_pop.shape
    parent_idx = rng.integers(0, pop_size, size=n_offspring_layers)
    alpha_p = alpha_pop[parent_idx].copy()
    L_p = L_pop[parent_idx].copy()

    tau = 1.0 / np.sqrt(2.0 * D)
    tau_p = 1.0 / np.sqrt(2.0 * np.sqrt(D))

    diag_idx = np.arange(D)
    diag_vals = L_p[:, :, diag_idx, diag_idx]
    xi_global = rng.standard_normal((n_offspring_layers, hidden_units))
    xi_local = rng.standard_normal((n_offspring_layers, hidden_units, D))
    new_diag = diag_vals * np.exp(tau_p * xi_global[:, :, None] + tau * xi_local)
    sigma_max = 10.0
    L_p[:, :, diag_idx, diag_idx] = np.clip(new_diag, sigma_min, sigma_max)

    i_idx, j_idx = np.tril_indices(D, k=-1)
    L_p[:, :, i_idx, j_idx] += beta_cov * rng.standard_normal(
        (n_offspring_layers, hidden_units, len(i_idx))
    )

    z = rng.standard_normal((n_offspring_layers, hidden_units, D, 1))
    delta_alpha = np.matmul(L_p, z).squeeze(-1)
    off_alpha = (alpha_p + delta_alpha).astype(np.float64)
    L_p[:, :, diag_idx, diag_idx] = np.clip(
        L_p[:, :, diag_idx, diag_idx],
        sigma_min,
        sigma_max,
    )
    return off_alpha, L_p


def train_pittsburgh_es(
    split: DatasetSplit,
    config: ArchitectureConfig,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    alpha_pop, L_pop = _initial_pittsburgh_population(
        rng=rng,
        population_size=config.pittsburgh_population,
        hidden_units=config.hidden_units,
        input_size=split.X_tr.shape[1],
        sigma_init=config.sigma_init,
    )

    history: list[dict[str, float | int]] = []
    best_checkpoint: dict[str, Any] | None = None

    for epoch in range(1, config.epochs + 1):
        epoch_t0 = time.perf_counter()
        off_alpha, off_L = _mutate_pittsburgh_layers(
            rng,
            alpha_pop,
            L_pop,
            n_offspring_layers=config.pittsburgh_offspring,
        )
        pool_alpha = np.concatenate([alpha_pop, off_alpha], axis=0)
        pool_L = np.concatenate([L_pop, off_L], axis=0)

        candidate_scores: list[dict[str, Any]] = []
        for candidate_idx in range(pool_alpha.shape[0]):
            scored = _fit_and_score_layer(pool_alpha[candidate_idx], split)
            candidate_scores.append(scored)
            candidate_checkpoint = {
                "epoch": epoch,
                "alpha": pool_alpha[candidate_idx].copy(),
                "L": pool_L[candidate_idx].copy(),
                "beta": np.asarray(scored["beta"], dtype=np.float64).copy(),
                "train_ce_objective": float(scored["train_ce_objective"]),
                "output_n_iter": int(scored["output_n_iter"]),
                "train": scored["train"],
                "validation": scored["validation"],
            }
            if _is_better_checkpoint(candidate_checkpoint, best_checkpoint):
                best_checkpoint = candidate_checkpoint

        val_acc = np.array(
            [float(score["validation"]["acc"]) for score in candidate_scores],
            dtype=np.float64,
        )
        val_ce = np.array(
            [float(score["validation"]["ce"]) for score in candidate_scores],
            dtype=np.float64,
        )
        selected_idx = np.lexsort((val_ce, -val_acc))[
            : config.pittsburgh_population
        ]
        alpha_pop = pool_alpha[selected_idx].copy()
        L_pop = pool_L[selected_idx].copy()

        best_gen_idx = int(selected_idx[0])
        best_gen_score = candidate_scores[best_gen_idx]
        history.append(
            {
                "epoch": epoch,
                "train_acc": float(best_gen_score["train"]["acc"]),
                "validation_acc": float(best_gen_score["validation"]["acc"]),
                "train_ce": float(best_gen_score["train"]["ce"]),
                "validation_ce": float(best_gen_score["validation"]["ce"]),
                "train_brier": float(best_gen_score["train"]["brier"]),
                "validation_brier": float(best_gen_score["validation"]["brier"]),
                "train_ece": float(best_gen_score["train"]["ece"]),
                "validation_ece": float(best_gen_score["validation"]["ece"]),
                "epoch_ms": float((time.perf_counter() - epoch_t0) * 1e3),
                "evaluated_layers": int(pool_alpha.shape[0]),
            }
        )

    if best_checkpoint is None:
        raise RuntimeError("Pittsburgh training produced no checkpoint.")

    final = _evaluate_final(
        np.asarray(best_checkpoint["alpha"], dtype=np.float64),
        np.asarray(best_checkpoint["beta"], dtype=np.float64),
        split,
    )
    elapsed = time.perf_counter() - t0
    evaluated_layers = config.pittsburgh_population + config.pittsburgh_offspring
    metrics = {
        "architecture": config.name,
        "label": config.label,
        "kind": config.kind,
        "hidden_units": config.hidden_units,
        "epochs": config.epochs,
        "selected_epoch": int(best_checkpoint["epoch"]),
        "train_acc": final["train"]["acc"],
        "validation_acc": final["validation"]["acc"],
        "test_acc": final["test"]["acc"],
        "validation_test_gap": float(final["validation"]["acc"] - final["test"]["acc"]),
        "train_ce": final["train"]["ce"],
        "validation_ce": final["validation"]["ce"],
        "test_ce": final["test"]["ce"],
        "train_brier": final["train"]["brier"],
        "validation_brier": final["validation"]["brier"],
        "test_brier": final["test"]["brier"],
        "train_ece": final["train"]["ece"],
        "validation_ece": final["validation"]["ece"],
        "test_ece": final["test"]["ece"],
        "train_roc_auc": final["train"]["roc_auc"],
        "validation_roc_auc": final["validation"]["roc_auc"],
        "test_roc_auc": final["test"]["roc_auc"],
        "output_n_iter": int(best_checkpoint["output_n_iter"]),
        "elapsed_seconds": float(elapsed),
        "pittsburgh_population": int(config.pittsburgh_population),
        "pittsburgh_offspring": int(config.pittsburgh_offspring),
        "candidate_hidden_units_per_generation": int(
            evaluated_layers * config.hidden_units
        ),
        "new_hidden_units_per_generation": int(
            config.pittsburgh_offspring * config.hidden_units
        ),
        "output_fits_per_generation": int(evaluated_layers),
        "select_ms_mean": 0.0,
    }
    return {"metrics": metrics, "history": history}


def _architecture_configs(args: argparse.Namespace) -> list[ArchitectureConfig]:
    return [
        ArchitectureConfig(
            name="michigan_cfs",
            label="Michigan ES-CFS",
            kind="michigan",
            hidden_units=args.hidden_units,
            epochs=args.epochs,
            sigma_init=args.sigma_init,
            michigan_offspring=args.michigan_offspring,
        ),
        ArchitectureConfig(
            name="pittsburgh_matched",
            label="Pittsburgh matched-budget",
            kind="pittsburgh",
            hidden_units=args.hidden_units,
            epochs=args.epochs,
            sigma_init=args.sigma_init,
            pittsburgh_population=args.pittsburgh_matched_population,
            pittsburgh_offspring=args.pittsburgh_matched_offspring,
        ),
    ]


def _selected_datasets(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [DATASET_PRESETS[name] for name in args.datasets]


def _load_split(
    dataset: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> DatasetSplit:
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_dataset_with_validation(
        data_id=int(dataset["id"]),
        data_name=dataset["name"],
        version=args.data_version,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=seed,
        data_home=args.data_home,
    )
    return DatasetSplit(X_tr, X_val, X_te, y_tr, y_val, y_te)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    configs = _architecture_configs(args)
    runs: list[dict[str, Any]] = []

    for dataset in _selected_datasets(args):
        for seed in args.seeds:
            print(f"\n=== Dataset={dataset['label']} | seed={seed} ===")
            split = _load_split(dataset, args, seed)
            for config in configs:
                print(f"  -> Training {config.label}")
                if config.kind == "michigan":
                    result = train_michigan_cfs(split, config, seed)
                elif config.kind == "pittsburgh":
                    result = train_pittsburgh_es(split, config, seed)
                else:
                    raise ValueError(f"Unknown architecture kind: {config.kind}")
                runs.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "architecture": config.name,
                        "label": config.label,
                        "metrics": result["metrics"],
                        "history": result["history"],
                    }
                )
                metrics = result["metrics"]
                print(
                    "     "
                    f"val={float(metrics['validation_acc']):.4f} "
                    f"test={float(metrics['test_acc']):.4f} "
                    f"auc={float(metrics['test_roc_auc']):.4f} "
                    f"time={float(metrics['elapsed_seconds']):.2f}s"
                )

    payload = {
        "config": {
            "datasets": args.datasets,
            "seeds": [int(seed) for seed in args.seeds],
            "epochs": int(args.epochs),
            "hidden_units": int(args.hidden_units),
            "michigan_offspring": int(args.michigan_offspring),
            "pittsburgh_matched_population": int(args.pittsburgh_matched_population),
            "pittsburgh_matched_offspring": int(args.pittsburgh_matched_offspring),
            "pittsburgh_strong_population": int(args.pittsburgh_strong_population),
            "pittsburgh_strong_offspring": int(args.pittsburgh_strong_offspring),
            "validation_size": float(args.validation_size),
            "test_size": float(args.test_size),
            "sigma_init": float(args.sigma_init),
        },
        "runs": runs,
    }
    payload["summary"] = summarise_runs(payload)
    return payload


def summarise_runs(payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload["runs"]
    arch_order = [config.name for config in _architecture_configs_from_payload(payload)]
    dataset_labels = sorted({str(run["dataset"]["label"]) for run in runs})
    metric_names = (
        "train_acc",
        "validation_acc",
        "test_acc",
        "validation_test_gap",
        "test_ce",
        "test_brier",
        "test_ece",
        "test_roc_auc",
        "elapsed_seconds",
        "selected_epoch",
        "candidate_hidden_units_per_generation",
        "new_hidden_units_per_generation",
        "output_fits_per_generation",
    )

    summary: dict[str, Any] = {"overall": {}}
    for dataset_label in dataset_labels:
        summary[dataset_label] = {}
        dataset_runs = [
            run for run in runs if str(run["dataset"]["label"]) == dataset_label
        ]
        for arch in arch_order:
            arch_runs = [run for run in dataset_runs if run["architecture"] == arch]
            summary[dataset_label][arch] = _summarise_arch_runs(
                arch_runs,
                metric_names,
            )

    for arch in arch_order:
        arch_runs = [run for run in runs if run["architecture"] == arch]
        summary["overall"][arch] = _summarise_arch_runs(arch_runs, metric_names)

    return summary


def _architecture_configs_from_payload(payload: dict[str, Any]) -> list[ArchitectureConfig]:
    config = payload["config"]
    namespace = argparse.Namespace(
        hidden_units=config["hidden_units"],
        epochs=config["epochs"],
        sigma_init=config["sigma_init"],
        michigan_offspring=config["michigan_offspring"],
        pittsburgh_matched_population=config["pittsburgh_matched_population"],
        pittsburgh_matched_offspring=config["pittsburgh_matched_offspring"],
        pittsburgh_strong_population=config["pittsburgh_strong_population"],
        pittsburgh_strong_offspring=config["pittsburgh_strong_offspring"],
    )
    return _architecture_configs(namespace)


def _summarise_arch_runs(
    arch_runs: list[dict[str, Any]],
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_runs": len(arch_runs),
        "label": arch_runs[0]["label"] if arch_runs else "",
    }
    for metric in metric_names:
        values = [
            float(run["metrics"][metric])
            for run in arch_runs
            if run["metrics"].get(metric) is not None
            and np.isfinite(float(run["metrics"][metric]))
        ]
        if values:
            out[metric] = _mean_sem(values)
    if (
        "test_acc" in out
        and "elapsed_seconds" in out
        and float(out["elapsed_seconds"]["mean"]) > 0.0
    ):
        out["accuracy_per_second"] = {
            "mean": float(out["test_acc"]["mean"] / out["elapsed_seconds"]["mean"]),
            "std": 0.0,
            "sem": 0.0,
        }
    return out


def save_json(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "michigan_pittsburgh_summary.json").open(
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(_json_safe(payload), fh, indent=2, sort_keys=True)


def _arch_labels(payload: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (config.name, config.label)
        for config in _architecture_configs_from_payload(payload)
    ]


def save_accuracy_plot(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]["overall"]
    arch_labels = _arch_labels(payload)
    names = [name for name, _ in arch_labels]
    labels = [label for _, label in arch_labels]
    x = np.arange(len(names))
    width = 0.36

    val_means = [summary[name]["validation_acc"]["mean"] for name in names]
    val_sem = [summary[name]["validation_acc"]["sem"] for name in names]
    test_means = [summary[name]["test_acc"]["mean"] for name in names]
    test_sem = [summary[name]["test_acc"]["sem"] for name in names]

    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    ax.bar(x - width / 2, val_means, width, yerr=val_sem, capsize=4, label="Validation")
    ax.bar(x + width / 2, test_means, width, yerr=test_sem, capsize=4, label="Test")
    ax.set_title("Matched-budget Pittsburgh variant: validation and test accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    for idx, (val, test) in enumerate(zip(val_means, test_means)):
        ax.text(idx - width / 2, val + 0.012, f"{val:.3f}", ha="center", fontsize=8)
        ax.text(idx + width / 2, test + 0.012, f"{test:.3f}", ha="center", fontsize=8)
    fig.savefig(output_dir / "michigan_pittsburgh_accuracy.png", dpi=180)
    plt.close(fig)


def save_dataset_plot(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    dataset_labels = [key for key in summary.keys() if key != "overall"]
    arch_labels = _arch_labels(payload)
    names = [name for name, _ in arch_labels]
    labels = [label for _, label in arch_labels]

    fig, axes = plt.subplots(
        len(dataset_labels),
        1,
        figsize=(10.5, 3.2 * len(dataset_labels)),
        constrained_layout=True,
        squeeze=False,
    )
    x = np.arange(len(names))
    for ax, dataset_label in zip(axes.ravel(), dataset_labels):
        means = [summary[dataset_label][name]["test_acc"]["mean"] for name in names]
        sems = [summary[dataset_label][name]["test_acc"]["sem"] for name in names]
        ax.bar(x, means, yerr=sems, capsize=4)
        ax.set_title(f"{dataset_label}: test accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10, ha="right")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_dir / "michigan_pittsburgh_dataset_accuracy.png", dpi=180)
    plt.close(fig)


def save_cost_plot(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]["overall"]
    arch_labels = _arch_labels(payload)
    names = [name for name, _ in arch_labels]
    labels = [label for _, label in arch_labels]
    x = np.arange(len(names))

    elapsed = [summary[name]["elapsed_seconds"]["mean"] for name in names]
    elapsed_sem = [summary[name]["elapsed_seconds"]["sem"] for name in names]
    output_fits = [
        summary[name]["output_fits_per_generation"]["mean"] for name in names
    ]
    candidate_units = [
        summary[name]["candidate_hidden_units_per_generation"]["mean"]
        for name in names
    ]

    new_units = [
        summary[name]["new_hidden_units_per_generation"]["mean"]
        for name in names
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), constrained_layout=True)
    axes = axes.ravel()
    axes[0].bar(x, elapsed, yerr=elapsed_sem, capsize=4)
    axes[0].set_title("Mean fit time")
    axes[0].set_ylabel("Seconds")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, output_fits)
    axes[1].set_title("Logistic output fits per generation")
    axes[1].set_ylabel("Output fits")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(x, new_units)
    axes[2].set_title("New hidden units per generation")
    axes[2].set_ylabel("Hidden units")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25)

    axes[3].bar(x, candidate_units)
    axes[3].set_title("Candidate hidden units per generation")
    axes[3].set_ylabel("Hidden units")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=15, ha="right")
    axes[3].grid(True, axis="y", alpha=0.25)

    fig.savefig(output_dir / "michigan_pittsburgh_cost.png", dpi=180)
    plt.close(fig)


def save_accuracy_time_plot(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]["overall"]
    arch_labels = _arch_labels(payload)

    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    for name, label in arch_labels:
        x = summary[name]["elapsed_seconds"]["mean"]
        y = summary[name]["test_acc"]["mean"]
        xerr = summary[name]["elapsed_seconds"]["sem"]
        yerr = summary[name]["test_acc"]["sem"]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=4, markersize=8)
        ax.annotate(label, (x, y), xytext=(8, 5), textcoords="offset points")
    ax.set_title("Accuracy--time trade-off")
    ax.set_xlabel("Mean fit time (s)")
    ax.set_ylabel("Mean test accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    fig.savefig(output_dir / "michigan_pittsburgh_accuracy_time.png", dpi=180)
    plt.close(fig)


def save_convergence_plot(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arch_labels = _arch_labels(payload)
    epochs = int(payload["config"]["epochs"])
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)

    for name, label in arch_labels:
        series = []
        for run in payload["runs"]:
            if run["architecture"] != name:
                continue
            values = [
                float(row["validation_acc"])
                for row in run["history"]
                if "validation_acc" in row
            ]
            if len(values) == epochs:
                series.append(values)
        if not series:
            continue
        arr = np.asarray(series, dtype=np.float64)
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=0) / np.sqrt(arr.shape[0])
        xs = np.arange(1, epochs + 1)
        ax.plot(xs, mean, linewidth=2.0, label=label)
        ax.fill_between(xs, mean - sem, mean + sem, alpha=0.18)

    ax.set_title("Validation accuracy during evolution")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "michigan_pittsburgh_convergence.png", dpi=180)
    plt.close(fig)


def save_plots(payload: dict[str, Any], output_dir: Path) -> None:
    save_accuracy_plot(payload, output_dir)
    save_dataset_plot(payload, output_dir)
    save_cost_plot(payload, output_dir)
    save_accuracy_time_plot(payload, output_dir)
    save_convergence_plot(payload, output_dir)


def save_latex_notes(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    overall = summary["overall"]
    arch_labels = _arch_labels(payload)
    lines: list[str] = []

    lines.append("% Auto-generated by benchmark_level1.py")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Matched-budget Pittsburgh variant over Breast-W, Ionosphere, and Sonar.}"
    )
    lines.append("\\label{tab:exp_michigan_pittsburgh_overall}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}lrrrrrrr@{}}")
    lines.append("\\toprule")
    lines.append(
        "Architecture & Val Acc & Test Acc & Val--Test Gap & Test AUC & Fit s & New neurons/gen & Output fits/gen \\\\"
    )
    lines.append("\\midrule")
    for name, label in arch_labels:
        row = overall[name]
        lines.append(
            f"{label} & "
            f"${_format_pm(row['validation_acc'], 'sem')}$ & "
            f"${_format_pm(row['test_acc'], 'sem')}$ & "
            f"${_format_pm(row['validation_test_gap'], 'sem')}$ & "
            f"${_format_pm(row['test_roc_auc'], 'sem')}$ & "
            f"${_format_pm(row['elapsed_seconds'], 'sem')}$ & "
            f"${row['new_hidden_units_per_generation']['mean']:.0f}$ & "
            f"${row['output_fits_per_generation']['mean']:.1f}$ \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Per-dataset mean test accuracy in the matched-budget Pittsburgh variant.}"
    )
    lines.append("\\label{tab:exp_michigan_pittsburgh_dataset}")
    lines.append("\\begin{tabular}{@{}lrrr@{}}")
    lines.append("\\toprule")
    labels = [label for _, label in arch_labels]
    lines.append(
        "Dataset & "
        + " & ".join(labels)
        + " \\\\"
    )
    lines.append("\\midrule")
    for dataset_label in [key for key in summary.keys() if key != "overall"]:
        cells = []
        for name, _ in arch_labels:
            cells.append(f"${_format_pm(summary[dataset_label][name]['test_acc'], 'sem')}$")
        lines.append(f"{dataset_label} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{figure}[H]")
    lines.append("\\centering")
    lines.append(
        "\\includegraphics[width=0.95\\textwidth]{runs/michigan_pittsburgh/michigan_pittsburgh_accuracy.png}"
    )
    lines.append(
        "\\caption{Validation and test accuracy for Michigan and the matched-budget Pittsburgh variant.}"
    )
    lines.append("\\label{fig:exp_michigan_pittsburgh_accuracy}")
    lines.append("\\end{figure}")
    lines.append("")
    lines.append("\\begin{figure}[H]")
    lines.append("\\centering")
    lines.append(
        "\\includegraphics[width=0.95\\textwidth]{runs/michigan_pittsburgh/michigan_pittsburgh_cost.png}"
    )
    lines.append(
        "\\caption{Computational cost of Michigan and the matched-budget Pittsburgh variant.}"
    )
    lines.append("\\label{fig:exp_michigan_pittsburgh_cost}")
    lines.append("\\end{figure}")
    lines.append("")
    lines.append("\\begin{figure}[H]")
    lines.append("\\centering")
    lines.append(
        "\\includegraphics[width=0.88\\textwidth]{runs/michigan_pittsburgh/michigan_pittsburgh_accuracy_time.png}"
    )
    lines.append(
        "\\caption{Accuracy--time trade-off in the matched-budget Pittsburgh variant.}"
    )
    lines.append("\\label{fig:exp_michigan_pittsburgh_accuracy_time}")
    lines.append("\\end{figure}")

    with (output_dir / "latex_notes.tex").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def print_summary(payload: dict[str, Any]) -> None:
    overall = payload["summary"]["overall"]
    print("\n## Matched-Budget Pittsburgh Variant Summary")
    print(
        "| Architecture | Val Acc | Test Acc | Val-Test Gap | Test AUC | Fit s | New neurons/gen | Output fits/gen |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, label in _arch_labels(payload):
        row = overall[name]
        print(
            f"| {label} | "
            f"{row['validation_acc']['mean']:.4f} +/- {row['validation_acc']['sem']:.4f} | "
            f"{row['test_acc']['mean']:.4f} +/- {row['test_acc']['sem']:.4f} | "
            f"{row['validation_test_gap']['mean']:.4f} +/- {row['validation_test_gap']['sem']:.4f} | "
            f"{row['test_roc_auc']['mean']:.4f} +/- {row['test_roc_auc']['sem']:.4f} | "
            f"{row['elapsed_seconds']['mean']:.2f} +/- {row['elapsed_seconds']['sem']:.2f} | "
            f"{row['new_hidden_units_per_generation']['mean']:.0f} | "
            f"{row['output_fits_per_generation']['mean']:.1f} |"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched-budget Pittsburgh variant for the binary logistic architecture."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_PRESETS.keys()),
        default=["breast-w", "ionosphere", "sonar"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-units", type=int, default=50)
    parser.add_argument("--michigan-offspring", type=int, default=250)
    parser.add_argument("--pittsburgh-matched-population", type=int, default=5)
    parser.add_argument("--pittsburgh-matched-offspring", type=int, default=5)
    parser.add_argument("--pittsburgh-strong-population", type=int, default=12)
    parser.add_argument("--pittsburgh-strong-offspring", type=int, default=12)
    parser.add_argument("--sigma-init", type=float, default=0.3)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-version", type=int, default=1)
    parser.add_argument("--data-home", default=".openml_cache")
    parser.add_argument("--output-dir", default="runs/michigan_pittsburgh")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    payload = run_benchmark(args)
    save_json(payload, output_dir)
    save_plots(payload, output_dir)
    save_latex_notes(payload, output_dir)
    print_summary(payload)
    print(f"\nSaved artefacts under {output_dir}")


if __name__ == "__main__":
    main()
