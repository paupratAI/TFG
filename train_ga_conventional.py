"""
train_ga_conventional.py — Conventional Genetic Algorithm baseline.

This script mirrors the prototype's objective (binary classification with a
tanh hidden layer and analytic Ridge+GCV output solve), but uses a standard
GA instead of a Michigan-style ES.

Individual representation
-------------------------
Each individual encodes the WHOLE hidden layer:
    chromosome shape = (hidden_units, n_features + 1)
where the last column is neuron bias.

Evolution cycle (generational GA with elitism)
----------------------------------------------
1) Evaluate population fitness.
2) Keep elite individuals.
3) Fill the rest via tournament selection + arithmetic crossover + mutation.
4) Repeat for the configured number of generations.

Usage
-----
    python train_ga_conventional.py
"""

from __future__ import annotations

import time
import numpy as np
from sklearn.metrics import accuracy_score

from data import load_dataset
from gcv import compute_ridge_gcv


def compute_hidden_activations(chromosome: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute hidden activations for one GA individual.

    Parameters
    ----------
    chromosome : (H, D) with D = n_features + 1.
    X          : (m, n_features)

    Returns
    -------
    Phi : (m, H) tanh activations.
    """
    W = chromosome[:, :-1]  # (H, n_features)
    b = chromosome[:, -1]   # (H,)
    return np.tanh(X @ W.T + b).astype(np.float64)


def evaluate_individual(
    chromosome: np.ndarray,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
) -> tuple[float, float, np.ndarray, float]:
    """
    Evaluate one individual with the same output-layer solver as the prototype.

    Returns
    -------
    fitness     : scalar fitness (higher is better)
    train_acc   : training accuracy
    beta        : output weights from Ridge+GCV
    best_gcv    : best GCV score
    """
    Phi = compute_hidden_activations(chromosome, X_tr)
    Phi_bias = np.c_[Phi, np.ones(len(y_tr))]

    _, beta, best_gcv = compute_ridge_gcv(Phi_bias, y_tr)
    y_hat = Phi_bias @ beta
    train_pred = np.where(y_hat.ravel() > 0, 1.0, -1.0)
    train_acc = accuracy_score(y_tr.ravel(), train_pred)

    # Maximise accuracy while softly preferring better (smaller) GCV.
    fitness = float(train_acc - 0.01 * np.log1p(best_gcv))
    return fitness, float(train_acc), beta, float(best_gcv)


def tournament_select(
    rng: np.random.Generator,
    fitness: np.ndarray,
    k: int = 3,
) -> int:
    """Return index of winner from k-way tournament."""
    idx = rng.integers(0, len(fitness), size=k)
    return int(idx[np.argmax(fitness[idx])])


def arithmetic_crossover(
    rng: np.random.Generator,
    p1: np.ndarray,
    p2: np.ndarray,
    crossover_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend parents with convex combination."""
    if rng.random() >= crossover_rate:
        return p1.copy(), p2.copy()

    alpha = rng.random()
    c1 = alpha * p1 + (1.0 - alpha) * p2
    c2 = alpha * p2 + (1.0 - alpha) * p1
    return c1, c2


def mutate(
    rng: np.random.Generator,
    child: np.ndarray,
    mutation_rate: float,
    mutation_scale: float,
    init_bound: float,
) -> np.ndarray:
    """Element-wise Gaussian mutation."""
    mask = rng.random(child.shape) < mutation_rate
    noise = rng.normal(0.0, mutation_scale, size=child.shape)
    child = child + mask * noise
    return np.clip(child, -3.0 * init_bound, 3.0 * init_bound)


def train_conventional_ga(
    generations: int = 100,
    pop_size: int = 80,
    hidden_units: int = 50,
    elite_size: int = 4,
    tournament_k: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.08,
    mutation_scale: float = 0.10,
    data_id: int = 150,
    verbose: int = 10,
    seed: int = 42,
) -> dict[str, float | np.ndarray]:
    """
    Run a conventional GA baseline and report comparable metrics.
    """
    if elite_size >= pop_size:
        raise ValueError("elite_size must be < pop_size.")

    rng = np.random.default_rng(seed)
    X_tr, X_te, y_tr, y_te = load_dataset(data_id=data_id, seed=seed)
    n_features = X_tr.shape[1]
    D = n_features + 1
    init_bound = 1.0 / np.sqrt(n_features)

    # Population shape: (pop_size, hidden_units, D)
    population = rng.uniform(
        low=-init_bound,
        high=init_bound,
        size=(pop_size, hidden_units, D),
    ).astype(np.float64)

    best_fitness = -np.inf
    best_chromosome = population[0].copy()
    best_beta = None
    best_gcv = np.inf
    best_train_acc = 0.0

    t0 = time.perf_counter()
    print(
        f"\nConventional GA | pop={pop_size} | H={hidden_units} | "
        f"gens={generations} | elite={elite_size}\n" + "-" * 65
    )

    for gen in range(1, generations + 1):
        fitness = np.empty(pop_size, dtype=np.float64)
        train_acc = np.empty(pop_size, dtype=np.float64)
        gcv_vals = np.empty(pop_size, dtype=np.float64)
        betas: list[np.ndarray] = []

        for i in range(pop_size):
            fi, acci, bi, gcvi = evaluate_individual(population[i], X_tr, y_tr)
            fitness[i] = fi
            train_acc[i] = acci
            gcv_vals[i] = gcvi
            betas.append(bi)

        gen_best = int(np.argmax(fitness))
        if fitness[gen_best] > best_fitness:
            best_fitness = float(fitness[gen_best])
            best_chromosome = population[gen_best].copy()
            best_beta = betas[gen_best].copy()
            best_gcv = float(gcv_vals[gen_best])
            best_train_acc = float(train_acc[gen_best])

        if gen % verbose == 0 or gen == 1:
            print(
                f"Gen {gen:4d}/{generations}  |  "
                f"Best fitness={fitness[gen_best]:.5f}  |  "
                f"Train Acc={train_acc[gen_best]:.4f}  |  "
                f"GCV={gcv_vals[gen_best]:.5f}"
            )

        # --- Next generation ---
        elite_idx = np.argsort(fitness)[-elite_size:]
        next_population = [population[i].copy() for i in elite_idx]

        while len(next_population) < pop_size:
            p1_idx = tournament_select(rng, fitness, k=tournament_k)
            p2_idx = tournament_select(rng, fitness, k=tournament_k)
            c1, c2 = arithmetic_crossover(
                rng, population[p1_idx], population[p2_idx], crossover_rate
            )
            c1 = mutate(rng, c1, mutation_rate, mutation_scale, init_bound)
            c2 = mutate(rng, c2, mutation_rate, mutation_scale, init_bound)
            next_population.append(c1)
            if len(next_population) < pop_size:
                next_population.append(c2)

        population = np.stack(next_population, axis=0)

    elapsed = time.perf_counter() - t0

    # Final test evaluation with best individual found during search.
    if best_beta is None:
        # Defensive fallback (should not happen).
        _, _, best_beta, best_gcv = evaluate_individual(best_chromosome, X_tr, y_tr)

    Phi_test = compute_hidden_activations(best_chromosome, X_te)
    Phi_test_bias = np.c_[Phi_test, np.ones(len(y_te))]
    test_hat = Phi_test_bias @ best_beta
    y_test_pred = np.where(test_hat.ravel() > 0, 1.0, -1.0)
    test_acc = accuracy_score(y_te.ravel(), y_test_pred)

    print(f"\n{'=' * 65}")
    print(f"  Best Train Acc: {best_train_acc:.4f}")
    print(f"  Best GCV      : {best_gcv:.6f}")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Total time    : {elapsed:.2f} s")
    print(f"{'=' * 65}\n")

    return {
        "best_fitness": best_fitness,
        "best_train_acc": best_train_acc,
        "best_gcv": best_gcv,
        "test_acc": float(test_acc),
        "elapsed_seconds": float(elapsed),
        "best_chromosome": best_chromosome,
        "best_beta": best_beta,
    }


if __name__ == "__main__":
    train_conventional_ga(
        generations=100,
        pop_size=80,
        hidden_units=50,
        elite_size=4,
        tournament_k=3,
        crossover_rate=0.9,
        mutation_rate=0.08,
        mutation_scale=0.10,
        data_id=150,
        verbose=10,
        seed=42,
    )
