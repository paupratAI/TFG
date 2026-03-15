"""
main.py — Michigan-style ES/MLP  (corrected after stagnation diagnosis)
=======================================================================

WHY THE PREVIOUS VERSION STAGNATED — ROOT CAUSE ANALYSIS
─────────────────────────────────────────────────────────

Root Cause 1 ── FATAL: Gradient sensitivity selected in the WRONG direction
    The formula  s(i) = Σ|∂E/∂α_i|  expands (chain rule) to:
        s(i) = |β̂_i| · (2/m) Σ_j |Σ_k r_k · sech²(z_{ki}) · x_{kj}|
    where r_k = ŷ_k - y_k are the RESIDUALS.

    A neuron that already fits the data well has r_k ≈ 0  →  s(i) ≈ 0
    and gets ELIMINATED.  A noisy, under-fitted neuron with large residuals
    survives.  This is the exact INVERSE of what selection should do once
    we have a pool larger than a single neuron.

Root Cause 2 ── FATAL: Cooperative Credit Assignment Failure
    β̂ is computed via Ridge on ALL 400 neurons simultaneously.
    If neurons A and B both explain the same variance in y, Ridge sets
    β̂_A ≈ c, β̂_B ≈ 0 (or splits them arbitrarily).
    Then s(B) ≈ 0 → B eliminated even though B is intrinsically good.
    The fitness signal becomes NOISE: which of two equally-good neurons
    survives depends on Ridge's arbitrary weight allocation, not quality.

Root Cause 3 ── IMPORTANT: Missing diversity pressure
    50 neurons can collapse onto a single dominant feature direction,
    wasting representational capacity (effectively H=1 not H=50).

Root Cause 4 ── MODERATE: σ random walk without boundary
    τ' ≈ 0.26 > τ ≈ 0.09 (for D=55) causes global correlated drift
    of all L diagonal entries.  Without σ_min clamping, sigma can
    become negligible and halt all exploration.

FIXES APPLIED
─────────────
Fix 1 — New fitness: R²(φᵢ, y) — Squared Pearson Correlation
    score(i) = Cor²(φᵢ, y)
    Properties:
      ✓ Independent per neuron  — no cooperative credit problem
      ✓ = 0 for saturated neurons  (std(φᵢ)→0)
      ✓ = 0 for random neurons  (Cov(φᵢ,y)≈0)
      ✓ Scale-invariant — unaffected by β̂
      ✓ O(mP) total, no backward pass needed

Fix 2 — Diversity-penalised greedy selection
    Greedily selects μ neurons maximising:
        effective_score(i) = R²(φᵢ,y) − γ · max_{j∈selected} |Cor(φᵢ,φⱼ)|
    Prevents all 50 survivors covering the same feature direction.
    γ = 0.3 balances fitness vs diversity.

Fix 3 — σ boundary rule
    Clamp  L_{ii} ≥ σ_min = 1e-4  after every mutation step.
    Prevents sigma collapse → continued exploration.

Fix 4 — Larger σ_init = 0.3
    The original σ_init = 0.1 produced mutation steps of ~0.1/√D ≈ 0.013
    in each weight coordinate — far too small to escape initialisation.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from gcv import compute_ridge_gcv

# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(data_id: int = 150, max_samples: int = 10_000):
    print(f"Loading OpenML dataset id={data_id} …")
    data  = fetch_openml(data_id=data_id, as_frame=False, parser="auto")
    X, y  = data.data, data.target

    unique = np.unique(y)
    y_bin  = np.where(y == unique[0], 1.0, -1.0).astype(np.float32)

    if X.shape[0] > max_samples:
        idx   = np.random.choice(X.shape[0], max_samples, replace=False)
        X, y_bin = X[idx], y_bin[idx]

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_bin, test_size=0.2, random_state=SEED
    )
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)

    print(f"  → {X_tr.shape[0]} train / {X_te.shape[0]} test, "
          f"{X_tr.shape[1]} features.")
    return X_tr, X_te, y_tr, y_te


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ES STATE
# ═══════════════════════════════════════════════════════════════════════════════

class MichiganESState:
    """
    Population of μ neurons.

    Attributes
    ----------
    alpha : (mu, D)     genotype matrix; row i = [weights_i | bias_i]
    L     : (mu, D, D)  lower-Cholesky strategy parameters
    beta  : (mu+1, 1)   analytic output weights (set after first Ridge solve)
    """
    def __init__(self, mu: int, input_size: int, sigma_init: float = 0.3):
        D = input_size + 1
        bound      = 1.0 / np.sqrt(input_size)
        self.alpha = np.random.uniform(-bound, bound, (mu, D)).astype(np.float32)
        self.L     = np.stack([np.eye(D, dtype=np.float32) * sigma_init
                               for _ in range(mu)])
        self.beta  = None
        self.mu    = mu
        self.D     = D


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MUTATION — self-adaptive full covariance ES (pure numpy)
# ═══════════════════════════════════════════════════════════════════════════════

def mutate_population(
    state      : MichiganESState,
    n_offspring: int   = 350,
    beta_cov   : float = 0.05,
    sigma_min  : float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n_offspring via multivariate Gaussian self-adaptive mutation.

    Mutation equations
    ------------------
    τ  = 1/√(2D)     global learning rate
    τ' = 1/√(2√D)    local  learning rate

    Diagonal (variance):
        L'_{ii} = max(L_{ii} · exp(τ'ξ₀ + τξᵢ),  σ_min)   [Boundary Rule]

    Off-diagonal (covariance):
        L'_{ij} = L_{ij} + β_cov · N(0,1)    for i > j

    Phenotype:
        α' = α + L' z,    z ~ N(0, I)
    """
    mu, D  = state.mu, state.D
    tau    = 1.0 / np.sqrt(2.0 * D)
    tau_p  = 1.0 / np.sqrt(2.0 * np.sqrt(D))

    parent_idx = np.random.randint(0, mu, n_offspring)
    alpha_p    = state.alpha[parent_idx].copy()     # (n_off, D)
    L_p        = state.L[parent_idx].copy()         # (n_off, D, D)

    # ── Mutate strategy (L matrix) ────────────────────────────────────── #
    xi_global = np.random.randn(n_offspring)         # (n_off,)
    xi_local  = np.random.randn(n_offspring, D)      # (n_off, D)

    # Diagonal — multiplicative, always positive
    diag      = np.arange(D)
    diag_vals = L_p[:, diag, diag]                   # (n_off, D)
    new_diag  = diag_vals * np.exp(
        tau_p * xi_global[:, None] + tau * xi_local
    )
    L_p[:, diag, diag] = np.maximum(new_diag, sigma_min)   # Boundary Rule

    # Off-diagonal — additive
    i_idx, j_idx = np.tril_indices(D, k=-1)
    L_p[:, i_idx, j_idx] += beta_cov * np.random.randn(n_offspring, len(i_idx))

    # ── Mutate phenotype ──────────────────────────────────────────────── #
    z           = np.random.randn(n_offspring, D, 1)
    delta_alpha = np.matmul(L_p, z).squeeze(-1)             # (n_off, D)
    off_alpha   = (alpha_p + delta_alpha).astype(np.float32)

    return off_alpha, L_p


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ACTIVATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_activations(alpha: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Φ = tanh(X Wᵀ + b)  ∈ ℝ^{m × P}

    Parameters
    ----------
    alpha : (P, D)
    X     : (m, n)

    Returns
    -------
    Phi : (m, P)
    """
    W = alpha[:, :-1]           # (P, n)
    b = alpha[:, -1]            # (P,)
    return np.tanh(X @ W.T + b).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FITNESS — R²(φᵢ, y) per neuron
# ═══════════════════════════════════════════════════════════════════════════════

def r2_fitness(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Squared Pearson Correlation (Your Method) """
    y_v   = y.ravel().astype(np.float64)
    Phi64 = Phi.astype(np.float64)
    phi_c  = Phi64 - Phi64.mean(axis=0, keepdims=True)
    y_c    = y_v   - y_v.mean()
    cov    = (phi_c * y_c[:, None]).mean(axis=0)
    std_ph = phi_c.std(axis=0)
    std_y  = y_c.std()
    denom  = np.maximum(std_ph * std_y, 1e-12)
    corr   = cov / denom
    return (corr ** 2).astype(np.float32)

def gradient_sensitivity_fitness(Phi: np.ndarray, y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Corrected Gradient Sensitivity (Professor's Method - L1 Norm).
    Computes S_i = sum_j | dE/dw_ij |
    """
    # If beta hasn't been computed yet (Gen 1), fallback to random assignment or R2
    if beta is None:
        return np.random.uniform(0, 1, Phi.shape[1]).astype(np.float32)
    
    # 1. Compute Network Predictions
    # beta is (P+1, 1). We separate weights and bias.
    beta_w = beta[:-1, 0] # (P,)
    y_hat = Phi @ beta_w + beta[-1, 0] # (m,)
    
    # 2. Compute Global Residuals
    residuals = (y_hat - y.ravel()) # (m,)
    
    # 3. Compute derivative of Tanh: (1 - Phi^2)
    d_phi = 1.0 - (Phi ** 2) # (m, P)
    
    # 4. Chain Rule Assembly: dE/dz_ki = residual_k * beta_i * d_phi_ki
    # dE_dz shape: (m, P)
    dE_dz = residuals[:, None] * beta_w[None, :] * d_phi 
    
    # 5. Sensitivity to weights w_ij: sum_k | dE/dz_ki * x_kj |
    # Using matrix multiplication for efficient summation over m samples
    # S_i shape: (P,)
    sensitivity = np.sum(np.abs(dE_dz.T @ X), axis=1) 
    
    # Normalize to [0,1] for stability in selection
    if np.max(sensitivity) > 0:
         sensitivity = sensitivity / np.max(sensitivity)
         
    return sensitivity.astype(np.float32)

'''def r2_fitness(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    score(i) = Cor²(φᵢ, y)  for each neuron column i.

    Mathematical justification
    --------------------------
    Cor²(φᵢ, y) is the squared Pearson correlation, equal to the
    coefficient of determination R² of the simple linear regression
    y ≈ a·φᵢ + b.  It measures the fraction of variance in y that
    neuron i can explain individually — exactly the right signal for
    selecting useful feature detectors.

    Why this fixes the credit assignment problem:
        The score depends ONLY on φᵢ and y, not on β̂ or any other neuron.
        Two neurons that encode the same feature both get the same high
        score — no arbitrary splitting of credit.

    Why this handles saturation:
        A fully saturated neuron has φᵢ ≈ const for all samples
        → std(φᵢ) ≈ 0  → Cor(φᵢ, y) = 0/0 → clamped to 0.

    Parameters
    ----------
    Phi : (m, P)  activation matrix (float32 or float64)
    y   : (m,) or (m,1)

    Returns
    -------
    scores : (P,) float32 in [0, 1]
    """
    y_v   = y.ravel().astype(np.float64)
    Phi64 = Phi.astype(np.float64)

    phi_c  = Phi64 - Phi64.mean(axis=0, keepdims=True)    # (m, P)
    y_c    = y_v   - y_v.mean()                            # (m,)

    cov    = (phi_c * y_c[:, None]).mean(axis=0)           # (P,)
    std_ph = phi_c.std(axis=0)                             # (P,)
    std_y  = y_c.std()

    denom  = np.maximum(std_ph * std_y, 1e-12)
    corr   = cov / denom                                   # (P,)
    return (corr ** 2).astype(np.float32)
'''

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SELECTION — diversity-penalised greedy
# ═══════════════════════════════════════════════════════════════════════════════

def diversity_penalised_selection(
    scores : np.ndarray,    # (P,)
    Phi    : np.ndarray,    # (m, P)
    mu     : int,
    gamma  : float = 0.3,
) -> np.ndarray:
    """
    Greedy sequential selection balancing individual fitness and diversity.

    Algorithm
    ---------
    selected = []
    For t = 1 … μ:
        penalty(i) = γ · max_{j ∈ selected} |Cor(φᵢ, φⱼ)|   (0 if selected empty)
        i* = argmax_{i ∉ selected} [score(i) − penalty(i)]
        selected.append(i*)

    Rationale
    ---------
    Without diversity pressure, the top-50 by R² could all detect the
    same dominant feature.  The penalty term forces the algorithm to
    look for the NEXT most informative direction not already covered,
    building an efficient basis for the hidden layer representation.

    Parameters
    ----------
    scores : (P,)  R² per neuron
    Phi    : (m, P) activations
    mu     : number to select
    gamma  : penalty weight  (0 = pure R², 1 = full diversity)

    Returns
    -------
    selected_idx : (mu,) integer indices into the pool
    """
    P      = len(scores)
    Phi64  = Phi.astype(np.float64)

    # Normalise columns once — makes correlation = dot product
    phi_c  = Phi64 - Phi64.mean(axis=0, keepdims=True)
    norms  = np.linalg.norm(phi_c, axis=0, keepdims=True)
    norms  = np.maximum(norms, 1e-12)
    phi_n  = phi_c / norms                   # (m, P) unit-norm columns

    selected  = []
    remaining = np.ones(P, dtype=bool)
    max_sim   = np.zeros(P, dtype=np.float64)  # max |corr| with selected set

    for _ in range(mu):
        eff = scores.astype(np.float64) - gamma * max_sim
        eff[~remaining] = -np.inf

        i_star = int(np.argmax(eff))
        selected.append(i_star)
        remaining[i_star] = False

        # Update similarity of each remaining candidate to the new selection
        # |Cor(φᵢ, φ_{i*})| = |phi_n[:,i]^T phi_n[:,i*]|
        new_sims = np.abs(phi_n.T @ phi_n[:, i_star])   # (P,)
        max_sim  = np.maximum(max_sim, new_sims)

    return np.array(selected, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    fitness_strategy: str = "r2", # "r2" or "gradient"
    epochs      : int   = 100,
    mu          : int   = 50,
    n_offspring : int   = 350,
    sigma_init  : float = 0.3,
    gamma       : float = 0.3,
    data_id     : int   = 150,
    verbose     : int   = 10,
):
    X_tr, X_te, y_tr, y_te = load_dataset(data_id=data_id)
    y_flat = y_tr.ravel()
    state = MichiganESState(mu=mu, input_size=X_tr.shape[1], sigma_init=sigma_init)
    lambda_cands = list(np.logspace(-4, 4, 17))

    for epoch in range(1, epochs + 1):
        off_alpha, off_L = mutate_population(state, n_offspring=n_offspring)
        pool_alpha = np.vstack([state.alpha, off_alpha])
        pool_L     = np.vstack([state.L,     off_L    ])
        Phi_pool = compute_activations(pool_alpha, X_tr)

        # --- ABLATION TOGGLE ---
        if fitness_strategy == "r2":
            fitness = r2_fitness(Phi_pool, y_tr)
            best_idx = diversity_penalised_selection(fitness, Phi_pool, mu=mu, gamma=gamma)
        elif fitness_strategy == "gradient":
            # Pass the beta from the previous generation (padded to current pool size conceptually, 
            # or we evaluate the pool using an intermediate Ridge solve). 
            # For exactness, we must do an intermediate Ridge solve on the whole pool to get current betas.
            Phi_pool_bias = np.c_[Phi_pool, np.ones(Phi_pool.shape[0])]
            _, pool_beta, _ = compute_ridge_gcv(Phi_pool_bias, y_tr, lambda_candidates=lambda_cands)
            fitness = gradient_sensitivity_fitness(Phi_pool, y_tr, X_tr, pool_beta)
            # Cannot use correlation-diversity directly if fitness is gradient-based, standard argsort:
            best_idx = np.argsort(fitness)[::-1][:mu] 
        else:
            raise ValueError("Invalid strategy.")

        state.alpha = pool_alpha[best_idx].copy()
        state.L     = pool_L[best_idx].copy()

        Phi_surv = Phi_pool[:, best_idx]
        Phi_surv_bias = np.c_[Phi_surv, np.ones(Phi_surv.shape[0])]

        best_lam, best_beta, best_gcv = compute_ridge_gcv(Phi_surv_bias, y_tr, lambda_candidates=lambda_cands)
        state.beta = best_beta

        if epoch % verbose == 0 or epoch == 1:
            train_hat = Phi_surv_bias @ state.beta
            train_acc = accuracy_score(y_flat, np.where(train_hat.ravel() > 0, 1, -1))
            mean_sigma = state.L[:, np.arange(state.D), np.arange(state.D)].mean()
            print(f"Gen {epoch:4d} | GCV={best_gcv:.5f} | λ={best_lam:.1e} | Acc={train_acc:.4f} | σ_mean={mean_sigma:.5f}")

    return state
    
'''def train(
    epochs      : int   = 100,
    mu          : int   = 50,
    n_offspring : int   = 350,
    sigma_init  : float = 0.3,
    gamma       : float = 0.3,
    data_id     : int   = 150,
    verbose     : int   = 10,
):
    """
    Michigan-style (μ+λ)-ES training.

    Each generation
    ---------------
    1. Mutate   :  λ offspring from μ parents  →  pool P = μ+λ = 400
    2. Activate :  Φ = tanh(X·αᵀ)  for all P neurons           O(mPn)
    3. Fitness  :  score(i) = R²(φᵢ, y)  per neuron            O(mP)
    4. Select   :  diversity-penalised greedy  →  μ survivors   O(μP)
    5. H2O      :  Ridge + GCV on μ survivors (analytic)        O(mμ + μ³)
    """
    # ── Data ─────────────────────────────────────────────────────────────── #
    X_tr, X_te, y_tr, y_te = load_dataset(data_id=data_id)
    y_flat = y_tr.ravel()

    # ── ES initialisation ────────────────────────────────────────────────── #
    state = MichiganESState(mu=mu, input_size=X_tr.shape[1],
                            sigma_init=sigma_init)

    # GCV λ grid: 17 log-spaced values from 1e-4 to 1e4
    lambda_cands = list(np.logspace(-4, 4, 17))

    for epoch in range(1, epochs + 1):

        # ── 1. Mutation ───────────────────────────────────────────────────── #
        off_alpha, off_L = mutate_population(state, n_offspring=n_offspring)

        pool_alpha = np.vstack([state.alpha, off_alpha])    # (P, D)
        pool_L     = np.vstack([state.L,     off_L    ])    # (P, D, D)

        # ── 2. Activations ────────────────────────────────────────────────── #
        Phi_pool = compute_activations(pool_alpha, X_tr)    # (m, P)

        # ── 3. R² fitness (independent per neuron) ────────────────────────── #
        fitness = r2_fitness(Phi_pool, y_tr)                # (P,)

        # ── 4. Diversity-penalised selection ──────────────────────────────── #
        best_idx = diversity_penalised_selection(
            fitness, Phi_pool, mu=mu, gamma=gamma
        )

        state.alpha = pool_alpha[best_idx].copy()
        state.L     = pool_L[best_idx].copy()

        # ── 5. Analytic H2O for survivors ─────────────────────────────────── #
        Phi_surv      = Phi_pool[:, best_idx]                       # (m, mu)
        Phi_surv_bias = np.c_[Phi_surv, np.ones(Phi_surv.shape[0])]# (m, mu+1)

        best_lam, best_beta, best_gcv = compute_ridge_gcv(
            Phi_surv_bias, y_tr, lambda_candidates=lambda_cands
        )
        state.beta = best_beta   # (mu+1, 1)

        # ── Logging ───────────────────────────────────────────────────────── #
        if epoch % verbose == 0 or epoch == 1:
            train_hat = Phi_surv_bias @ state.beta          # (m, 1)
            train_acc = accuracy_score(
                y_flat, np.where(train_hat.ravel() > 0, 1, -1)
            )
            surv_r2    = fitness[best_idx]
            mean_sigma = state.L[:, np.arange(state.D),
                                    np.arange(state.D)].mean()
            print(
                f"Gen {epoch:4d}/{epochs} | "
                f"GCV={best_gcv:.5f} | λ={best_lam:.1e} | "
                f"Train Acc={train_acc:.4f} | "
                f"Mean R²={surv_r2.mean():.4f} | "
                f"σ_mean={mean_sigma:.5f}"
            )

    # ── Final test evaluation ─────────────────────────────────────────────── #
    Phi_test      = compute_activations(state.alpha, X_te)
    Phi_test_bias = np.c_[Phi_test, np.ones(Phi_test.shape[0])]
    test_hat      = Phi_test_bias @ state.beta
    test_acc      = accuracy_score(
        y_te.ravel(), np.where(test_hat.ravel() > 0, 1, -1)
    )
    print(f"\n{'='*55}")
    print(f"  Final Test Accuracy : {test_acc:.4f}")
    print(f"{'='*55}")
    return state

'''
# ═══════════════════════════════════════════════════════════════════════════════
# 8.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    trained = train(
        epochs      = 100,
        mu          = 50,
        n_offspring = 350,
        sigma_init  = 0.3,
        gamma       = 0.3,
        data_id     = 150,
        verbose     = 10,
    )