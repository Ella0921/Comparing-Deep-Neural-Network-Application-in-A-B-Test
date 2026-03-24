"""
Decentralised A/B Testing via Partial Factorial Design (Orthogonal Arrays).

Implements the three design levels described in the paper:
  - Full factorial   (all 2^n combinations)
  - Half-fraction    (Resolution V design)
  - Partial factorial (screening / saturated design — as few as 16 runs for 8 factors)

PCS is evaluated by fitting a linear main-effects model on the OA runs,
then predicting the best combination over all 2^n options.
"""

import numpy as np
from itertools import product
from typing import Tuple


# ---------------------------------------------------------------------------
# Orthogonal Array helpers
# ---------------------------------------------------------------------------

def _hadamard_matrix(n: int) -> np.ndarray:
    """
    Recursively build a Hadamard matrix of order n (must be power of 2).
    Used to construct balanced OAs.
    """
    if n == 1:
        return np.array([[1]])
    H_half = _hadamard_matrix(n // 2)
    return np.block([[H_half, H_half], [H_half, -H_half]])


def get_oa_design(n_factors: int, fraction: str = "partial") -> np.ndarray:
    """
    Return an Orthogonal Array design matrix with entries in {0, 1}.

    fraction : "full" | "half" | "partial"
        full    → 2^n runs (complete factorial)
        half    → 2^(n-1) runs
        partial → smallest OA that accommodates n_factors (saturated design)

    Returns
    -------
    design : (n_runs, n_factors) array with binary entries
    """
    if fraction == "full":
        runs = 2 ** n_factors
        design = np.array(list(product([0, 1], repeat=n_factors)), dtype=float)
        return design

    # Smallest power-of-2 >= n_factors + 1  (saturated OA)
    if fraction == "partial":
        n_cols_needed = n_factors
        p = 1
        while (2 ** p - 1) < n_cols_needed:
            p += 1
        n_runs = 2 ** p

    elif fraction == "half":
        n_runs = 2 ** (n_factors - 1)
    else:
        raise ValueError(f"Unknown fraction '{fraction}'")

    # Build Hadamard-based OA
    H = _hadamard_matrix(n_runs)
    # Convert ±1 → 0/1
    oa_full = ((H[:, 1:] + 1) / 2).astype(int)   # drop intercept column

    # Select first n_factors columns
    if oa_full.shape[1] < n_factors:
        raise ValueError(
            f"OA with {n_runs} runs can only accommodate "
            f"{oa_full.shape[1]} factors, but {n_factors} requested."
        )
    return oa_full[:, :n_factors].astype(float)


# ---------------------------------------------------------------------------
# Linear main-effects estimator
# ---------------------------------------------------------------------------

class LinearFactorialModel:
    """
    Simple OLS estimator for main effects only.
    y = beta_0 + X @ beta + noise
    
    Prediction over all 2^n combinations allows identification of the
    best factor combination without running every experiment.
    """

    def __init__(self, **kwargs):
        # Accept (and ignore) extra kwargs so API matches MLP
        self.coef_ = None
        self.intercept_ = None
        self._seed = kwargs.get("seed", None)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "LinearFactorialModel":
        """OLS fit on design matrix X."""
        n = X.shape[0]
        X_aug = np.hstack([np.ones((n, 1)), X])   # add intercept
        # Normal equations
        self.coef_full_ = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        self.intercept_ = self.coef_full_[0]
        self.coef_ = self.coef_full_[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def predict_best_combination(self, all_combinations: np.ndarray) -> int:
        preds = self.predict(all_combinations)
        return int(np.argmax(preds))


# ---------------------------------------------------------------------------
# OA-based PCS evaluation
# ---------------------------------------------------------------------------

def compute_pcs_factorial(design_fraction: str, n_factors: int,
                          beta: np.ndarray, n_reps: int = 1000,
                          noise_std: float = 0.5,
                          base_seed: int = 42) -> Tuple[float, int]:
    """
    Estimate PCS for the linear factorial model using the specified OA design.

    Returns
    -------
    pcs      : float — Probability of Correct Selection
    n_runs   : int  — number of experimental runs used
    """
    from mlp.evaluate import all_factor_combinations, true_best_combination

    design = get_oa_design(n_factors, fraction=design_fraction)
    n_runs = design.shape[0]
    combinations = all_factor_combinations(n_factors)
    best_idx = true_best_combination(combinations, beta)

    correct = 0
    for rep in range(n_reps):
        rng = np.random.default_rng(base_seed + rep)
        y = design @ beta + rng.normal(0, noise_std, size=n_runs)
        model = LinearFactorialModel(seed=base_seed + rep)
        model.fit(design, y)
        if model.predict_best_combination(combinations) == best_idx:
            correct += 1

    return correct / n_reps, n_runs
