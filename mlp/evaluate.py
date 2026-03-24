"""
Data generation and evaluation utilities for A/B Test experiments.

Ground truth: A linear model over binary factors with additive noise.
PCS (Probability of Correct Selection): fraction of runs where the
model correctly identifies the best factor combination.
"""

import numpy as np
from itertools import product
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Ground-truth response surface
# ---------------------------------------------------------------------------

def make_ground_truth(n_factors: int, seed: int = 0) -> np.ndarray:
    """
    Generate fixed true coefficients for the linear response model.
        y = X @ beta + noise
    
    Returns beta of shape (n_factors,).
    """
    rng = np.random.default_rng(seed)
    beta = rng.uniform(-1, 1, size=n_factors)
    return beta


def all_factor_combinations(n_factors: int) -> np.ndarray:
    """Return all 2^n binary factor combinations, shape (2^n, n_factors)."""
    return np.array(list(product([0, 1], repeat=n_factors)), dtype=float)


def true_responses(combinations: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute noiseless responses for every combination."""
    return combinations @ beta


def true_best_combination(combinations: np.ndarray, beta: np.ndarray) -> int:
    """Return index of the combination with the highest true response."""
    return int(np.argmax(true_responses(combinations, beta)))


# ---------------------------------------------------------------------------
# Training data sampling
# ---------------------------------------------------------------------------

def sample_data(n_samples: int, n_factors: int, beta: np.ndarray,
                noise_std: float = 0.5, seed: Optional[int] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw random binary factor combinations and noisy responses.
    
    Returns
    -------
    X : (n_samples, n_factors) binary array
    y : (n_samples,) noisy response
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_samples, n_factors)).astype(float)
    noise = rng.normal(0, noise_std, size=n_samples)
    y = X @ beta + noise
    return X, y


# ---------------------------------------------------------------------------
# PCS evaluation
# ---------------------------------------------------------------------------

def compute_pcs(model_class, model_kwargs: dict, n_factors: int,
                beta: np.ndarray, sample_size: int,
                n_reps: int = 100, noise_std: float = 0.5,
                fit_kwargs: Optional[dict] = None,
                base_seed: int = 42) -> float:
    """
    Estimate PCS for a given model and sample size.

    For each repetition:
      1. Sample training data.
      2. Train the model from scratch.
      3. Predict the best combination over all 2^n options.
      4. Check if it matches the true best.

    Parameters
    ----------
    model_class   : class with .fit() and .predict_best_combination()
    model_kwargs  : constructor arguments (depth, width, etc.)
    n_factors     : number of binary factors
    beta          : ground-truth coefficients
    sample_size   : number of training observations per repetition
    n_reps        : number of independent repetitions
    noise_std     : observation noise standard deviation
    fit_kwargs    : extra keyword args passed to .fit()
    base_seed     : seed offset for reproducibility

    Returns
    -------
    pcs : float in [0, 1]
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    combinations = all_factor_combinations(n_factors)
    best_idx = true_best_combination(combinations, beta)
    correct = 0

    for rep in range(n_reps):
        seed = base_seed + rep
        X, y = sample_data(sample_size, n_factors, beta,
                           noise_std=noise_std, seed=seed)
        model = model_class(**model_kwargs, seed=seed)
        model.fit(X, y, **fit_kwargs)
        predicted_best = model.predict_best_combination(combinations)
        if predicted_best == best_idx:
            correct += 1

    return correct / n_reps


# ---------------------------------------------------------------------------
# Iterative (expanding) dataset experiment
# ---------------------------------------------------------------------------

def compute_pcs_iterative(model_class, model_kwargs: dict, n_factors: int,
                          beta: np.ndarray, max_samples: int,
                          batch_size: int = 100, n_reps: int = 100,
                          noise_std: float = 0.5,
                          fit_kwargs: Optional[dict] = None,
                          retrain_from_scratch: bool = True,
                          base_seed: int = 42
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCS vs sample size using an *iteratively expanding* dataset.

    Key fix vs. original MATLAB code: the model is retrained from scratch
    on the full accumulated dataset at each step (retrain_from_scratch=True).
    Setting it to False reproduces the bug where PCS plateaus at ~0.5.

    Returns
    -------
    sizes : 1-D array of cumulative sample sizes
    pcs   : 1-D array of PCS estimates, one per size checkpoint
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    combinations = all_factor_combinations(n_factors)
    best_idx = true_best_combination(combinations, beta)
    sizes = np.arange(batch_size, max_samples + 1, batch_size)
    pcs_values = []

    for size in sizes:
        correct = 0
        for rep in range(n_reps):
            rng = np.random.default_rng(base_seed + rep)
            # Accumulate data batch by batch
            X_acc, y_acc = np.empty((0, n_factors)), np.empty(0)
            for batch_start in range(0, size, batch_size):
                n_new = min(batch_size, size - batch_start)
                X_new = rng.integers(0, 2, size=(n_new, n_factors)).astype(float)
                y_new = X_new @ beta + rng.normal(0, noise_std, n_new)
                X_acc = np.vstack([X_acc, X_new])
                y_acc = np.concatenate([y_acc, y_new])

            if retrain_from_scratch:
                # FIX: always train a fresh model on the full dataset
                model = model_class(**model_kwargs, seed=base_seed + rep)
                model.fit(X_acc, y_acc, **fit_kwargs)
            else:
                # BUG reproduction: train only on the last batch
                model = model_class(**model_kwargs, seed=base_seed + rep)
                model.fit(X_new, y_new, **fit_kwargs)

            if model.predict_best_combination(combinations) == best_idx:
                correct += 1

        pcs_values.append(correct / n_reps)

    return sizes, np.array(pcs_values)
