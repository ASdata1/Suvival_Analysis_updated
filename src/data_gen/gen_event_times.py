# ==================================
# src/data_generation/generate_event_times.py
# ==================================

import numpy as np
import pandas as pd


def generate_event_times_weibull(
    features: pd.DataFrame,
    beta: dict[str, float] | None = None,
    shape: float = 1.5,
    baseline_rate: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate time-to-event (e.g. time from study entry to pension claim)
    from a Weibull AFT / PH-style model.

    Hazard(t | X) = λ0 * k * (λ0 * t)^(k-1) * exp(βᵀX)
    where:
      - k = shape
      - λ0 = baseline_rate
      - X are features, β are coefficients.

    Inversion formula (for a Weibull PH model):
      U ~ Uniform(0, 1)
      T = [ -log(U) / (λ0 * exp(η)) ]^(1 / k)
      where η = βᵀX

    Parameters
    ----------
    features : pd.DataFrame
        Must contain at least: "age", "salary", "tenure", "risk_score".
    beta : dict[str, float] | None
        Coefficient mapping: feature_name -> coefficient.
        If None, a reasonable default for pensions claim narrative is used.
    shape : float
        Weibull shape parameter k (>1 ⇒ increasing hazard with time).
    baseline_rate : float
        Baseline rate λ0 for the Weibull model.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Event times T (same length as features).
    """
    rng = np.random.default_rng(seed) # random number generator

    if beta is None:
        # Default: older age, longer tenure, higher risk_score → earlier claim.
        # Higher salary → slightly later claim (can afford to delay).
        beta = {
            "age": 0.03,
            "salary": -0.000005,  # scaled by salary units
            "tenure": -0.01,      # longer tenure → maybe more secure, delay a bit
            "risk_score": 0.25,
        }

    # Construct linear predictor 
    eta = np.zeros(len(features))
    for col, coef in beta.items():
        if col not in features.columns:
            continue
        eta += coef * features[col].to_numpy()

    # Draw uniform U for inversion
    u = rng.uniform(0.0, 1.0, size=len(features))

    # Avoid numerical issues
    u = np.clip(u, 1e-10, 1 - 1e-10)

    # Inverse CDF for Weibull PH-style model
    # T = [ -log(U) / (λ0 * exp(η)) ]^(1/k)
    event_times = (-np.log(u) / (baseline_rate * np.exp(eta))) ** (1.0 / shape)

    return event_times
