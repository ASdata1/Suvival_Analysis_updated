# ======================================
# src/data_generation/generate_censoring_times.py
# ======================================

import numpy as np
import pandas as pd


def generate_random_censoring(
    n: int,
    min_follow_up: float = 0.0,
    max_follow_up: float = 20.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Scenario 1: Non-informative random censoring.

    Censoring times are drawn independently of features and event times.
    E.g. administrative censoring at random times up to max_follow_up.

    Parameters
    ----------
    n : int
        Number of individuals.
    min_follow_up : float
        Minimum possible censoring time.
    max_follow_up : float
        Maximum possible censoring time (e.g. study end).
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Censoring times C.
    """
    rng = np.random.default_rng(seed)
    c = rng.uniform(min_follow_up, max_follow_up, size=n)
    return c


def generate_covariate_censoring(
    features: pd.DataFrame,
    gamma: dict[str, float] | None = None,
    shape: float = 1.2,
    baseline_rate: float = 0.03,
    seed: int | None = None,
) -> np.ndarray:
    """
    Scenario 2: Covariate-dependent censoring.

    Censoring is independent of T given X, but depends on features like age,
    salary, and risk_score. This mimics:
      - lower income or younger members being more likely to be lost to follow-up,
      - high-risk members being followed up more carefully, etc.

    We use the same Weibull inversion idea as for the event times.

    Parameters
    ----------
    features : pd.DataFrame
        Must contain at least: "age", "salary", "risk_score".
    gamma : dict[str, float] | None
        Coefficients mapping for censoring hazard.
    shape : float
        Weibull shape parameter (>1 ⇒ increasing hazard of censoring over time).
    baseline_rate : float
        Baseline censoring rate.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Censoring times C.
    """
    rng = np.random.default_rng(seed)

    if gamma is None:
        # Example logic:
        # - Younger members → more mobile → more likely censored earlier (positive coef on -age).
        # - Lower salary → more likely to change jobs (censored earlier).
        # - Higher risk_score → more engaged / monitored, so slightly less likely to be censored.
        gamma = {
            "age": -0.02,        # older => less censoring
            "salary": -0.000004, # higher salary => less censoring
            "risk_score": -0.2,  # higher risk => less censoring
        }

    eta_c = np.zeros(len(features))
    for col, coef in gamma.items():
        if col not in features.columns:
            continue
        eta_c += coef * features[col].to_numpy()

    u = rng.uniform(0.0, 1.0, size=len(features))
    u = np.clip(u, 1e-10, 1 - 1e-10)

    censor_times = (-np.log(u) / (baseline_rate * np.exp(eta_c))) ** (1.0 / shape)
    return censor_times


def generate_informative_censoring(
    event_times: np.ndarray,
    features: pd.DataFrame,
    delta: dict[str, float] | None = None,
    proportion_pre_event: float = 0.6,
    noise_sd: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """
    Scenario 3: Informative censoring depending directly on the (unobserved) event time.

    Idea:
    - For a proportion of individuals, censoring happens *before* the event and
      is earlier when the true event time is earlier.
    - This mimics, for example, high-risk people dropping out or dying from other
      causes just before claiming their pension.

    This violates the standard assumption C ⟂ T | X, so even IPCW will struggle
    to fully remove bias.

    Construction:
    - With probability p = proportion_pre_event, set:
        C = T * exp(η) * exp(ε)
      where η depends on some features and ε ~ Normal(0, noise_sd).
      We cap C to be strictly less than T (pre-event censoring).
    - Otherwise, censor after event or far into the future, so they are mostly observed.

    Parameters
    ----------
    event_times : np.ndarray
        True event times T.
    features : pd.DataFrame
        Features used to optionally shape the censoring.
    delta : dict[str, float] | None
        Coefficients for η in the pre-event censoring part.
    proportion_pre_event : float
        Proportion of individuals censored before event time, 0 < p < 1.
    noise_sd : float
        Standard deviation of log-normal noise for censoring times.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Censoring times C.
    """
    rng = np.random.default_rng(seed)
    n = len(event_times)

    if delta is None:
        # Example: higher risk_score ⇒ more likely to be censored earlier
        # relative to their event time.
        delta = {
            "risk_score": -0.3,
            "age": 0.01,  # older => slightly later censoring relative to T
        }

    eta = np.zeros(n)
    for col, coef in delta.items():
        if col not in features.columns:
            continue
        eta += coef * features[col].to_numpy()

    # Decide who gets pre-event informative censoring
    pre_event_mask = rng.uniform(0.0, 1.0, size=n) < proportion_pre_event

    censor_times = np.empty(n, dtype=float)

    # Pre-event censoring: directly tied to event_times
    # C = T * exp(η + ε), forced to be < T
    eps = rng.normal(loc=0.0, scale=noise_sd, size=n)
    c_pre = event_times * np.exp(eta + eps)

    # Cap pre-event censoring to be at most 90% of event time (to ensure it's pre-event)
    c_pre = np.minimum(c_pre, 0.9 * event_times)
    c_pre = np.clip(c_pre, 0.01, None)  # positive times

    censor_times[pre_event_mask] = c_pre[pre_event_mask]

    # Non-pre-event censoring: mostly administrative and late
    # e.g. censored at some time after the event or far into follow-up
    max_extra = np.quantile(event_times, 0.9)  # upper attempt at follow-up
    c_post = event_times + rng.uniform(0.1, max_extra, size=n)
    censor_times[~pre_event_mask] = c_post[~pre_event_mask]

    return censor_times
