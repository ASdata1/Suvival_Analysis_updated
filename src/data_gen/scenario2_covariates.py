# ================================
# src/data_generation/scenario2_covariates.py
# ================================

import numpy as np
import pandas as pd

from .gen_features import generate_features
from .gen_event_times import generate_event_times_weibull
from .gen_censoring_times import generate_covariate_censoring


def simulate_scenario2(
    n: int = 2000,
    seed: int | None = 123,
) -> pd.DataFrame:
    """
    Scenario 2: Covariate-dependent censoring.

    - Event times depend on features (age, salary, tenure, risk_score).
    - Censoring also depends on features (e.g. younger / lower-income
      members more likely to be lost to follow-up).

    This satisfies C ⟂ T | X (independent censoring given covariates),
    so IPCW can be effective if the censoring model uses the same X.

    Returns DataFrame with:
      features + ["true_event_time", "censor_time", "time", "event"].
    """
    rng = np.random.default_rng(seed)
    feature_seed = rng.integers(0, 1_000_000)
    event_seed = rng.integers(0, 1_000_000)
    censor_seed = rng.integers(0, 1_000_000)

    features = generate_features(n=n, seed=feature_seed)

    event_times = generate_event_times_weibull(
        features,
        shape=1.5,
        baseline_rate=0.02,
        seed=event_seed,
    )

    censor_times = generate_covariate_censoring(
        features,
        shape=1.2,
        baseline_rate=0.03,
        seed=censor_seed,
    )

    observed_time = np.minimum(event_times, censor_times)
    event_observed = (event_times <= censor_times).astype(int)

    df = features.copy()
    df["true_event_time"] = event_times
    df["censor_time"] = censor_times
    df["time"] = observed_time
    df["event"] = event_observed
    df["scenario"] = "covariate_censoring"

    TRUE_PARAMS = {
    "beta": [0.04, -0.03, 0.07],
    "weibull_shape": 1.4,
    "weibull_scale": 18.0,
    "censoring_model_gamma": [0.02, -0.05, 0.01],
    "censoring_type": "covariate-dependent"
}


    return df, TRUE_PARAMS  