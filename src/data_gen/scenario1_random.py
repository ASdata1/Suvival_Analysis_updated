# ================================
# src/data_generation/scenario1_random.py
# ================================

import numpy as np
import pandas as pd

from .gen_features import generate_features


from .gen_event_times import generate_event_times_weibull
from .gen_censoring_times import generate_random_censoring


def simulate_scenario1(
    n: int = 2000,
    seed: int | None = 42,
    min_follow_up: float = 0.0,
    max_follow_up: float = 20.0,
) -> pd.DataFrame:
    """
    Scenario 1: Non-informative random censoring.

    - Event times depend on pension-related features via a Weibull model.
    - Censoring times are random and independent of both T and X.

    Returns a DataFrame with:
      ["age", "salary", "tenure", "gender", "risk_score",
       "true_event_time", "censor_time", "time", "event"]
    """
    # We use a master seed and split it logically for reproducibility
    rng = np.random.default_rng(seed)
    feature_seed = rng.integers(0, 1_000_000)
    event_seed = rng.integers(0, 1_000_000)
    censor_seed = rng.integers(0, 1_000_000)

    features = generate_features(n=n, seed=feature_seed)
    event_times = generate_event_times_weibull(
        features, shape=1.5, baseline_rate=0.02, seed=event_seed
    )
    censor_times = generate_random_censoring(
        n=n,
        min_follow_up=min_follow_up,
        max_follow_up=max_follow_up,
        seed=censor_seed,
    )

    observed_time = np.minimum(event_times, censor_times)
    event_observed = (event_times <= censor_times).astype(int)

    df = features.copy()
    df["true_event_time"] = event_times
    df["censor_time"] = censor_times
    df["time"] = observed_time
    df["event"] = event_observed
    df["scenario"] = "random_censoring"

    TRUE_PARAMS = {
    "beta": [0.03, -0.02, 0.05],   # Age, Income, Risk
    "weibull_shape": 1.5,
    "weibull_scale": 20.0,
    "censoring_rate": 0.3,
    "censoring_type": "random"
}
    return df, TRUE_PARAMS  

