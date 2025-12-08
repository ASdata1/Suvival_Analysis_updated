# ================================
# src/data_generation/scenario3_informative.py
# ================================

import numpy as np
import pandas as pd

from .gen_features import generate_features
from .gen_event_times import generate_event_times_weibull
from .gen_censoring_times import generate_informative_censoring


def simulate_scenario3(
    n: int = 2000,
    seed: int | None = 999,
    proportion_pre_event: float = 0.6,
) -> pd.DataFrame:
    """
    Scenario 3: Informative censoring depending on the true event time.

    - Event times depend on pension features as before.
    - For a proportion of members, censoring happens before the event and
      is directly tied to the event time (shorter T ⇒ earlier censoring).
    - This breaks the C ⟂ T | X assumption, so even IPCW will not fully
      remove bias.

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

    censor_times = generate_informative_censoring(
        event_times=event_times,
        features=features,
        proportion_pre_event=proportion_pre_event,
        seed=censor_seed,
    )

    observed_time = np.minimum(event_times, censor_times)
    event_observed = (event_times <= censor_times).astype(int)

    df = features.copy()
    df["true_event_time"] = event_times
    df["censor_time"] = censor_times
    df["time"] = observed_time
    df["event"] = event_observed
    df["scenario"] = "informative_censoring"

    TRUE_PARAMS = {
        "beta": [0.05, -0.01, 0.1],
        "weibull_shape": 1.3,
        "weibull_scale": 17.0,
        "informative_censor_strength": 0.5,
        "censoring_type": "informative"
    }
    return df, TRUE_PARAMS    