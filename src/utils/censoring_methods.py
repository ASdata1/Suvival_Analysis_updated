"""
Censoring handling methods for survival datasets.

Provides:
- zero_censoring
- discard_censored
- ipcw_km
- ipcw_model_based

All functions return:
    df_out, weights

where:
    df_out = modified dataset
    weights = weights to apply in weighted models
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LogisticRegression


def zero_censoring(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Treat censored observations as if the event happened at the censoring time.
    Equivalent to setting event = 1 for all.

    BAD method (biases estimates) – included for comparison.
    """
    df_out = df.copy()
    df_out["event"] = 1
    weights = np.ones(len(df_out))
    return df_out, weights


def discard_censored(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Complete case analysis:
    Remove all censored individuals.
    """
    df_out = df[df["event"] == 1].copy()
    weights = np.ones(len(df_out))
    return df_out, weights


def ipcw_km(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    IPCW using Kaplan–Meier estimator of censoring survival function:
        w_i = 1 / G(t_i)

    where G(t) = P(C > t)

    Works when censoring is random or depends weakly on covariates.
    """
    df_out = df.copy()
    km = KaplanMeierFitter()
    km.fit(
        durations=df_out["time"],
        event_observed=1 - df_out["event"],  # censoring indicator
    )

    # survival of censoring
    G_t = km.survival_function_at_times(df_out["time"]).values
    weights = 1.0 / np.clip(G_t, 1e-6, None)
    return df_out, weights


def ipcw_model_based(df: pd.DataFrame, covariates=None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    IPCW using a model for censoring probability:
        fit P(C > t | X)

    Typically logistic regression predicting censorship.

    Best for:
        - covariate-dependent censoring
        - informative censoring conditional on X
    """

    df_out = df.copy()

    if covariates is None:
        covariates = ["age", "salary", "tenure", "risk_score"]

    X = df_out[covariates]
    y = 1 - df_out["event"]  # 1 = censored

    model = LogisticRegression(max_iter=500)
    model.fit(X, y)

    censor_prob = model.predict_proba(X)[:, 1]  # P(censored)
    survival_prob = 1 - censor_prob

    weights = 1.0 / np.clip(survival_prob, 1e-6, None)

    return df_out, weights
