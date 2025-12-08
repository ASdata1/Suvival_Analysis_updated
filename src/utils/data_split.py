import numpy as np
import pandas as pd
from typing import Tuple


class DataSplitter:
    """
    Flexible time-to-event dataset splitting utility
    supporting:
      - random split (baseline)
      - censoring-aware split
      - time-based split (early vs late follow-up)
      - stratified censoring split
      - feature-shift split for informative censoring
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ----------------------------------------------------
    # 1. RANDOM SPLIT (baseline)
    # ----------------------------------------------------
    def random_split(
        self, df: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        df_shuffled = df.sample(frac=1, random_state=self.rng.integers(1e6))

        n = len(df)
        n_train = int(train_size * n)
        n_val = int(val_size * n)

        train = df_shuffled.iloc[:n_train]
        val = df_shuffled.iloc[n_train:n_train + n_val]
        test = df_shuffled.iloc[n_train + n_val:]

        return train, val, test

    # ----------------------------------------------------
    # 2. CENSORING-AWARE SPLIT
    # ensures similar censoring proportions
    # ----------------------------------------------------
    def censoring_aware_split(
        self, df: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15
    ):
        censored = df[df["event"] == 0]
        events = df[df["event"] == 1]

        censored = censored.sample(frac=1, random_state=self.rng.integers(1e6))
        events = events.sample(frac=1, random_state=self.rng.integers(1e6))

        def alloc(group):
            n = len(group)
            n_train = int(train_size * n)
            n_val = int(val_size * n)
            return (
                group.iloc[:n_train],
                group.iloc[n_train:n_train + n_val],
                group.iloc[n_train + n_val:]
            )

        c_train, c_val, c_test = alloc(censored)
        e_train, e_val, e_test = alloc(events)

        train = pd.concat([c_train, e_train]).sample(frac=1)
        val = pd.concat([c_val, e_val]).sample(frac=1)
        test = pd.concat([c_test, e_test]).sample(frac=1)

        return train, val, test

    # ----------------------------------------------------
    # 3. TIME-BASED SPLIT (early vs late follow-up)
    # simulates real-world pension modelling
    # ----------------------------------------------------
    def time_based_split(
        self, df: pd.DataFrame, cutoff: float = 10.0
    ):
        """
        Train on early follow-up, test on late follow-up.

        cutoff = max follow-up years included in training
        """

        train = df[df["time"] <= cutoff]
        test = df[df["time"] > cutoff]

        # small validation from train
        val = train.sample(frac=0.2, random_state=self.rng.integers(1e6))
        train = train.drop(val.index)

        return train, val, test

    # ----------------------------------------------------
    # 4. FEATURE SHIFT SPLIT
    # key for informative censoring scenario
    # ----------------------------------------------------
    def feature_shift_split(
        self, df: pd.DataFrame, quantile: float = 0.7
    ):
        """
        Train on lower-risk population,
        test on higher-risk population

        Creates distribution shift:
          - older
          - higher salary
          - higher risk_score
        """

        threshold = df["risk_score"].quantile(quantile)

        train = df[df["risk_score"] <= threshold]
        test = df[df["risk_score"] > threshold]

        val = train.sample(frac=0.2, random_state=self.rng.integers(1e6))
        train = train.drop(val.index)

        return train, val, test
