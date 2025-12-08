# src/models/aft_model.py
import pandas as pd
from lifelines import WeibullAFTFitter
from typing import Optional


class AFTModel:
    def __init__(self):
        self.model = WeibullAFTFitter()

    def fit(
        self,
        train_df: pd.DataFrame,
        duration_col: str = "time",
        event_col: str = "event",
        weights: Optional[pd.Series] = None,
    ):
        df = train_df.copy()
        if weights is not None:
            df["weights"] = weights
            self.model.fit(
                df,
                duration_col=duration_col,
                event_col=event_col,
                weights_col="weights"
            )
        else:
            self.model.fit(df, duration_col=duration_col, event_col=event_col)

    def predict_median(self, df: pd.DataFrame):
        return self.model.predict_median(df)

    def score(self, test_df: pd.DataFrame):
        return self.model.concordance_index_
