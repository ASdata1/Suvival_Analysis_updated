# src/models/cox_model.py
import pandas as pd
from lifelines import CoxPHFitter
from typing import Optional, Dict


class CoxModel:
    def __init__(self):
        self.model = CoxPHFitter()

    def fit(
        self,
        train_df: pd.DataFrame,
        duration_col: str = "time",
        event_col: str = "event",
        weights: Optional[pd.Series] = None,
    ):
        """Fit Cox model with optional IPCW weights."""
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

    def predict_risk(self, df: pd.DataFrame):
        """Return partial hazard predictions."""
        return self.model.predict_partial_hazard(df)

    def score(self, test_df: pd.DataFrame):
        """Return concordance."""
        return self.model.concordance_index_
