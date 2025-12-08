# src/models/ML_classifiers.py
import pandas as pd
from typing import Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


class MLClassifiers:
    def __init__(self, horizon: float = 10.0):
        """
        horizon = prediction window e.g. 'event within 10 years?'
        """
        self.horizon = horizon
        self.models = {
            "logistic": LogisticRegression(max_iter=500),
            "rf": RandomForestClassifier(n_estimators=300)
        }

    def _make_binary_label(self, df):
        """Create binary label: event before horizon."""
        return ((df["time"] <= self.horizon) & (df["event"] == 1)).astype(int)

    def fit(
        self,
        train_df: pd.DataFrame,
        weights: Optional[pd.Series] = None,
    ):
        X = train_df.drop(["time", "event"], axis=1)
        y = self._make_binary_label(train_df)

        for name, model in self.models.items():
            if weights is not None:
                model.fit(X, y, sample_weight=weights)
            else:
                model.fit(X, y)

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        X = test_df.drop(["time", "event"], axis=1)
        y = self._make_binary_label(test_df)

        metrics = {}
        for name, model in self.models.items():
            preds = model.predict_proba(X)[:, 1]
            metrics[name] = {
                "AUC": roc_auc_score(y, preds),
                "Accuracy": accuracy_score(y, preds > 0.5),
                "F1": f1_score(y, preds > 0.5),
            }
        return metrics
