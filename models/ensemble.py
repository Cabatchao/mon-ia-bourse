"""Méthodes d'ensemble: averaging, bagging, stacking."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier


class EnsemblePredictor:
    def __init__(self) -> None:
        self.meta_model = LogisticRegression(max_iter=500)
        self.bagging_model = BaggingClassifier(estimator=LogisticRegression(max_iter=300), n_estimators=20, random_state=42)

    @staticmethod
    def model_averaging(pred_matrix: np.ndarray) -> np.ndarray:
        return pred_matrix.mean(axis=1)

    def fit_stacking(self, base_predictions: np.ndarray, y_true: np.ndarray) -> None:
        self.meta_model.fit(base_predictions, y_true)

    def predict_stacking(self, base_predictions: np.ndarray) -> np.ndarray:
        return self.meta_model.predict_proba(base_predictions)[:, 1]

    def fit_bagging(self, X: np.ndarray, y: np.ndarray) -> None:
        self.bagging_model.fit(X, y)

    def predict_bagging(self, X: np.ndarray) -> np.ndarray:
        return self.bagging_model.predict_proba(X)[:, 1]
