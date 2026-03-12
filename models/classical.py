"""Modèles de secours sans dépendances deep learning lourdes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class ClassicalModelWrapper:
    model: LogisticRegression

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def build_logistic_baseline() -> ClassicalModelWrapper:
    return ClassicalModelWrapper(model=LogisticRegression(max_iter=500))
