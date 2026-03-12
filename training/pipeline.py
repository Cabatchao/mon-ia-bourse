"""Pipeline d'entraînement avec validation temporelle et walk-forward."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from config.settings import TRAINING
from models.classical import build_logistic_baseline

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from models.deep_models import HybridLSTMTransformerMLP

    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    TORCH_AVAILABLE = False
    torch = None
    nn = object  # type: ignore


@dataclass
class TrainingResult:
    fold_metrics: list[dict[str, float]]
    final_model: object
    backend: str


class TemporalTrainer:
    def __init__(self, lookback: int = TRAINING.lookback) -> None:
        self.lookback = lookback

    def build_windows(self, df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        arr = df[feature_cols].values.astype(np.float32)
        target = df["target"].values.astype(np.float32)
        for i in range(self.lookback, len(df)):
            X.append(arr[i - self.lookback : i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def _fit_one_fold_torch(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> None:
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=TRAINING.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(TRAINING.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

    @staticmethod
    def _eval_torch(model: nn.Module, X: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X))
            probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "log_loss": float(log_loss(y, np.clip(probs, 1e-6, 1 - 1e-6))),
        }, probs

    @staticmethod
    def _eval_sklearn(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = (probs >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "log_loss": float(log_loss(y, np.clip(probs, 1e-6, 1 - 1e-6))),
        }

    def walk_forward_train(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> TrainingResult:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics: list[dict[str, float]] = []
        final_model: object | None = None

        for train_idx, test_idx in tscv.split(X):
            if TORCH_AVAILABLE:
                model = HybridLSTMTransformerMLP(n_features=X.shape[-1])
                self._fit_one_fold_torch(model, X[train_idx], y[train_idx])
                fold_metrics, _ = self._eval_torch(model, X[test_idx], y[test_idx])
                final_model = model
            else:
                model = build_logistic_baseline()
                X_train = X[train_idx][:, -1, :]
                X_test = X[test_idx][:, -1, :]
                model.fit(X_train, y[train_idx])
                probs = model.predict_proba(X_test)
                fold_metrics = self._eval_sklearn(probs, y[test_idx])
                final_model = model
            metrics.append(fold_metrics)

        if final_model is None:
            raise ValueError("Aucun fold entraîné.")
        backend = "torch" if TORCH_AVAILABLE else "sklearn_fallback"
        return TrainingResult(fold_metrics=metrics, final_model=final_model, backend=backend)
