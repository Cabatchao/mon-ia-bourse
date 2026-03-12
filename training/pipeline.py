"""Pipeline d'entraînement avec validation temporelle et walk-forward."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import TRAINING
from models.deep_models import HybridLSTMTransformerMLP


@dataclass
class TrainingResult:
    fold_metrics: list[dict[str, float]]
    final_model: nn.Module


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

    def _fit_one_fold(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> None:
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
    def _evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X))
            probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "log_loss": float(log_loss(y, np.clip(probs, 1e-6, 1 - 1e-6))),
        }

    def walk_forward_train(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> TrainingResult:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics: list[dict[str, float]] = []
        final_model: nn.Module | None = None

        for train_idx, test_idx in tscv.split(X):
            model = HybridLSTMTransformerMLP(n_features=X.shape[-1])
            self._fit_one_fold(model, X[train_idx], y[train_idx])
            metrics.append(self._evaluate(model, X[test_idx], y[test_idx]))
            final_model = model

        if final_model is None:
            raise ValueError("Aucun fold entraîné.")
        return TrainingResult(fold_metrics=metrics, final_model=final_model)
