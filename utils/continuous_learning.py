"""Boucle d'apprentissage continu: journalisation, analyse d'erreurs, ré-entrainement."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class LearningStats:
    brier_score: float
    directional_accuracy: float
    calibration_gap: float


class ContinuousLearningManager:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append_predictions(self, predictions: pd.DataFrame) -> None:
        if self.log_path.exists():
            hist = pd.read_parquet(self.log_path)
            merged = pd.concat([hist, predictions], ignore_index=True)
        else:
            merged = predictions.copy()
        merged.to_parquet(self.log_path, index=False)

    def evaluate(self, realized: pd.DataFrame) -> LearningStats:
        data = pd.read_parquet(self.log_path).merge(realized, on=["asset", "horizon", "timestamp"], how="inner")
        err = data["prob_up"] - data["target_realized"]
        brier = float((err ** 2).mean())
        pred_dir = (data["prob_up"] >= 0.5).astype(int)
        acc = float((pred_dir == data["target_realized"]).mean())
        calib = float(abs(data["prob_up"].mean() - data["target_realized"].mean()))
        return LearningStats(brier_score=brier, directional_accuracy=acc, calibration_gap=calib)
