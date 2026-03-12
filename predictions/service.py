"""Service de prédiction probabiliste multi-actifs et scoring de confiance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None


@dataclass
class PredictionOutput:
    asset: str
    current_price: float
    horizon: str
    prob_up: float
    prob_down: float
    confidence: float
    explanatory_factors: dict[str, float]


class PredictionService:
    @staticmethod
    def infer(model: object, latest_window: np.ndarray, feature_names: list[str], asset: str, price: float, horizon: str) -> PredictionOutput:
        if torch is not None and hasattr(model, "eval"):
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(latest_window[None, ...], dtype=torch.float32))
                prob_up = float(torch.sigmoid(logits).item())
        else:
            prob_up = float(model.predict_proba(latest_window[-1:, :])[0])

        prob_down = 1.0 - prob_up
        confidence = float(abs(prob_up - 0.5) * 2)

        factors = {feature_names[i]: float(latest_window[-1, i]) for i in np.argsort(np.abs(latest_window[-1]))[-5:]}
        return PredictionOutput(
            asset=asset,
            current_price=price,
            horizon=horizon,
            prob_up=prob_up,
            prob_down=prob_down,
            confidence=confidence,
            explanatory_factors=factors,
        )

    @staticmethod
    def to_frame(preds: list[PredictionOutput]) -> pd.DataFrame:
        return pd.DataFrame([p.__dict__ for p in preds])
