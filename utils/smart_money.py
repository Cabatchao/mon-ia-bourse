"""Détection activity whales / smart money via anomalies volume et microstructure."""
from __future__ import annotations

import numpy as np
import pandas as pd


class SmartMoneyDetector:
    @staticmethod
    def volume_anomaly_score(volume: pd.Series, window: int = 30) -> float:
        if len(volume) < window + 1:
            return 0.0
        z = (volume.iloc[-1] - volume.rolling(window).mean().iloc[-1]) / (volume.rolling(window).std().iloc[-1] + 1e-8)
        return float(np.clip((z / 4) * 100, 0, 100))

    @staticmethod
    def orderbook_imbalance_score(bid_volume: float | None, ask_volume: float | None) -> float:
        if bid_volume is None or ask_volume is None:
            return 0.0
        imb = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)
        return float(np.clip((imb + 1) * 50, 0, 100))

    @staticmethod
    def spoofing_risk_score(spread_series: pd.Series, depth_churn_series: pd.Series) -> float:
        if spread_series.empty or depth_churn_series.empty:
            return 0.0
        spread_jump = float((spread_series.iloc[-1] / (spread_series.median() + 1e-8)) - 1)
        churn = float(depth_churn_series.iloc[-1] / (depth_churn_series.rolling(20).mean().iloc[-1] + 1e-8))
        raw = max(0.0, spread_jump) * 40 + max(0.0, churn - 1) * 60
        return float(np.clip(raw, 0, 100))

    def smart_money_score(
        self,
        volume: pd.Series,
        bid_volume: float | None = None,
        ask_volume: float | None = None,
        spread_series: pd.Series | None = None,
        depth_churn_series: pd.Series | None = None,
    ) -> float:
        spread_series = spread_series if spread_series is not None else pd.Series(dtype=float)
        depth_churn_series = depth_churn_series if depth_churn_series is not None else pd.Series(dtype=float)
        scores = [
            self.volume_anomaly_score(volume),
            self.orderbook_imbalance_score(bid_volume, ask_volume),
            self.spoofing_risk_score(spread_series, depth_churn_series),
        ]
        return float(np.mean(scores))
