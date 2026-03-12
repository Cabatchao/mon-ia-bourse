"""Moteur de backtesting et métriques de performance."""
from __future__ import annotations

import numpy as np
import pandas as pd


class BacktestEngine:
    @staticmethod
    def run(df: pd.DataFrame, prob_up_col: str = "prob_up", threshold: float = 0.55) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = np.where(out[prob_up_col] >= threshold, 1, -1)
        out["ret"] = out["close"].pct_change().fillna(0)
        out["strategy_ret"] = out["signal"].shift(1).fillna(0) * out["ret"]
        out["equity"] = (1 + out["strategy_ret"]).cumprod()
        return out

    @staticmethod
    def sharpe(returns: pd.Series, freq: int = 252) -> float:
        return float(np.sqrt(freq) * returns.mean() / (returns.std() + 1e-8))

    @staticmethod
    def sortino(returns: pd.Series, freq: int = 252) -> float:
        downside = returns[returns < 0]
        return float(np.sqrt(freq) * returns.mean() / (downside.std() + 1e-8))

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        running_max = equity.cummax()
        drawdown = (equity / running_max) - 1
        return float(drawdown.min())

    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        gross_profit = returns[returns > 0].sum()
        gross_loss = -returns[returns < 0].sum()
        return float(gross_profit / (gross_loss + 1e-8))

    @staticmethod
    def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
        return float((y_true.values == y_pred.values).mean())
