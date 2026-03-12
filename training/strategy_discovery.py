"""Découverte automatique de stratégies via RL (optionnel) et recherche génétique légère."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyCandidate:
    name: str
    sharpe: float
    max_drawdown: float
    profit_factor: float


class StrategyDiscovery:
    @staticmethod
    def generate_rule_candidates() -> list[dict]:
        return [
            {"name": "mom_20", "lookback": 20, "threshold": 0.02},
            {"name": "mom_50", "lookback": 50, "threshold": 0.03},
            {"name": "mean_revert_10", "lookback": 10, "threshold": -0.015},
        ]

    @staticmethod
    def evaluate_candidate(df: pd.DataFrame, candidate: dict) -> StrategyCandidate:
        ret = df["close"].pct_change().fillna(0)
        signal = (df["close"].pct_change(candidate["lookback"]).fillna(0) > candidate["threshold"]).astype(int).replace({0: -1})
        strat = signal.shift(1).fillna(0) * ret
        eq = (1 + strat).cumprod()
        sharpe = float(np.sqrt(252) * strat.mean() / (strat.std() + 1e-8))
        dd = float((eq / eq.cummax() - 1).min())
        pf = float(strat[strat > 0].sum() / (-strat[strat < 0].sum() + 1e-8))
        return StrategyCandidate(candidate["name"], sharpe, dd, pf)

    def select_robust(self, df: pd.DataFrame) -> list[StrategyCandidate]:
        evaluated = [self.evaluate_candidate(df, c) for c in self.generate_rule_candidates()]
        return [e for e in evaluated if e.sharpe > 1.5 and e.max_drawdown > -0.2 and e.profit_factor > 1.5]


def try_rl_training_note() -> str:
    """Hook RL: utilise stable-baselines3 si disponible sans casser l'exécution."""
    try:
        import stable_baselines3  # noqa: F401
        import gymnasium  # noqa: F401
        return "RL dependencies available: PPO/DQN/A2C pipeline can be enabled."
    except Exception:
        return "RL dependencies unavailable in runtime; genetic-rule discovery used as fallback."
