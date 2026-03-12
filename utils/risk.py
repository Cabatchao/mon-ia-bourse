"""Gestion du risque: sizing, stop-loss, contraintes drawdown."""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import RISK


@dataclass
class PositionDecision:
    size_pct: float
    stop_loss_price: float
    allowed: bool


class RiskManager:
    @staticmethod
    def position_size(confidence: float, volatility: float) -> float:
        raw = confidence / (1 + volatility)
        return float(min(RISK.max_position_pct, max(0.0, raw)))

    @staticmethod
    def stop_loss(entry_price: float, long: bool = True) -> float:
        return float(entry_price * (1 - RISK.stop_loss_pct if long else 1 + RISK.stop_loss_pct))

    @staticmethod
    def allow_trade(current_drawdown: float) -> bool:
        return current_drawdown > -RISK.max_portfolio_drawdown
