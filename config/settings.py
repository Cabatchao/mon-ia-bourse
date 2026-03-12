"""Configuration centralisée du système quantitatif."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "lake"
PREDICTIONS_DIR = BASE_DIR / "predictions" / "outputs"
MODEL_DIR = BASE_DIR / "models" / "artifacts"

for folder in (DATA_DIR, PREDICTIONS_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataSources:
    yahoo_enabled: bool = True
    binance_enabled: bool = True
    fred_enabled: bool = True
    coingecko_enabled: bool = True
    alpha_vantage_enabled: bool = bool(os.getenv("ALPHA_VANTAGE_KEY"))


@dataclass(frozen=True)
class TrainingConfig:
    lookback: int = 64
    horizon: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class InferenceConfig:
    confidence_floor: float = 0.55
    prediction_horizons: tuple[str, ...] = ("1D", "5D", "20D")


@dataclass(frozen=True)
class RiskConfig:
    max_position_pct: float = 0.03
    stop_loss_pct: float = 0.02
    max_portfolio_drawdown: float = 0.12


DATA_SOURCES = DataSources()
TRAINING = TrainingConfig()
INFERENCE = InferenceConfig()
RISK = RiskConfig()
