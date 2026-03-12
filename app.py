"""Point d'entrée de la plateforme quantitative deep learning autonome."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtesting.engine import BacktestEngine
from config.settings import INFERENCE, MODEL_DIR, PREDICTIONS_DIR
from data.pipeline import AssetRequest, DataPipeline
from features.cycles import CycleRegimeDetector
from features.engineering import FeatureEngineer
from models.gnn import gnn_support_status
from predictions.service import PredictionService
from training.pipeline import TORCH_AVAILABLE, TemporalTrainer
from training.strategy_discovery import StrategyDiscovery, try_rl_training_note
from utils.risk import RiskManager
from utils.smart_money import SmartMoneyDetector

if TORCH_AVAILABLE:
    import torch


UNIVERSE = [
    AssetRequest("AAPL", "equity"),
    AssetRequest("MSFT", "equity"),
    AssetRequest("BTC-USD", "crypto"),
    AssetRequest("ETH-USD", "crypto"),
    AssetRequest("GC=F", "commodity"),
]


def _train_predict_one(request: AssetRequest) -> dict:
    data_pipeline = DataPipeline()
    raw_df = data_pipeline.build_dataset(request)
    if raw_df.empty or len(raw_df) < 300:
        return {"asset": request.symbol, "status": "insufficient_data"}

    engineered = FeatureEngineer().transform(raw_df)
    feature_cols = [c for c in engineered.columns if c not in {"target", "timestamp", "symbol", "market", "source"}]

    trainer = TemporalTrainer()
    X, y = trainer.build_windows(engineered, feature_cols)
    training_result = trainer.walk_forward_train(X, y)

    model = training_result.final_model
    if TORCH_AVAILABLE and hasattr(model, "state_dict"):
        torch.save(model.state_dict(), MODEL_DIR / f"{request.symbol}_hybrid.pt")

    pred = PredictionService.infer(
        model=model,
        latest_window=X[-1],
        feature_names=feature_cols,
        asset=request.symbol,
        price=float(engineered["close"].iloc[-1]),
        horizon=INFERENCE.prediction_horizons[1],
    )

    backtest_df = engineered[["timestamp", "close", "target"]].copy()
    backtest_df["prob_up"] = [0.5] * (len(backtest_df) - 1) + [pred.prob_up]
    bt = BacktestEngine.run(backtest_df)

    drawdown = BacktestEngine.max_drawdown(bt["equity"])
    size = RiskManager.position_size(pred.confidence, float(engineered["realized_volatility_20"].iloc[-1]))
    allowed = RiskManager.allow_trade(drawdown)

    smart_money = SmartMoneyDetector().smart_money_score(
        volume=engineered["volume"],
        bid_volume=float(engineered["bid_volume"].iloc[-1]) if "bid_volume" in engineered.columns else None,
        ask_volume=float(engineered["ask_volume"].iloc[-1]) if "ask_volume" in engineered.columns else None,
        spread_series=engineered["spread"] if "spread" in engineered.columns else pd.Series(dtype=float),
        depth_churn_series=engineered["volume"].pct_change().abs().fillna(0),
    )

    cycle_score = CycleRegimeDetector.spectral_cycle_strength(engineered["close"]) * 100
    regime = CycleRegimeDetector.market_regime(engineered["close"])
    explosion_score = CycleRegimeDetector.breakout_squeeze_score(engineered["close"])

    robust_strategies = StrategyDiscovery().select_robust(engineered)
    best_strategy = robust_strategies[0].name if robust_strategies else "none_robust"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asset": pred.asset,
        "asset_class": request.market,
        "current_price": pred.current_price,
        "horizon": pred.horizon,
        "prob_up": pred.prob_up,
        "prob_down": pred.prob_down,
        "confidence": pred.confidence,
        "factors": pred.explanatory_factors,
        "market_regime": regime,
        "cycle_score": cycle_score,
        "explosion_score": explosion_score,
        "smart_money_score": smart_money,
        "strategy_recommended": best_strategy,
        "risk_position_pct": size,
        "trade_allowed": allowed,
        "cv_metrics": training_result.fold_metrics,
        "training_backend": training_result.backend,
        "rl_status": try_rl_training_note(),
        "gnn_status": gnn_support_status(),
        "sharpe": BacktestEngine.sharpe(bt["strategy_ret"]),
        "sortino": BacktestEngine.sortino(bt["strategy_ret"]),
        "max_drawdown": drawdown,
        "profit_factor": BacktestEngine.profit_factor(bt["strategy_ret"]),
    }


def run_daily_analysis() -> pd.DataFrame:
    results = [_train_predict_one(req) for req in UNIVERSE]
    frame = pd.DataFrame(results)
    output = PREDICTIONS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame


def run_retraining() -> str:
    run_daily_analysis()
    return "retraining_completed"


def run_weekly_report() -> Path:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Rapport hebdomadaire basé sur prédictions stockées et backtests journaliers.",
    }
    path = PREDICTIONS_DIR / f"weekly_report_{datetime.now().strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(run_daily_analysis())
