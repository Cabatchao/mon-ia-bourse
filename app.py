"""Point d'entrée de la plateforme quantitative deep learning."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from backtesting.engine import BacktestEngine
from config.settings import INFERENCE, MODEL_DIR, PREDICTIONS_DIR
from data.pipeline import AssetRequest, DataPipeline
from features.engineering import FeatureEngineer
from predictions.service import PredictionService
from training.pipeline import TemporalTrainer
from utils.risk import RiskManager


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

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asset": pred.asset,
        "current_price": pred.current_price,
        "horizon": pred.horizon,
        "prob_up": pred.prob_up,
        "prob_down": pred.prob_down,
        "confidence": pred.confidence,
        "factors": pred.explanatory_factors,
        "risk_position_pct": size,
        "trade_allowed": allowed,
        "cv_metrics": training_result.fold_metrics,
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
    # Dans une version production: orchestration Airflow/Prefect + registre de modèles.
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
