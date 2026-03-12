"""Feature engineering technique + statistique pour séries temporelles financières."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import acf as sm_acf
    STATSMODELS_AVAILABLE = True
except Exception:  # noqa: BLE001
    sm_acf = None
    STATSMODELS_AVAILABLE = False

try:
    import ta
    TA_AVAILABLE = True
except Exception:  # noqa: BLE001
    ta = None
    TA_AVAILABLE = False


def _acf_lag1(values: np.ndarray) -> float:
    if len(values) < 3:
        return float("nan")
    v = np.asarray(values, dtype=float)
    if np.allclose(v.std(), 0):
        return 0.0
    if STATSMODELS_AVAILABLE:
        return float(sm_acf(v, nlags=1, fft=True)[1])
    # fallback sans statsmodels
    return float(np.corrcoef(v[:-1], v[1:])[0, 1])


class FeatureEngineer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in out.columns:
                raise ValueError(f"Colonne requise absente: {col}")

        if TA_AVAILABLE:
            out["rsi_14"] = ta.momentum.rsi(out["close"], window=14)
            out["macd"] = ta.trend.macd_diff(out["close"])
            out["sma_20"] = out["close"].rolling(20).mean()
            out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
            bb = ta.volatility.BollingerBands(close=out["close"], window=20, window_dev=2)
            out["bb_high"] = bb.bollinger_hband()
            out["bb_low"] = bb.bollinger_lband()
            out["atr_14"] = ta.volatility.average_true_range(out["high"], out["low"], out["close"], window=14)
            out["momentum_10"] = ta.momentum.roc(out["close"], window=10)
        else:
            out["rsi_14"] = 100 - (
                100
                / (
                    1
                    + out["close"].pct_change().clip(lower=0).rolling(14).mean()
                    / (out["close"].pct_change().abs().rolling(14).mean() + 1e-8)
                )
            )
            out["macd"] = out["close"].ewm(span=12, adjust=False).mean() - out["close"].ewm(span=26, adjust=False).mean()
            out["sma_20"] = out["close"].rolling(20).mean()
            out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
            out["bb_high"] = out["sma_20"] + 2 * out["close"].rolling(20).std()
            out["bb_low"] = out["sma_20"] - 2 * out["close"].rolling(20).std()
            out["atr_14"] = (out["high"] - out["low"]).rolling(14).mean()
            out["momentum_10"] = out["close"].pct_change(10)

        returns = np.log(out["close"] / out["close"].shift(1))
        out["autocorr_lag1"] = returns.rolling(90).apply(lambda x: _acf_lag1(x), raw=False)
        out["volatility_cluster_30"] = returns.rolling(30).std().rolling(30).std()
        out["skewness_60"] = returns.rolling(60).skew()
        out["kurtosis_60"] = returns.rolling(60).kurt()

        out["target"] = (out["close"].shift(-1) > out["close"]).astype(int)

        # Nettoyage robuste: on ne supprime pas tout à cause de colonnes optionnelles manquantes.
        out = out.replace([np.inf, -np.inf], np.nan)
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            out[col] = out[col].ffill().bfill()
        out = out.dropna(subset=["open", "high", "low", "close", "volume", "target"]).reset_index(drop=True)
        return out
