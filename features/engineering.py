"""Feature engineering technique + statistique pour séries temporelles financières."""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta
from statsmodels.tsa.stattools import acf


class FeatureEngineer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["rsi_14"] = ta.momentum.rsi(out["close"], window=14)
        out["macd"] = ta.trend.macd_diff(out["close"])
        out["sma_20"] = out["close"].rolling(20).mean()
        out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()

        bb = ta.volatility.BollingerBands(close=out["close"], window=20, window_dev=2)
        out["bb_high"] = bb.bollinger_hband()
        out["bb_low"] = bb.bollinger_lband()
        out["atr_14"] = ta.volatility.average_true_range(out["high"], out["low"], out["close"], window=14)
        out["momentum_10"] = ta.momentum.roc(out["close"], window=10)

        returns = np.log(out["close"] / out["close"].shift(1))
        out["autocorr_lag1"] = returns.rolling(90).apply(lambda x: acf(x, nlags=1, fft=True)[1] if len(x) > 2 else np.nan)
        out["volatility_cluster_30"] = returns.rolling(30).std().rolling(30).std()
        out["skewness_60"] = returns.rolling(60).skew()
        out["kurtosis_60"] = returns.rolling(60).kurt()

        out["target"] = (out["close"].shift(-1) > out["close"]).astype(int)
        return out.dropna().reset_index(drop=True)
