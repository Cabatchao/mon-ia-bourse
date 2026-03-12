"""Détection de cycles, régimes et probabilité d'explosion."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class CycleRegimeDetector:
    @staticmethod
    def spectral_cycle_strength(close: pd.Series, top_k: int = 3) -> float:
        vals = close.pct_change().dropna().values
        if len(vals) < 64:
            return 0.0
        fft_mag = np.abs(np.fft.rfft(vals))
        fft_mag[0] = 0
        top = np.sort(fft_mag)[-top_k:]
        return float(top.sum() / (fft_mag.sum() + 1e-8))

    @staticmethod
    def market_regime(close: pd.Series, vol_window: int = 20) -> str:
        ret = close.pct_change().dropna()
        if len(ret) < vol_window:
            return "unknown"
        trend = close.iloc[-1] / close.iloc[-vol_window] - 1
        vol = ret.rolling(vol_window).std().iloc[-1]
        if trend > 0.03 and vol < 0.03:
            return "bull_low_vol"
        if trend > 0.03 and vol >= 0.03:
            return "bull_high_vol"
        if trend < -0.03 and vol >= 0.03:
            return "bear_high_vol"
        return "range"

    @staticmethod
    def breakout_squeeze_score(close: pd.Series, window: int = 20) -> float:
        if len(close) < window + 5:
            return 0.0
        rolling_std = close.pct_change().rolling(window).std().dropna()
        if rolling_std.empty:
            return 0.0
        compression = float((rolling_std.quantile(0.2) - rolling_std.iloc[-1]) / (rolling_std.quantile(0.2) + 1e-8))
        peaks, _ = find_peaks(close.values[-(window + 5):])
        breakout_hint = 1.0 if len(peaks) >= 2 and close.iloc[-1] > close.iloc[-(window + 5):].max() * 0.99 else 0.0
        raw = max(0.0, compression) * 70 + breakout_hint * 30
        return float(np.clip(raw, 0, 100))
