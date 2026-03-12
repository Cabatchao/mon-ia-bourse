"""Pipeline de collecte et normalisation de données financières multi-sources."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import requests
try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None
from pandas import DataFrame

from config.settings import DATA_DIR, DATA_SOURCES

LOGGER = logging.getLogger(__name__)


@dataclass
class AssetRequest:
    symbol: str
    market: str  # equity | crypto | commodity
    start: str = "2018-01-01"


class DataPipeline:
    """Collecte OHLCV + enrichissements macro/sentiment + microstructure si dispo."""

    def __init__(self) -> None:
        self.raw_dir = DATA_DIR / "raw"
        self.curated_dir = DATA_DIR / "curated"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.curated_dir.mkdir(parents=True, exist_ok=True)

    def fetch_yahoo(self, request: AssetRequest) -> DataFrame:
        if not DATA_SOURCES.yahoo_enabled or yf is None:
            return pd.DataFrame()
        data = yf.download(request.symbol, start=request.start, progress=False, auto_adjust=False)
        if data.empty:
            return data
        data.columns = [c.lower().replace(" ", "_") for c in data.columns]
        data["symbol"] = request.symbol
        data["market"] = request.market
        data["source"] = "yahoo"
        return data.reset_index().rename(columns={"Date": "timestamp", "date": "timestamp"})

    def fetch_binance_orderbook_proxy(self, symbol: str) -> dict[str, float]:
        if not DATA_SOURCES.binance_enabled:
            return {}
        try:
            import ccxt  # type: ignore

            exchange = ccxt.binance()
            market_symbol = symbol.replace("-USD", "/USDT")
            ob = exchange.fetch_order_book(market_symbol, limit=50)
            bid_vol = float(sum(b[1] for b in ob.get("bids", [])))
            ask_vol = float(sum(a[1] for a in ob.get("asks", [])))
            spread = float(ob["asks"][0][0] - ob["bids"][0][0]) if ob.get("asks") and ob.get("bids") else np.nan
            return {"bid_volume": bid_vol, "ask_volume": ask_vol, "spread": spread}
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Binance indisponible pour %s: %s", symbol, exc)
            return {}

    def fetch_coingecko_sentiment_proxy(self, symbol: str) -> DataFrame:
        if not DATA_SOURCES.coingecko_enabled:
            return pd.DataFrame()
        coin_id = symbol.replace("-USD", "").lower()
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        try:
            payload = requests.get(url, timeout=10).json()
            score = payload.get("sentiment_votes_up_percentage", np.nan)
            market_cap_rank = payload.get("market_cap_rank", np.nan)
            return pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp.utcnow().normalize()],
                    "symbol": [symbol],
                    "sentiment_score": [score],
                    "market_cap_rank": [market_cap_rank],
                }
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("CoinGecko indisponible pour %s: %s", symbol, exc)
            return pd.DataFrame()

    def fetch_fred_macro(self, series_id: str = "DGS10") -> DataFrame:
        if not DATA_SOURCES.fred_enabled:
            return pd.DataFrame()
        try:
            macro = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
            macro.columns = ["timestamp", "macro_value"]
            macro["timestamp"] = pd.to_datetime(macro["timestamp"])
            macro["macro_value"] = pd.to_numeric(macro["macro_value"], errors="coerce")
            return macro.dropna().reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("FRED indisponible: %s", exc)
            return pd.DataFrame()

    def build_dataset(self, request: AssetRequest) -> DataFrame:
        price = self.fetch_yahoo(request)
        if price.empty:
            return price

        macro = self.fetch_fred_macro()
        if not macro.empty:
            price = price.merge(macro, how="left", on="timestamp")

        sentiment = self.fetch_coingecko_sentiment_proxy(request.symbol)
        if not sentiment.empty:
            price = price.merge(sentiment, how="left", on=["timestamp", "symbol"])

        if request.market == "crypto":
            micro = self.fetch_binance_orderbook_proxy(request.symbol)
            for key, value in micro.items():
                price[key] = value

        price["log_return"] = np.log(price["close"] / price["close"].shift(1))
        price["realized_volatility_20"] = price["log_return"].rolling(20).std() * np.sqrt(252)
        price["spread"] = price.get("spread", np.nan)
        return price.dropna().reset_index(drop=True)

    def persist(self, df: DataFrame, symbol: str) -> str:
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = self.curated_dir / f"{symbol}_{now}.parquet"
        df.to_parquet(out_path, index=False)
        return str(out_path)

    def run_batch(self, requests_: Iterable[AssetRequest]) -> dict[str, str]:
        outputs: dict[str, str] = {}
        for req in requests_:
            dataset = self.build_dataset(req)
            if dataset.empty:
                continue
            outputs[req.symbol] = self.persist(dataset, req.symbol)
        return outputs
