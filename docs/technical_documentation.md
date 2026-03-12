# Plateforme d'analyse quantitative deep learning autonome

## Architecture

- **Ingestion (`data/pipeline.py`)** : Yahoo (OHLCV), FRED (macro), CoinGecko (sentiment), Binance order book via `ccxt` (bid/ask/spread) quand disponible.
- **Storage (`data/storage.py`)** : Parquet historique, PostgreSQL métadonnées, Redis prévu pour cache temps réel.
- **Features (`features/engineering.py`)** : RSI, MACD, SMA/EMA, Bollinger, ATR, momentum, autocorr, clustering de vol, skew, kurtosis.
- **Cycles/Régimes (`features/cycles.py`)** : score spectral (FFT), classification de régime, score breakout/squeeze (0-100).
- **Modèles (`models/deep_models.py`)** : LSTM/GRU/TCN/CNN/Transformer/hybride.
- **Fallback robuste (`models/classical.py`, `training/pipeline.py`)** : si `torch` absent, entraînement automatique en régression logistique.
- **GNN status (`models/gnn.py`)** : détection explicite de disponibilité `torch_geometric`.
- **Ensemble (`models/ensemble.py`)** : averaging, bagging, stacking.
- **Discovery stratégies (`training/strategy_discovery.py`)** : recherche génétique légère + hook RL (PPO/DQN/A2C) selon dépendances.
- **Smart money (`utils/smart_money.py`)** : anomalies volume, imbalance order book, risque spoofing/layering, score 0-100.
- **Backtesting (`backtesting/engine.py`)** : Sharpe, Sortino, max drawdown, profit factor.
- **Risk (`utils/risk.py`)** : sizing, stop-loss, limite drawdown.
- **Apprentissage continu (`utils/continuous_learning.py`)** : suivi brier score / calibration / accuracy directionnelle.
- **Orchestration (`app.py`, `scripts/scheduler.py`)** : analyse quotidienne, réentraînement, rapport hebdomadaire.

## Garanties anti-hallucination

- Les scores sont calculés uniquement à partir des séries réelles chargées.
- Les modules externes absents sont explicitement signalés (`training_backend`, `rl_status`, `gnn_status`) au lieu de simuler des résultats.
- Toute indisponibilité API retourne des structures vides et un log d’avertissement.

## Scalabilité 50 000 actifs/jour

- Partitionnement par univers + `multiprocessing` (`utils/distributed.py`).
- En production: scheduler distribué, file de jobs, workers CPU/GPU, monitoring de latence et dérive.
