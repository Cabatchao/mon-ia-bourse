# Plateforme d'analyse quantitative deep learning

## 1) Architecture complète

- **Data Pipeline (`data/pipeline.py`)** : collecte Yahoo Finance + macro FRED + proxy sentiment CoinGecko, puis normalisation OHLCV et volatilité réalisée.
- **Data Storage (`data/storage.py`)** : design Data Lake avec Parquet (historique), PostgreSQL (métadonnées) et extension Redis prévue pour cache temps réel.
- **Feature Engineering (`features/engineering.py`)** : indicateurs techniques (RSI, MACD, SMA/EMA, Bollinger, ATR, Momentum) + stats avancées (autocorr, clustering de vol, skew, kurtosis).
- **Deep Models (`models/deep_models.py`)** : LSTM, GRU, TCN, CNN-TS, Transformer, hybride LSTM+Transformer+MLP.
- **Ensemble (`models/ensemble.py`)** : model averaging, bagging, stacking.
- **Training (`training/pipeline.py`)** : fenêtres temporelles, TimeSeriesSplit, walk-forward validation, métriques accuracy/log-loss.
- **Backtesting (`backtesting/engine.py`)** : Sharpe, Sortino, Max Drawdown, Profit Factor, accuracy directionnelle.
- **Prédiction (`predictions/service.py`)** : probabilités hausse/baisse, confiance, horizon, facteurs explicatifs.
- **Learning loop (`utils/continuous_learning.py`)** : log des prédictions, comparaison avec réalisés, Brier score, calibration.
- **Risk (`utils/risk.py`)** : sizing, stop-loss, garde-fou drawdown.
- **Automatisation (`scripts/scheduler.py`)** : jobs quotidiens, ré-entraînement, rapport hebdomadaire.

## 2) Réalisme / anti-hallucination

- Les appels API sont faits sur des fournisseurs réels (Yahoo/FRED/CoinGecko).
- Les données indisponibles sont gérées explicitement (DataFrame vide + logs), sans fabrication de résultats.
- Les sorties de probabilité sont dérivées de modèles entraînés et d'une fonction sigmoïde.

## 3) Passage à 50 000 actifs / jour

- `utils/distributed.py` fournit un `parallel_map` multiprocessing.
- En production, recommander:
  - découpage univers en partitions,
  - exécution cluster (Ray/Dask/Spark),
  - feature store + registre modèles,
  - file de messages (Kafka) et workers GPU.

## 4) Validation empirique recommandée

- Validation walk-forward systématique.
- A/B testing de modèles et d'ensembles.
- Monitoring de dérive de distribution et recalibration probabiliste.
- Stress tests par régime de marché.
