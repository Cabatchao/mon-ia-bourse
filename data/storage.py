"""Abstractions de stockage: PostgreSQL métadonnées, Parquet historique, Redis cache."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


@dataclass
class StorageConfig:
    postgres_url: str = "postgresql+psycopg2://user:pass@localhost:5432/quant"
    redis_url: str = "redis://localhost:6379/0"


class DataLakeStorage:
    def __init__(self, root: Path, config: StorageConfig) -> None:
        self.root = root
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)

    def save_history(self, df: pd.DataFrame, key: str) -> Path:
        path = self.root / f"{key}.parquet"
        df.to_parquet(path, index=False)
        return path

    def load_history(self, key: str) -> pd.DataFrame:
        path = self.root / f"{key}.parquet"
        return pd.read_parquet(path)

    def save_metadata(self, metadata: pd.DataFrame, table: str = "asset_metadata") -> None:
        engine = create_engine(self.config.postgres_url)
        metadata.to_sql(table, engine, if_exists="append", index=False)
