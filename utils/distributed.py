"""Traitement distribué pour grands univers d'actifs."""
from __future__ import annotations

from multiprocessing import Pool, cpu_count
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(func: Callable[[T], R], items: Iterable[T], workers: int | None = None) -> list[R]:
    worker_count = workers or max(1, cpu_count() - 1)
    with Pool(worker_count) as pool:
        return pool.map(func, items)
