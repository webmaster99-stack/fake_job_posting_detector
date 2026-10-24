"""Can a model be served on Render's free tier? Size on disk and single-posting latency.

The limits are `params.yaml: selection` (max_model_size_mb, max_p95_latency_ms). Latency is
measured one posting at a time, as the API serves them, on one CPU thread to approximate a small
server. It depends on the machine, so it is logged to MLflow and never committed as a metric.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WARMUP = 5


def size_mb(path: Path) -> float:
    """Size of a file, or of every file under a directory, in MB."""
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / 1e6


def latencies_ms(predict: Callable[[pd.DataFrame], Any], postings: pd.DataFrame) -> np.ndarray:
    """Wall time of `predict` on each posting alone, after a few warm-up calls."""
    for i in range(min(WARMUP, len(postings))):
        predict(postings.iloc[[i]])
    times = []
    for i in range(len(postings)):
        start = time.perf_counter()
        predict(postings.iloc[[i]])
        times.append((time.perf_counter() - start) * 1000)
    return np.array(times)


def check(
    model_path: Path,
    predict: Callable[[pd.DataFrame], Any],
    postings: pd.DataFrame,
    limits: dict[str, Any],
) -> dict[str, Any]:
    """Size, p50/p95 latency, and whether both are within `limits` (`params.yaml: selection`)."""
    times = latencies_ms(predict, postings)
    size = size_mb(model_path)
    p95 = float(np.percentile(times, 95))
    return {
        "size_mb": round(size, 1),
        "p50_latency_ms": round(float(np.percentile(times, 50)), 1),
        "p95_latency_ms": round(p95, 1),
        "deployable": size <= limits["max_model_size_mb"] and p95 <= limits["max_p95_latency_ms"],
    }
