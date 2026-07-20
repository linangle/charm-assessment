"""
Two-sample Komogorov-Smirnov (K-S)

Z-score normalization is used because data is extremely right skewed.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .config import ALPHA, KS_C_ALPHA

def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Standardize to mean 0, sd 1. Constant series -> all zeros."""
    x = np.asarray(x, dtype=float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd

def ks_dcrit(n1: int, n2: int, alpha: float = ALPHA) -> float:
    if alpha not in KS_C_ALPHA:
        raise ValueError(f"no tabulated c(alpha) for alpha={alpha}; have {sorted(KS_C_ALPHA)}")
    if n1 < 1 or n2 < 1:
        return np.nan
    return KS_C_ALPHA[alpha] * np.sqrt((n1 + n2) / (n1 * n2))


def ks_test(obs: np.ndarray, model: np.ndarray, alpha: float = ALPHA) -> dict:
    obs = np.asarray(obs, dtype=float)
    model = np.asarray(model, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(model)
    obs, model = obs[ok], model[ok]

    n = obs.size
    if n < 3:
        return {
            "n": n,
            "D": np.nan,
            "p": np.nan,
            "Dcrit": np.nan,
            "reject": np.nan,
            "normalization": "zscore",
        }

    obs_normalized = zscore_normalize(obs)
    model_normalized = zscore_normalize(model)

    result = stats.ks_2samp(
        obs_normalized,
        model_normalized,
        alternative="two-sided",
        method="auto",
    )

    d = float(result.statistic)
    p = float(result.pvalue)
    dcrit = float(ks_dcrit(n, n, alpha))

    return {
        "n": n,
        "D": d,
        "p": p,
        "Dcrit": dcrit,
        "reject": bool(d > dcrit),
        "normalization": "zscore",
    }
