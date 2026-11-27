"""Draws from each model's one-day predictive distribution.

A sampler is a function `(rng, n_sim) -> array (n_sim, T)`. Column t holds returns
drawn from the distribution the model forecast for day t, with the information it
had at t-1. They are what the Acerbi-Szekely p-value is simulated from.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg, models

Sampler = Callable[[np.random.Generator, int], np.ndarray]


def normal_sampler(sigma: np.ndarray) -> Sampler:
    """r_t ~ N(0, sigma_t^2)."""
    sigma = np.asarray(sigma, dtype=float)
    return lambda rng, n: rng.standard_normal((n, len(sigma))) * sigma


def student_sampler(sigma: np.ndarray, nu: np.ndarray) -> Sampler:
    """r_t = sigma_t * z_t, z_t a Student-t(nu_t) rescaled to unit variance."""
    sigma, nu = np.asarray(sigma, dtype=float), np.asarray(nu, dtype=float)
    scale = sigma * np.sqrt((nu - 2) / nu)
    return lambda rng, n: stats.t.rvs(nu, size=(n, len(sigma)), random_state=rng) * scale


def empirical_sampler(pools: np.ndarray, scale: np.ndarray | None = None) -> Sampler:
    """r_t = scale_t * a draw with replacement from row t of `pools` (T, m)."""
    pools = np.asarray(pools, dtype=float)
    T, m = pools.shape
    scale = np.ones(T) if scale is None else np.asarray(scale, dtype=float)
    days = np.arange(T)
    return lambda rng, n: pools[days, rng.integers(0, m, size=(n, T))] * scale


def model_samplers(returns: pd.Series, forecasts: dict[str, pd.DataFrame],
                   garch_params: pd.DataFrame, residuals: dict[pd.Timestamp, np.ndarray],
                   window: int = cfg.HS_WINDOW) -> dict[str, Sampler]:
    """One sampler per model of `models.run_all`, on the same forecast dates.

    The parametric volatilities are read back from the 97.5% VaR forecasts
    (VaR_t = sigma_t * unit VaR), so each sampler matches its forecast exactly.
    `residuals` holds the standardised residuals of each GARCH estimation window,
    keyed by refit date; each set stays in use until the next refit.
    """
    oos = forecasts["GARCH-t"].index
    unit_normal = models.normal_var(cfg.ES_ALPHA)
    nu = garch_params["nu"].reindex(oos, method="ffill").to_numpy()
    garch_sigma = (forecasts["GARCH-t"]["var_975"].to_numpy()
                   / models.student_var(cfg.ES_ALPHA, nu))

    r = returns.to_numpy()
    first = returns.index.get_loc(oos[0])
    hs_pools = np.stack([r[t - window:t] for t in range(first, first + len(oos))])

    refits = pd.DatetimeIndex(sorted(residuals))
    in_use = refits.searchsorted(oos, side="right") - 1
    fhs_pools = np.stack([residuals[d] for d in refits])[in_use]

    return {
        "Historical simulation": empirical_sampler(hs_pools),
        "Normal (rolling)": normal_sampler(forecasts["Normal (rolling)"]["var_975"] / unit_normal),
        "EWMA normal": normal_sampler(forecasts["EWMA normal"]["var_975"] / unit_normal),
        "GARCH-t": student_sampler(garch_sigma, nu),
        "Filtered HS": empirical_sampler(fhs_pools, garch_sigma),
    }
