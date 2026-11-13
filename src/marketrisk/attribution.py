"""Which positions carry the risk. Both decompositions add up to the total."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def component_var(asset_returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.01) -> pd.DataFrame:
    """Euler decomposition of normal VaR.

    VaR = z * sigma_p with sigma_p = sqrt(w' S w). VaR is homogeneous of degree 1
    in w, so VaR = sum_i w_i * dVaR/dw_i, and component_i = z * w_i * (S w)_i / sigma_p.
    """
    w = np.asarray(weights, dtype=float)
    cov = asset_returns.cov().to_numpy()
    sigma_p = np.sqrt(w @ cov @ w)
    comp = -stats.norm.ppf(alpha) * w * (cov @ w) / sigma_p
    return pd.DataFrame({"weight_%": 100 * w, "component_var_%": 100 * comp,
                         "share_of_var_%": 100 * comp / comp.sum()}, index=asset_returns.columns)


def component_es(asset_returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.025) -> pd.DataFrame:
    """Historical ES by position: each asset's average contribution to portfolio
    P&L on the days when the portfolio was in its alpha tail. No normality needed."""
    w = np.asarray(weights, dtype=float)
    contrib = asset_returns * w
    port = contrib.sum(axis=1)
    tail = port <= port.quantile(alpha)
    comp = -contrib[tail].mean()
    return pd.DataFrame({"weight_%": 100 * w, "component_es_%": 100 * comp,
                         "share_of_es_%": 100 * comp / comp.sum()}, index=asset_returns.columns)
