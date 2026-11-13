"""Stress tests: what VaR does not show."""
from __future__ import annotations

import numpy as np
import pandas as pd


def historical_scenarios(asset_returns: pd.DataFrame, weights: np.ndarray,
                         scenarios: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Loss of today's portfolio, bought and held, through past episodes."""
    rows = []
    for name, (start, end) in scenarios.items():
        window = asset_returns.loc[start:end]
        asset_total = (1 + window).prod() - 1
        rows.append({
            "scenario": name, "from": start, "to": end, "days": len(window),
            "portfolio_return_%": 100 * float(asset_total @ weights),
            "worst_asset": asset_total.idxmin(),
            "worst_asset_return_%": 100 * float(asset_total.min()),
        })
    return pd.DataFrame(rows).set_index("scenario")


def worst_losses(portfolio_returns: pd.Series, horizons=(1, 5, 10)) -> pd.DataFrame:
    """Largest realised loss over each horizon (overlapping, compounded)."""
    rows = []
    for h in horizons:
        cum = (1 + portfolio_returns).rolling(h).apply(np.prod, raw=True) - 1
        rows.append({"horizon_days": h, "worst_return_%": 100 * cum.min(),
                     "window_end": cum.idxmin().date()})
    return pd.DataFrame(rows).set_index("horizon_days")


def stressed_var(portfolio_returns: pd.Series, alpha: float = 0.01, window: int = 250) -> dict:
    """Historical VaR calibrated on the worst 250-day period in the sample,
    the idea behind Basel 2.5 stressed VaR and the FRTB stressed calibration."""
    rolling = -portfolio_returns.rolling(window).quantile(alpha)
    end = rolling.idxmax()
    start = portfolio_returns.index[portfolio_returns.index.get_loc(end) - window + 1]
    return {"stressed_var_%": 100 * float(rolling.max()),
            "current_var_%": 100 * float(rolling.iloc[-1]),
            "stress_window": f"{start.date()} to {end.date()}"}
