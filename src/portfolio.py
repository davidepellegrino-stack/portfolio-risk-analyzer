"""
Mean-variance portfolio optimization.
  - Efficient frontier (Monte Carlo simulation of random portfolios)
  - Minimum variance portfolio
  - Maximum Sharpe ratio portfolio
  - Equal-weight benchmark
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import RISK_FREE_RATE, TRADING_DAYS, MC_SIMULATIONS


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def portfolio_performance(weights: np.ndarray, returns: pd.DataFrame) -> tuple:
    """
    Compute annualised return, volatility, and Sharpe ratio for a given
    weight vector.
    """
    weights = np.array(weights)
    ann_return = np.dot(weights, returns.mean()) * TRADING_DAYS
    ann_vol = np.sqrt(
        weights @ (returns.cov() * TRADING_DAYS) @ weights
    )
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan
    return ann_return, ann_vol, sharpe


# ═══════════════════════════════════════════════════════════════════════════
# EFFICIENT FRONTIER (Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_random_portfolios(returns: pd.DataFrame,
                               n_portfolios: int = MC_SIMULATIONS) -> pd.DataFrame:
    """
    Generate random portfolios to trace the efficient frontier.
    Returns DataFrame with weights, return, volatility, Sharpe ratio.
    """
    n_assets = len(returns.columns)
    results = []

    np.random.seed(42)
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))  # Random weights summing to 1
        r, v, s = portfolio_performance(w, returns)
        results.append({
            **{f"w_{t}": w[i] for i, t in enumerate(returns.columns)},
            "return": r,
            "volatility": v,
            "sharpe": s,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMISED PORTFOLIOS
# ═══════════════════════════════════════════════════════════════════════════

def _neg_sharpe(weights, returns):
    _, _, sharpe = portfolio_performance(weights, returns)
    return -sharpe


def _portfolio_vol(weights, returns):
    _, vol, _ = portfolio_performance(weights, returns)
    return vol


def max_sharpe_portfolio(returns: pd.DataFrame) -> dict:
    """
    Find portfolio weights that maximise the Sharpe ratio.
    Uses SLSQP constrained optimisation.
    """
    n = len(returns.columns)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        _neg_sharpe, w0, args=(returns,),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000}
    )

    weights = result.x
    r, v, s = portfolio_performance(weights, returns)

    return {
        "weights": dict(zip(returns.columns, weights.round(4))),
        "return": r,
        "volatility": v,
        "sharpe": s,
        "label": "Max Sharpe",
    }


def min_variance_portfolio(returns: pd.DataFrame) -> dict:
    """
    Find the global minimum variance portfolio.
    """
    n = len(returns.columns)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        _portfolio_vol, w0, args=(returns,),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000}
    )

    weights = result.x
    r, v, s = portfolio_performance(weights, returns)

    return {
        "weights": dict(zip(returns.columns, weights.round(4))),
        "return": r,
        "volatility": v,
        "sharpe": s,
        "label": "Min Variance",
    }


def equal_weight_portfolio(returns: pd.DataFrame) -> dict:
    """Equal-weight benchmark portfolio."""
    n = len(returns.columns)
    weights = np.ones(n) / n
    r, v, s = portfolio_performance(weights, returns)

    return {
        "weights": dict(zip(returns.columns, weights.round(4))),
        "return": r,
        "volatility": v,
        "sharpe": s,
        "label": "Equal Weight",
    }


def summarise_portfolios(returns: pd.DataFrame) -> pd.DataFrame:
    """Compare equal weight, max Sharpe, and min variance portfolios."""
    portfolios = [
        equal_weight_portfolio(returns),
        max_sharpe_portfolio(returns),
        min_variance_portfolio(returns),
    ]

    rows = []
    for p in portfolios:
        row = {
            "Portfolio": p["label"],
            "Ann. Return (%)": round(p["return"] * 100, 2),
            "Ann. Volatility (%)": round(p["volatility"] * 100, 2),
            "Sharpe Ratio": round(p["sharpe"], 4),
        }
        for ticker, w in p["weights"].items():
            row[ticker] = f"{w*100:.1f}%"
        rows.append(row)

    return pd.DataFrame(rows)
