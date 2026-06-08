"""
Portfolio risk metrics:
  - Value-at-Risk: Historical Simulation, Variance-Covariance, Monte Carlo
  - Expected Shortfall (CVaR)
  - Rolling volatility
  - Correlation matrix
  - Maximum drawdown
"""
import pandas as pd
import numpy as np
from scipy import stats
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (CONFIDENCE_LEVEL, MC_SIMULATIONS, MC_HORIZON,
                    ROLLING_WINDOW, TRADING_DAYS)


# ═══════════════════════════════════════════════════════════════════════════
# VALUE-AT-RISK
# ═══════════════════════════════════════════════════════════════════════════

def var_historical(returns: pd.Series, confidence: float = CONFIDENCE_LEVEL) -> float:
    """
    Historical simulation VaR.
    Uses the empirical quantile of realised portfolio returns.
    """
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def var_parametric(returns: pd.Series, confidence: float = CONFIDENCE_LEVEL) -> float:
    """
    Variance-covariance (parametric) VaR.
    Assumes returns are normally distributed.
    VaR = -(mu + z * sigma)
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def var_monte_carlo(returns: pd.DataFrame, weights: np.ndarray,
                    confidence: float = CONFIDENCE_LEVEL,
                    n_sims: int = MC_SIMULATIONS,
                    horizon: int = MC_HORIZON) -> float:
    """
    Monte Carlo VaR.
    Simulates correlated multivariate normal returns over a given horizon.
    Returns the portfolio-level VaR.
    """
    mu = returns.mean().values
    cov = returns.cov().values

    # Cholesky decomposition for correlated draws
    L = np.linalg.cholesky(cov)

    # Simulate paths
    np.random.seed(42)
    simulated_returns = np.zeros(n_sims)

    for i in range(n_sims):
        cumulative = 0
        for _ in range(horizon):
            z = np.random.standard_normal(len(mu))
            daily_returns = mu + L @ z
            portfolio_return = np.dot(weights, daily_returns)
            cumulative += portfolio_return
        simulated_returns[i] = cumulative

    return -np.percentile(simulated_returns, (1 - confidence) * 100)


# ═══════════════════════════════════════════════════════════════════════════
# EXPECTED SHORTFALL (CVaR)
# ═══════════════════════════════════════════════════════════════════════════

def cvar_historical(returns: pd.Series, confidence: float = CONFIDENCE_LEVEL) -> float:
    """
    Expected Shortfall: mean of losses beyond the VaR threshold.
    """
    var = var_historical(returns, confidence)
    tail = returns[returns <= -var]
    return -tail.mean() if len(tail) > 0 else var


def cvar_parametric(returns: pd.Series, confidence: float = CONFIDENCE_LEVEL) -> float:
    """
    Parametric ES assuming normal distribution.
    ES = -(mu - sigma * phi(z_alpha) / (1 - alpha))
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    phi = stats.norm.pdf(z)
    return -(mu - sigma * phi / (1 - confidence))


# ═══════════════════════════════════════════════════════════════════════════
# VOLATILITY AND CORRELATION
# ═══════════════════════════════════════════════════════════════════════════

def rolling_volatility(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Annualised rolling standard deviation."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlations."""
    return returns.corr()


def rolling_correlation(returns: pd.DataFrame, asset1: str, asset2: str,
                        window: int = ROLLING_WINDOW) -> pd.Series:
    """Rolling pairwise correlation between two assets."""
    return returns[asset1].rolling(window).corr(returns[asset2])


# ═══════════════════════════════════════════════════════════════════════════
# DRAWDOWN
# ═══════════════════════════════════════════════════════════════════════════

def max_drawdown(returns: pd.Series) -> dict:
    """
    Compute maximum drawdown from a return series.
    Returns dict with max_drawdown, peak_date, trough_date, recovery_date.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = cumulative.loc[:trough_date].idxmax()

    # Recovery: first date after trough where cumulative >= peak value
    peak_value = cumulative.loc[peak_date]
    recovery = cumulative.loc[trough_date:]
    recovered = recovery[recovery >= peak_value]
    recovery_date = recovered.index[0] if len(recovered) > 0 else None

    return {
        "max_drawdown": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

def portfolio_stats(returns: pd.Series, risk_free: float = 0.03) -> dict:
    """
    Annualised return, volatility, Sharpe ratio, skewness, kurtosis.
    """
    ann_return = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else np.nan

    return {
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "skewness": stats.skew(returns.dropna()),
        "excess_kurtosis": stats.kurtosis(returns.dropna()),
        "max_drawdown": max_drawdown(returns)["max_drawdown"],
    }
