"""One-day-ahead VaR and ES forecasts.

Every function takes the daily portfolio return series and returns a DataFrame
indexed by forecast date with three columns, all positive loss fractions:

    var_99   99% VaR
    var_975  97.5% VaR
    es_975   97.5% Expected Shortfall

The forecast for day t uses returns up to t-1 only. tests/test_models.py checks
this by changing the future and confirming that past forecasts do not move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg

COLUMNS = ["var_99", "var_975", "es_975"]


# ── Tail formulas for a standardised (mean 0, variance 1) innovation ─────────

def normal_var(alpha: float) -> float:
    """VaR of a N(0,1) loss at tail probability alpha, e.g. 2.326 for alpha = 1%."""
    return -stats.norm.ppf(alpha)


def normal_es(alpha: float) -> float:
    """ES of a N(0,1) loss: phi(z_alpha) / alpha, e.g. 2.338 for alpha = 2.5%."""
    return stats.norm.pdf(stats.norm.ppf(alpha)) / alpha


def student_var(alpha: float, nu: float) -> float:
    """VaR of a unit-variance Student-t loss. sqrt((nu-2)/nu) rescales t to variance 1."""
    return -stats.t.ppf(alpha, nu) * np.sqrt((nu - 2) / nu)


def student_es(alpha: float, nu: float) -> float:
    """ES of a unit-variance Student-t loss (McNeil, Frey and Embrechts, 2015, ch. 2)."""
    q = stats.t.ppf(alpha, nu)
    return stats.t.pdf(q, nu) / alpha * (nu + q**2) / (nu - 1) * np.sqrt((nu - 2) / nu)


def empirical_tail(sample: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical VaR and ES of a return sample, as positive losses."""
    q = np.quantile(sample, alpha)
    return -q, -sample[sample <= q].mean()


def _frame(index, var_99, var_975, es_975) -> pd.DataFrame:
    return pd.DataFrame({"var_99": var_99, "var_975": var_975, "es_975": es_975}, index=index)


# ── 1. Historical simulation ────────────────────────────────────────────────

def historical_simulation(returns: pd.Series, window: int = cfg.HS_WINDOW,
                          start: int = cfg.BURN_IN) -> pd.DataFrame:
    """Empirical quantile of the last `window` returns. No distribution assumed,
    but every day in the window has the same weight, so it reacts slowly."""
    r = returns.to_numpy()
    rows = []
    for t in range(start, len(r)):
        past = r[t - window:t]
        v99, _ = empirical_tail(past, cfg.VAR_ALPHA)
        v975, e975 = empirical_tail(past, cfg.ES_ALPHA)
        rows.append((v99, v975, e975))
    return pd.DataFrame(rows, index=returns.index[start:], columns=COLUMNS)


# ── 2. Normal with rolling volatility ───────────────────────────────────────

def normal_rolling(returns: pd.Series, window: int = cfg.NORMAL_WINDOW,
                   start: int = cfg.BURN_IN) -> pd.DataFrame:
    """Variance-covariance VaR: normal returns, zero mean, equal-weighted sigma.
    The daily mean is set to zero: it is tiny next to sigma and badly estimated."""
    sigma = returns.rolling(window).std().shift(1).iloc[start:]
    return _frame(sigma.index, normal_var(cfg.VAR_ALPHA) * sigma,
                  normal_var(cfg.ES_ALPHA) * sigma, normal_es(cfg.ES_ALPHA) * sigma)


# ── 3. EWMA (RiskMetrics) ───────────────────────────────────────────────────

def ewma_volatility(returns: pd.Series, lam: float = cfg.EWMA_LAMBDA, init: int = 250) -> pd.Series:
    """sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2, a forecast made at t-1."""
    r = returns.to_numpy()
    var = np.empty(len(r))
    var[:init + 1] = r[:init].var()
    for t in range(init + 1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return pd.Series(np.sqrt(var), index=returns.index)


def ewma_normal(returns: pd.Series, lam: float = cfg.EWMA_LAMBDA,
                start: int = cfg.BURN_IN) -> pd.DataFrame:
    """Normal VaR on EWMA volatility: reacts within days, but the tails are still normal."""
    sigma = ewma_volatility(returns, lam).iloc[start:]
    return _frame(sigma.index, normal_var(cfg.VAR_ALPHA) * sigma,
                  normal_var(cfg.ES_ALPHA) * sigma, normal_es(cfg.ES_ALPHA) * sigma)


# ── 4 and 5. GARCH(1,1)-t and filtered historical simulation ────────────────

def garch_models(returns: pd.Series, window: int = cfg.GARCH_WINDOW,
                 refit: int = cfg.GARCH_REFIT, start: int = cfg.BURN_IN
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """GARCH(1,1) with Student-t innovations, and FHS built on the same volatility.

    sigma2_t = omega + a * r_{t-1}^2 + b * sigma2_{t-1}

    Parameters are re-estimated by maximum likelihood every `refit` days on the
    last `window` returns. Between two estimations the parameters are frozen and
    sigma is rolled forward with the recursion above.

    GARCH-t : VaR and ES = sigma_t x the tail of a unit-variance Student-t(nu).
    FHS     : VaR and ES = sigma_t x the empirical tail of the standardised
              residuals r/sigma from the estimation window (Barone-Adesi et al.,
              1999). GARCH supplies the volatility, the data supply the tail shape.

    Returns (garch_t forecasts, fhs forecasts, fitted parameters by refit date).
    """
    from arch import arch_model

    r = returns.to_numpy() * 100.0        # percent returns keep the optimiser well scaled
    garch_rows, fhs_rows, params = [], [], []

    for t in range(start, len(r)):
        if (t - start) % refit == 0:
            fit = arch_model(r[t - window:t], mean="Zero", vol="GARCH", p=1, q=1,
                             dist="t").fit(disp="off")
            omega, a, b, nu = fit.params[["omega", "alpha[1]", "beta[1]", "nu"]]
            resid = r[t - window:t] / fit.conditional_volatility
            sigma2_prev, r_prev = fit.conditional_volatility[-1] ** 2, r[t - 1]
            params.append((returns.index[t], omega, a, b, nu))
        sigma2 = omega + a * r_prev**2 + b * sigma2_prev
        sigma = np.sqrt(sigma2) / 100.0

        garch_rows.append((student_var(cfg.VAR_ALPHA, nu) * sigma,
                           student_var(cfg.ES_ALPHA, nu) * sigma,
                           student_es(cfg.ES_ALPHA, nu) * sigma))
        v99, _ = empirical_tail(resid, cfg.VAR_ALPHA)
        v975, e975 = empirical_tail(resid, cfg.ES_ALPHA)
        fhs_rows.append((v99 * sigma, v975 * sigma, e975 * sigma))

        sigma2_prev, r_prev = sigma2, r[t]

    idx = returns.index[start:]
    return (pd.DataFrame(garch_rows, index=idx, columns=COLUMNS),
            pd.DataFrame(fhs_rows, index=idx, columns=COLUMNS),
            pd.DataFrame(params, columns=["date", "omega", "alpha", "beta", "nu"]).set_index("date"))


def run_all(returns: pd.Series) -> dict[str, pd.DataFrame]:
    """All five models on the same out-of-sample dates."""
    garch_t, fhs, _ = garch_models(returns)
    return {
        "Historical simulation": historical_simulation(returns),
        "Normal (rolling)": normal_rolling(returns),
        "EWMA normal": ewma_normal(returns),
        "GARCH-t": garch_t,
        "Filtered HS": fhs,
    }
