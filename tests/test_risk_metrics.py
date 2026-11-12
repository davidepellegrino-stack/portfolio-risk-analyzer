import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_metrics import (
    correlation_matrix, cvar_historical, cvar_parametric, max_drawdown,
    rolling_volatility, var_historical, var_monte_carlo, var_parametric,
)

CONFS = [0.90, 0.95, 0.99]


def test_historical_var_matches_known_quantile():
    # Evenly spaced returns on [-10%, +10%]: the 5% quantile is exactly -9%.
    r = pd.Series(np.linspace(-0.10, 0.10, 1001))
    assert var_historical(r, 0.95) == pytest.approx(0.09)


def test_parametric_var_matches_closed_form(portfolio_returns):
    z = stats.norm.ppf(0.05)
    expected = -(portfolio_returns.mean() + z * portfolio_returns.std())
    assert var_parametric(portfolio_returns, 0.95) == pytest.approx(expected)


def test_parametric_var_zero_mean_is_z_times_sigma():
    r = pd.Series([-0.02, 0.02] * 500)  # mean 0, constant dispersion
    assert var_parametric(r, 0.99) == pytest.approx(stats.norm.ppf(0.99) * r.std())


@pytest.mark.parametrize("func", [var_historical, var_parametric])
def test_var_is_monotonic_in_confidence(portfolio_returns, func):
    values = [func(portfolio_returns, c) for c in CONFS]
    assert values == sorted(values)
    assert values[0] < values[-1]


@pytest.mark.parametrize("conf", CONFS)
def test_cvar_is_at_least_var(portfolio_returns, conf):
    assert cvar_historical(portfolio_returns, conf) >= var_historical(portfolio_returns, conf)
    assert cvar_parametric(portfolio_returns, conf) >= var_parametric(portfolio_returns, conf)


def test_parametric_es_for_standard_normal():
    # ES_95 of a N(0, 1) loss is phi(1.645) / 0.05 = 2.0627
    r = pd.Series(np.random.default_rng(0).standard_normal(200_000))
    r = (r - r.mean()) / r.std()  # force mean 0, sigma 1 exactly
    assert cvar_parametric(r, 0.95) == pytest.approx(2.0627, abs=1e-3)


def test_historical_cvar_is_mean_of_tail():
    r = pd.Series(np.linspace(-0.10, 0.10, 1001))
    var = var_historical(r, 0.95)
    tail = r[r <= -var]
    assert cvar_historical(r, 0.95) == pytest.approx(-tail.mean())
    assert cvar_historical(r, 0.95) == pytest.approx(0.095, abs=1e-3)


def test_monte_carlo_var_close_to_scaled_parametric(asset_returns):
    w = np.ones(3) / 3
    horizon = 10
    mc = var_monte_carlo(asset_returns, w, confidence=0.95, n_sims=4000, horizon=horizon)
    port = asset_returns @ w
    analytic = -(horizon * port.mean() + stats.norm.ppf(0.05) * port.std() * np.sqrt(horizon))
    assert mc == pytest.approx(analytic, rel=0.10)


def test_monte_carlo_var_is_reproducible(asset_returns):
    w = np.ones(3) / 3
    a = var_monte_carlo(asset_returns, w, n_sims=300, horizon=5)
    b = var_monte_carlo(asset_returns, w, n_sims=300, horizon=5)
    assert a == b


def test_max_drawdown_on_known_path():
    # Wealth path: 1 -> 1.10 -> 0.88 -> 0.99 -> 1.20
    prices = pd.Series([1.0, 1.10, 0.88, 0.99, 1.20],
                       index=pd.date_range("2024-01-01", periods=5))
    r = prices.pct_change().dropna()
    # Start from the first return, so cumulative wealth is 1.10, 0.88, 0.99, 1.20 (x 1.0).
    dd = max_drawdown(r)
    assert dd["max_drawdown"] == pytest.approx(-0.20)
    assert dd["peak_date"] == pd.Timestamp("2024-01-02")
    assert dd["trough_date"] == pd.Timestamp("2024-01-03")
    assert dd["recovery_date"] == pd.Timestamp("2024-01-05")


def test_max_drawdown_without_recovery_is_none():
    r = pd.Series([0.10, -0.30, 0.05], index=pd.date_range("2024-01-01", periods=3))
    assert max_drawdown(r)["recovery_date"] is None


def test_max_drawdown_of_monotonic_gains_is_zero():
    r = pd.Series([0.01] * 20, index=pd.date_range("2024-01-01", periods=20))
    assert max_drawdown(r)["max_drawdown"] == 0


def test_rolling_volatility_is_annualised():
    r = pd.Series([0.01, -0.01] * 100)
    vol = rolling_volatility(r, window=20).dropna()
    expected = r.iloc[-20:].std() * np.sqrt(252)
    assert vol.iloc[-1] == pytest.approx(expected)
    assert len(vol) == len(r) - 19


def test_correlation_matrix_is_valid(asset_returns):
    c = correlation_matrix(asset_returns)
    assert np.allclose(np.diag(c), 1.0)
    assert np.allclose(c, c.T)
    assert ((c >= -1) & (c <= 1)).all().all()
