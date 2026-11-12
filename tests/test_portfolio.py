import numpy as np
import pytest

from portfolio import (
    equal_weight_portfolio, max_sharpe_portfolio, min_variance_portfolio,
    portfolio_performance, simulate_random_portfolios, summarise_portfolios,
)


def _weights(p):
    return np.array(list(p["weights"].values()))


def test_equal_weight_sums_to_one(asset_returns):
    p = equal_weight_portfolio(asset_returns)
    assert _weights(p).sum() == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(_weights(p), 1 / 3, atol=1e-3)


@pytest.mark.parametrize("builder", [max_sharpe_portfolio, min_variance_portfolio])
def test_optimised_weights_are_long_only_and_fully_invested(asset_returns, builder):
    w = _weights(builder(asset_returns))
    assert w.sum() == pytest.approx(1.0, abs=1e-3)
    assert (w >= -1e-6).all() and (w <= 1 + 1e-6).all()


def test_min_variance_matches_analytic_solution(asset_returns):
    # Unconstrained GMV: w = S^-1 1 / (1' S^-1 1). All weights are positive for
    # this covariance, so the long-only optimum must coincide with it.
    s = asset_returns.cov().values
    inv_one = np.linalg.solve(s, np.ones(3))
    analytic = inv_one / inv_one.sum()
    assert (analytic > 0).all()
    w = _weights(min_variance_portfolio(asset_returns))
    assert np.allclose(w, analytic, atol=1e-3)


def test_min_variance_has_lowest_volatility(asset_returns):
    mv = min_variance_portfolio(asset_returns)["volatility"]
    ew = equal_weight_portfolio(asset_returns)["volatility"]
    ms = max_sharpe_portfolio(asset_returns)["volatility"]
    assert mv <= ew + 1e-9
    assert mv <= ms + 1e-9


def test_max_sharpe_beats_equal_weight(asset_returns):
    assert (max_sharpe_portfolio(asset_returns)["sharpe"]
            >= equal_weight_portfolio(asset_returns)["sharpe"] - 1e-9)


def test_random_portfolios_are_valid_and_below_the_optimum(asset_returns):
    sims = simulate_random_portfolios(asset_returns, n_portfolios=200)
    wcols = [c for c in sims.columns if c.startswith("w_")]
    assert np.allclose(sims[wcols].sum(axis=1), 1.0)
    assert (sims[wcols] >= 0).all().all()
    assert sims["sharpe"].max() <= max_sharpe_portfolio(asset_returns)["sharpe"] + 1e-6
    assert sims["volatility"].min() >= min_variance_portfolio(asset_returns)["volatility"] - 1e-6


def test_portfolio_performance_single_asset(asset_returns):
    r, v, _ = portfolio_performance(np.array([1.0, 0.0, 0.0]), asset_returns)
    assert r == pytest.approx(asset_returns["A"].mean() * 252)
    assert v == pytest.approx(asset_returns["A"].std() * np.sqrt(252))


def test_summary_table_has_three_portfolios(asset_returns):
    table = summarise_portfolios(asset_returns)
    assert list(table["Portfolio"]) == ["Equal Weight", "Max Sharpe", "Min Variance"]
