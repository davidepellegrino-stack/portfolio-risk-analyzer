import numpy as np
import pandas as pd
import pytest

from marketrisk import data


def test_usd_price_is_divided_by_eurusd():
    idx = pd.date_range("2024-01-01", periods=2)
    prices = pd.DataFrame({"US": [110.0, 121.0], "EU": [50.0, 55.0]}, index=idx)
    eur = data.to_base_currency(prices, {"US": "USD", "EU": "EUR"}, pd.Series([1.10, 1.10], index=idx))
    assert eur["US"].tolist() == pytest.approx([100.0, 110.0])
    assert eur["EU"].tolist() == [50.0, 55.0]


def test_eur_investor_bears_fx_risk():
    # The USD price is flat and the dollar loses 10% against the euro: the EUR return is -9.09%.
    idx = pd.date_range("2024-01-01", periods=2)
    eur = data.to_base_currency(pd.DataFrame({"US": [100.0, 100.0]}, index=idx), {"US": "USD"},
                                pd.Series([1.00, 1.10], index=idx))
    assert data.simple_returns(eur)["US"].iloc[0] == pytest.approx(1 / 1.10 - 1)


def test_portfolio_return_is_weighted_sum_of_simple_returns():
    # 50/50 in an asset that gains 10% and one that loses 10% is exactly flat.
    # With log returns the same sum would give -0.5%, which is wrong.
    r = pd.DataFrame({"A": [0.10], "B": [-0.10]})
    assert data.portfolio_returns(r, [0.5, 0.5]).iloc[0] == pytest.approx(0.0)
    assert 0.5 * np.log(1.10) + 0.5 * np.log(0.90) == pytest.approx(-0.005, abs=1e-4)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        data.portfolio_returns(pd.DataFrame({"A": [0.01], "B": [0.02]}), [0.5, 0.6])


def test_closed_market_keeps_last_price_and_calendar_is_respected():
    idx = pd.date_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4.0]}, index=idx)
    calendar = pd.Series([1.0, 1.0, np.nan, 1.0], index=idx)   # calendar market shut on day 3
    out = data.align_calendar(prices, calendar)
    assert list(out.index) == [idx[0], idx[1], idx[3]]
    assert out["A"].tolist() == [1.0, 1.0, 4.0]
