import numpy as np
import pandas as pd
import pytest
from scipy import stats

from marketrisk import attribution, backtest, models, stress


def test_kupiec_is_zero_when_hit_rate_equals_alpha():
    lr, p = backtest.kupiec_pof(10, 1000, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-12) and p == pytest.approx(1.0)


def test_kupiec_against_hand_calculation():
    # T = 250, x = 9, alpha = 1%: LR = 2[9 ln(0.036/0.01) + 241 ln(0.964/0.99)] = 10.23
    lr, p = backtest.kupiec_pof(9, 250, 0.01)
    assert lr == pytest.approx(2 * (9 * np.log(0.036 / 0.01) + 241 * np.log(0.964 / 0.99)))
    assert lr == pytest.approx(10.23, abs=0.01) and p < 0.01


def test_kupiec_handles_zero_exceedances():
    lr, _ = backtest.kupiec_pof(0, 250, 0.01)
    assert lr == pytest.approx(-2 * 250 * np.log(0.99))


def test_independence_flags_clustering_but_not_spread_hits():
    spread = np.zeros(1000, dtype=int)
    spread[::100] = 1                       # 10 isolated exceedances
    clustered = np.zeros(1000, dtype=int)
    clustered[500:510] = 1                  # the same 10, back to back
    assert backtest.christoffersen_independence(spread)[1] > 0.5
    assert backtest.christoffersen_independence(clustered)[1] < 1e-6
    # Same count, so Kupiec cannot tell them apart; conditional coverage can.
    assert backtest.conditional_coverage(spread, 0.01)[1] > 0.5
    assert backtest.conditional_coverage(clustered, 0.01)[1] < 1e-6


def test_independence_against_hand_calculation():
    # 0 0 1 1 0 0 0 1 0 0 -> n00 = 4, n01 = 2, n10 = 2, n11 = 1
    hits = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    pi01, pi11, pi = 2 / 6, 1 / 3, 3 / 9
    ll_alt = 4 * np.log(1 - pi01) + 2 * np.log(pi01) + 2 * np.log(1 - pi11) + np.log(pi11)
    ll_null = 6 * np.log(1 - pi) + 3 * np.log(pi)
    assert backtest.christoffersen_independence(hits)[0] == pytest.approx(-2 * (ll_null - ll_alt), abs=1e-12)


@pytest.mark.parametrize("n, zone", [(0, "green"), (4, "green"), (5, "yellow"),
                                     (9, "yellow"), (10, "red"), (25, "red")])
def test_traffic_light_reproduces_basel_table(n, zone):
    assert backtest.traffic_light(n, 250, 0.01) == zone


def test_correct_model_passes_every_test(normal_returns):
    """True model: N(0, 1%). Its VaR and ES forecasts must not be rejected."""
    r = normal_returns
    f = pd.DataFrame({"var_99": 0.01 * models.normal_var(0.01),
                      "var_975": 0.01 * models.normal_var(0.025),
                      "es_975": 0.01 * models.normal_es(0.025)}, index=r.index)
    res = backtest.backtest_model(r, f)
    assert res["kupiec_p"] > 0.05 and res["cond_coverage_p"] > 0.05
    assert abs(res["es_z2"]) < 0.35


def test_z2_detects_understated_es(normal_returns):
    r = normal_returns
    half = pd.Series(0.5 * 0.01 * models.normal_es(0.025), index=r.index)
    var = pd.Series(0.01 * models.normal_var(0.025), index=r.index)
    assert backtest.es_z2(r, var, half, 0.025) < -0.7


def test_scoring_rules_prefer_the_true_model(normal_returns):
    r = normal_returns
    def forecast(scale):
        return (pd.Series(scale * 0.01 * models.normal_var(0.025), index=r.index),
                pd.Series(scale * 0.01 * models.normal_es(0.025), index=r.index))
    true_fz = backtest.fz0_loss(r, *forecast(1.0), 0.025)
    assert true_fz < backtest.fz0_loss(r, *forecast(0.6), 0.025)
    assert true_fz < backtest.fz0_loss(r, *forecast(1.6), 0.025)
    assert true_fz < 0                      # ln(ES) dominates for daily returns
    true_pb = backtest.pinball_loss(r, forecast(1.0)[0], 0.025)
    assert true_pb < backtest.pinball_loss(r, forecast(0.6)[0], 0.025)
    assert true_pb < backtest.pinball_loss(r, forecast(1.6)[0], 0.025)


def _assets():
    rng = np.random.default_rng(3)
    vols = np.array([0.01, 0.02, 0.03])
    corr = np.array([[1, .5, .2], [.5, 1, .4], [.2, .4, 1]])
    return pd.DataFrame(rng.multivariate_normal(np.zeros(3), np.outer(vols, vols) * corr, 5000),
                        columns=list("ABC"))


def test_component_var_adds_up_to_total_var():
    a, w = _assets(), np.array([0.5, 0.3, 0.2])
    comp = attribution.component_var(a, w, 0.01)
    total = -stats.norm.ppf(0.01) * np.sqrt(w @ a.cov().to_numpy() @ w)
    assert comp["component_var_%"].sum() == pytest.approx(100 * total)
    assert comp["share_of_var_%"].sum() == pytest.approx(100)


def test_component_es_adds_up_to_portfolio_es():
    a, w = _assets(), np.array([0.5, 0.3, 0.2])
    port = (a * w).sum(axis=1)
    _, es = models.empirical_tail(port.to_numpy(), 0.025)
    assert attribution.component_es(a, w, 0.025)["component_es_%"].sum() == pytest.approx(100 * es)


def test_scenario_compounds_buy_and_hold():
    idx = pd.date_range("2024-01-01", periods=3)
    a = pd.DataFrame({"A": [-0.10, -0.10, 0.0], "B": [0.0, 0.0, 0.0]}, index=idx)
    out = stress.historical_scenarios(a, np.array([0.5, 0.5]), {"s": ("2024-01-01", "2024-01-02")})
    assert out.loc["s", "portfolio_return_%"] == pytest.approx(100 * 0.5 * (0.9 * 0.9 - 1))
    assert out.loc["s", "worst_asset"] == "A" and out.loc["s", "days"] == 2


def test_stressed_var_is_at_least_current_var(normal_returns):
    s = stress.stressed_var(normal_returns)
    assert s["stressed_var_%"] >= s["current_var_%"] > 0
