import numpy as np
import pandas as pd
import pytest
from scipy import stats

from marketrisk import attribution, backtest, models, simulate, stress


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


def test_dq_with_constant_only_is_the_score_test_of_coverage():
    # No lags and a flat VaR leave only the intercept: DQ = T (pi - a)^2 / (a (1 - a)), chi2(1).
    hits = np.zeros(1000, dtype=int)
    hits[::50] = 1                          # 20 exceedances, 10 expected
    dq, p = backtest.dynamic_quantile(hits, np.full(1000, 0.02), 0.01, lags=0)
    assert dq == pytest.approx(1000 * (0.02 - 0.01) ** 2 / (0.01 * 0.99))
    assert p == pytest.approx(stats.chi2.sf(dq, 1))


def test_dq_sees_dependence_at_two_days_that_christoffersen_misses():
    hits = np.zeros(2000, dtype=int)
    starts = np.arange(100, 2000, 200)
    hits[starts] = hits[starts + 2] = 1     # 20 exceedances in pairs two days apart
    var = np.random.default_rng(0).uniform(0.02, 0.03, 2000)
    assert backtest.christoffersen_independence(hits)[1] > 0.4
    assert backtest.dynamic_quantile(hits, var, 0.01)[1] < 1e-6


def test_dq_flags_exceedances_that_depend_on_the_var_level():
    var = np.where(np.arange(2000) < 1000, 0.01, 0.03)
    hits = np.zeros(2000, dtype=int)
    hits[25:1000:50] = 1                    # all 20 fall on low-VaR days: right count overall
    assert backtest.kupiec_pof(20, 2000, 0.01)[1] > 0.99
    assert backtest.dynamic_quantile(hits, var, 0.01, lags=1)[1] < 0.01


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
    assert res["dq_p"] > 0.05               # flat VaR: the collinear regressor is dropped
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


def _normal_setup(r, vol=0.01):
    """A model that says r_t ~ N(0, vol^2): VaR, ES and sampler all agree."""
    sigma = np.full(len(r), vol)
    var = pd.Series(sigma * models.normal_var(0.025), index=r.index)
    es = pd.Series(sigma * models.normal_es(0.025), index=r.index)
    return var, es, simulate.normal_sampler(sigma)


def test_z2_simulated_p_value_accepts_true_model_and_rejects_low_es(normal_returns):
    r = normal_returns
    for scale, low, high in [(0.01, 0.05, 1.0), (0.007, 0.0, 0.01)]:
        var, es, sampler = _normal_setup(r, scale)   # data have vol 1%
        z = backtest.es_z2(r, var, es, 0.025)
        p, crit = backtest.es_z2_test(var, es, 0.025, sampler, z, n_sim=2000)
        assert low <= p <= high and crit < 0


def test_z2_p_value_bounds_and_reproducibility(normal_returns):
    var, es, sampler = _normal_setup(normal_returns.iloc[:500])
    def run(z_obs, chunk=100):
        return backtest.es_z2_test(var, es, 0.025, sampler, z_obs, n_sim=999, chunk=chunk)
    assert run(-100.0)[0] == pytest.approx(1 / 1000)   # never zero
    assert run(100.0)[0] == pytest.approx(1.0)
    assert run(-0.2) == run(-0.2) == run(-0.2, chunk=999)


def test_samplers_reproduce_the_forecast_tail():
    """Each sampler must breach its own 97.5% VaR 2.5% of the time, with mean breach ES."""
    nu, sigma = np.full(4, 6.0), np.array([0.01, 0.02, 0.01, 0.03])
    r = simulate.student_sampler(sigma, nu)(np.random.default_rng(1), 200_000)
    breach = r < -sigma * models.student_var(0.025, 6.0)
    assert breach.mean(axis=0) == pytest.approx(np.full(4, 0.025), abs=0.002)
    assert (-r / sigma)[breach].mean() == pytest.approx(models.student_es(0.025, 6.0), rel=0.03)


def test_empirical_sampler_draws_each_day_from_its_own_scaled_pool():
    pools = np.array([[1.0, 2.0], [10.0, 20.0]])
    r = simulate.empirical_sampler(pools, np.array([1.0, 0.5]))(np.random.default_rng(0), 500)
    assert set(r[:, 0]) == {1.0, 2.0} and set(r[:, 1]) == {5.0, 10.0}


def test_model_samplers_use_only_past_returns_and_the_refit_in_use(normal_returns):
    r = normal_returns.iloc[:1300]
    residuals: dict = {}
    forecasts, params = models.run_all(r, residuals=residuals)
    assert list(residuals) == list(params.index)
    samplers = simulate.model_samplers(r, forecasts, params, residuals)
    assert set(samplers) == set(forecasts)
    hs = samplers["Historical simulation"](np.random.default_rng(0), 200)
    assert np.isin(hs[:, 0], r.iloc[500:1000]).all()   # the 500 days before day 1000
    res = backtest.backtest_all(r, forecasts, samplers)
    assert res["es_z2_p"].between(0, 1).all()


def test_dq_monte_carlo_p_value_bounds_and_reproducibility():
    var = np.random.default_rng(0).uniform(0.02, 0.03, 500)
    def run(dq_obs):
        return backtest.dq_test_mc(var, 0.01, dq_obs, n_sim=199)
    assert run(1e12)[0] == pytest.approx(1 / 200)      # never zero
    assert run(0.0)[0] == pytest.approx(1.0)
    assert run(5.0) == run(5.0)


def test_dq_monte_carlo_agrees_with_the_data():
    """No exceedance pattern is rejected; exceedances in pairs two days apart are."""
    var = np.random.default_rng(0).uniform(0.02, 0.03, 2000)
    paired, random = np.zeros(2000, dtype=int), np.zeros(2000, dtype=int)
    starts = np.arange(100, 2000, 200)
    paired[starts] = paired[starts + 2] = 1
    random[np.random.default_rng(5).choice(2000, 20, replace=False)] = 1
    for hits, low, high in [(paired, 0.0, 0.01), (random, 0.05, 1.0)]:
        dq = backtest.dynamic_quantile(hits, var, 0.01)[0]
        assert low <= backtest.dq_test_mc(var, 0.01, dq, n_sim=999)[0] <= high
