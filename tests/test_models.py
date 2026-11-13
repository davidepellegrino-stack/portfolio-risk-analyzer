import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from marketrisk import models


def test_normal_tail_constants():
    assert models.normal_var(0.01) == pytest.approx(2.3263, abs=1e-4)
    assert models.normal_es(0.025) == pytest.approx(2.3378, abs=1e-4)


@pytest.mark.parametrize("nu", [4.0, 6.5, 12.0])
def test_student_es_matches_numerical_integration(nu):
    # ES = -(1/alpha) * integral of x f(x) below the alpha-quantile, for the unit-variance t.
    alpha, scale = 0.025, np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(alpha, nu)
    integral, _ = integrate.quad(lambda x: x * stats.t.pdf(x, nu), -np.inf, q)
    assert models.student_es(alpha, nu) == pytest.approx(-integral / alpha * scale, rel=1e-6)
    assert models.student_es(alpha, nu) > models.student_var(alpha, nu)


def test_student_tail_converges_to_normal():
    assert models.student_var(0.01, 1e6) == pytest.approx(models.normal_var(0.01), abs=1e-4)
    assert models.student_es(0.025, 1e6) == pytest.approx(models.normal_es(0.025), abs=1e-4)


def test_empirical_tail_on_a_known_grid():
    sample = np.linspace(-0.10, 0.10, 1001)        # 5% quantile is exactly -9%
    var, es = models.empirical_tail(sample, 0.05)
    assert var == pytest.approx(0.09)
    assert es == pytest.approx(0.095, abs=1e-4)


def test_ewma_recursion_by_hand():
    r = pd.Series([0.01, -0.01, 0.03, 0.0])
    sigma = models.ewma_volatility(r, lam=0.94, init=2)
    v0 = np.var([0.01, -0.01])
    assert sigma.iloc[2] == pytest.approx(np.sqrt(v0))                       # uses r[0:2] only
    assert sigma.iloc[3] == pytest.approx(np.sqrt(0.94 * v0 + 0.06 * 0.03**2))


@pytest.mark.parametrize("model", [models.historical_simulation, models.normal_rolling,
                                   models.ewma_normal])
def test_no_look_ahead(normal_returns, model):
    """Changing returns from day 1500 onwards must not change any forecast up to day 1500."""
    r = normal_returns.iloc[:1600]
    shocked = r.copy()
    shocked.iloc[1500:] = -0.20
    a, b = model(r), model(shocked)
    cut = r.index[1500]
    pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])
    assert not a.loc[cut:].iloc[1:].equals(b.loc[cut:].iloc[1:])


def test_garch_has_no_look_ahead_and_sane_output(normal_returns):
    r = normal_returns.iloc[:1100]
    shocked = r.copy()
    shocked.iloc[1050:] = -0.20
    g_a, f_a, params = models.garch_models(r, refit=50)
    g_b, f_b, _ = models.garch_models(shocked, refit=50)
    cut = r.index[1050]
    pd.testing.assert_frame_equal(g_a.loc[:cut], g_b.loc[:cut])
    pd.testing.assert_frame_equal(f_a.loc[:cut], f_b.loc[:cut])
    assert (g_a > 0).all().all() and (f_a > 0).all().all()
    assert (g_a["es_975"] > g_a["var_975"]).all() and (f_a["es_975"] > f_a["var_975"]).all()
    assert (params["alpha"] + params["beta"] < 1).all()                       # stationary
    # On N(0, 1%) data every model's 99% VaR should sit near 2.326 x 1%.
    assert g_a["var_99"].mean() == pytest.approx(0.0233, rel=0.15)


def test_all_models_share_forecast_dates(normal_returns):
    r = normal_returns.iloc[:1060]
    hs, nr, ew = models.historical_simulation(r), models.normal_rolling(r), models.ewma_normal(r)
    assert hs.index.equals(nr.index) and nr.index.equals(ew.index)
    assert len(hs) == 60 and not hs.isna().any().any() and not nr.isna().any().any()
