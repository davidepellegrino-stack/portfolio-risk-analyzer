"""Backtests of VaR and ES forecasts against realised returns.

VaR  : Kupiec (1995) unconditional coverage, Christoffersen (1998) independence
       and conditional coverage, Engle-Manganelli (2004) dynamic quantile,
       Basel traffic light.
ES   : Acerbi and Szekely (2014) Z2 statistic.
Rank : quantile (pinball) loss for VaR, Fissler-Ziegel FZ0 loss for (VaR, ES).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy   # xlogy(0, 0) = 0, the convention the LR tests need

from .simulate import Sampler


def exceedances(returns: pd.Series, var: pd.Series) -> pd.Series:
    """True on days when the loss was larger than the VaR forecast."""
    return returns < -var


def kupiec_pof(n_exceed: int, n_obs: int, alpha: float) -> tuple[float, float]:
    """Proportion-of-failures test. H0: P(exceedance) = alpha.

    LR = -2 ln[ L(alpha) / L(pi_hat) ], pi_hat = x / T, asymptotically chi2(1).
    Two-sided: it rejects a model with too many OR too few exceedances.
    """
    x, T = n_exceed, n_obs
    pi = x / T
    ll_null = xlogy(T - x, 1 - alpha) + xlogy(x, alpha)
    ll_alt = xlogy(T - x, 1 - pi) + xlogy(x, pi)
    lr = -2.0 * (ll_null - ll_alt)
    return lr, stats.chi2.sf(lr, 1)


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    """Independence test. H0: an exceedance today does not change the probability
    of one tomorrow. Clustered exceedances mean the model is slow to react.

    Counts transitions n_ij (state i yesterday, state j today) and compares a
    first-order Markov chain (pi_01 != pi_11) with an i.i.d. sequence. chi2(1).
    """
    h = np.asarray(hits).astype(int)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
              + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    lr = -2.0 * (ll_null - ll_alt)
    return lr, stats.chi2.sf(lr, 1)


def conditional_coverage(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Christoffersen joint test: right number of exceedances AND no clustering.
    LR_cc = LR_pof + LR_ind, chi2(2)."""
    h = np.asarray(hits).astype(int)
    lr = kupiec_pof(int(h.sum()), len(h), alpha)[0] + christoffersen_independence(h)[0]
    return lr, stats.chi2.sf(lr, 2)


def dynamic_quantile(hits: np.ndarray, var: np.ndarray, alpha: float,
                     lags: int = 4) -> tuple[float, float]:
    """Engle and Manganelli (2004) out-of-sample dynamic quantile (DQ) test.

    With Hit_t = I_t - alpha, a correct model makes Hit_t unpredictable from
    anything known at t-1. Regress it by OLS on X = [1, Hit_{t-1} ... Hit_{t-lags},
    VaR_t]. H0: all coefficients are zero.

        DQ = b' X'X b / (alpha * (1 - alpha))  ~  chi2(number of regressors)

    It sees what the Christoffersen test cannot: dependence beyond one day, and
    exceedances that are more likely when the VaR forecast is high (or low).
    """
    hit = np.asarray(hits).astype(float) - alpha
    var = np.asarray(var, dtype=float)
    y = hit[lags:]
    X = np.column_stack([np.ones(len(y))]
                        + [hit[lags - k:-k] for k in range(1, lags + 1)]
                        + [var[lags:]])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta                       # b' X'X b is the squared length of X b
    dq = float(fitted @ fitted / (alpha * (1 - alpha)))
    return dq, float(stats.chi2.sf(dq, rank))   # rank < columns if a regressor is constant


def dq_test_mc(var: np.ndarray, alpha: float, dq_obs: float, lags: int = 4,
               n_sim: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Monte Carlo p-value of the DQ statistic.

    With about 20 exceedances the chi-squared limit is a poor guide: one pair of
    exceedances a few days apart moves DQ a lot. Under H0 the hits are i.i.d.
    Bernoulli(alpha) whatever the VaR is, so they can be drawn directly. Each
    draw is run through the same regression against the observed VaR series:

        p = (1 + #{DQ_sim >= DQ_obs}) / (n_sim + 1)

    Returns (p-value, simulated 5% critical value of DQ).
    """
    rng = np.random.default_rng(seed)
    sims = np.array([dynamic_quantile(rng.random(len(var)) < alpha, var, alpha, lags)[0]
                     for _ in range(n_sim)])
    # The statistic is discrete in the hits: count equal values despite rounding noise.
    p = (1 + np.count_nonzero(sims >= dq_obs * (1 - 1e-9))) / (n_sim + 1)
    return float(p), float(np.quantile(sims, 0.95))


def traffic_light(n_exceed: int, n_obs: int = 250, alpha: float = 0.01) -> str:
    """Basel zone from the binomial CDF of the exceedance count under a correct model.

    green  : CDF < 95%
    yellow : 95% <= CDF < 99.99%
    red    : CDF >= 99.99%
    With 250 days at 99% this gives the Basel table: 0-4 green, 5-9 yellow, 10+ red.
    """
    cdf = stats.binom.cdf(n_exceed, n_obs, alpha)
    if cdf < 0.95:
        return "green"
    return "yellow" if cdf < 0.9999 else "red"


def rolling_exceedances(hits: pd.Series, window: int = 250) -> pd.Series:
    """Exceedances over the trailing `window` days: what a supervisor sees each day."""
    return hits.astype(int).rolling(window).sum().dropna()


def es_z2(returns: pd.Series, var: pd.Series, es: pd.Series, alpha: float) -> float:
    """Acerbi-Szekely second test of ES.

    Z2 = sum_t( r_t * I_t / (T * alpha * ES_t) ) + 1,  I_t = 1 on a VaR exceedance.

    E[Z2] = 0 if the model is right. Z2 < 0 means losses in the tail were more
    frequent and/or larger than forecast. Acerbi and Szekely show that the 5%
    critical value is close to -0.7 for a wide range of distributions, so the
    test can be read without simulating from the model.
    """
    hits = exceedances(returns, var)
    return float((returns[hits] / (len(returns) * alpha * es[hits])).sum() + 1.0)


def es_z2_test(var: pd.Series, es: pd.Series, alpha: float, sampler: Sampler,
               z2_obs: float, n_sim: int = 10_000, seed: int = 0,
               chunk: int = 1_000) -> tuple[float, float]:
    """Simulated Z2 test. H0: r_t is drawn from the model's predictive distribution.

    Each simulation draws one return per day from that distribution and
    recomputes Z2 with the same VaR and ES forecasts. The test is one-sided
    (only Z2 far below zero means risk was understated):

        p = (1 + #{Z2_sim <= Z2_obs}) / (n_sim + 1)

    Returns (p-value, simulated 5% critical value of Z2).
    """
    v, e = var.to_numpy(), es.to_numpy()
    rng = np.random.default_rng(seed)
    sims = []
    for done in range(0, n_sim, chunk):
        r = sampler(rng, min(chunk, n_sim - done))
        sims.append(np.where(r < -v, r / e, 0.0).sum(axis=1) / (len(v) * alpha) + 1.0)
    z = np.concatenate(sims)
    p = (1 + np.count_nonzero(z <= z2_obs)) / (n_sim + 1)
    return float(p), float(np.quantile(z, 0.05))


def pinball_loss(returns: pd.Series, var: pd.Series, alpha: float) -> float:
    """Average quantile loss of the VaR forecast (lower is better). The quantile
    is the forecast that minimises this loss, so it gives a fair ranking of VaR models."""
    q = -var
    return float(((returns - q) * (alpha - (returns < q))).mean())


def fz0_loss(returns: pd.Series, var: pd.Series, es: pd.Series, alpha: float) -> float:
    """Average FZ0 loss (Patton, Ziegel and Chen, 2019), lower is better.

    ES alone cannot be scored, but the pair (VaR, ES) can (Fissler and Ziegel,
    2016). With v = -VaR and e = -ES, both negative:

        L = -(1 / (alpha * e)) * 1{r <= v} * (v - r) + v / e + ln(-e) - 1
    """
    v, e = -var, -es
    hit = (returns <= v).astype(float)          # cast first: minus on a boolean Series means NOT
    loss = -hit * (v - returns) / (alpha * e) + v / e + np.log(-e) - 1.0
    return float(loss.mean())


def backtest_model(returns: pd.Series, forecast: pd.DataFrame,
                   var_alpha: float = 0.01, es_alpha: float = 0.025,
                   sampler: Sampler | None = None) -> dict:
    """Every statistic for one model, on the dates where it has a forecast.

    With a `sampler` (see simulate.py) Z2 and DQ also get simulated p-values."""
    r = returns.loc[forecast.index]
    hits = exceedances(r, forecast["var_99"])
    x, T = int(hits.sum()), len(hits)
    roll = rolling_exceedances(hits)
    zones = roll.map(lambda n: traffic_light(int(n)))
    stats_by_name = {
        "days": T,
        "exceedances": x,
        "expected": round(T * var_alpha, 1),
        "hit_rate_%": round(100 * x / T, 2),
        "kupiec_p": kupiec_pof(x, T, var_alpha)[1],
        "independence_p": christoffersen_independence(hits.to_numpy())[1],
        "cond_coverage_p": conditional_coverage(hits.to_numpy(), var_alpha)[1],
        "dq_p": dynamic_quantile(hits.to_numpy(), forecast["var_99"].to_numpy(), var_alpha)[1],
        "max_exceed_250d": int(roll.max()),
        "green_%": round(100 * (zones == "green").mean(), 1),
        "yellow_%": round(100 * (zones == "yellow").mean(), 1),
        "red_%": round(100 * (zones == "red").mean(), 1),
        "es_z2": es_z2(r, forecast["var_975"], forecast["es_975"], es_alpha),
        "pinball_loss_x1e4": 1e4 * pinball_loss(r, forecast["var_99"], var_alpha),
        "fz0_loss": fz0_loss(r, forecast["var_975"], forecast["es_975"], es_alpha),
        "mean_var_99_%": round(100 * forecast["var_99"].mean(), 3),
    }
    if sampler is not None:
        p, crit = es_z2_test(forecast["var_975"], forecast["es_975"], es_alpha,
                             sampler, stats_by_name["es_z2"])
        stats_by_name.update({"es_z2_p": p, "es_z2_crit_5%": crit})
        var_99 = forecast["var_99"].to_numpy()
        dq = dynamic_quantile(hits.to_numpy(), var_99, var_alpha)[0]
        stats_by_name["dq_p_mc"] = dq_test_mc(var_99, var_alpha, dq)[0]
    return stats_by_name


def backtest_all(returns: pd.Series, forecasts: dict[str, pd.DataFrame],
                 samplers: dict[str, Sampler] | None = None) -> pd.DataFrame:
    return pd.DataFrame({name: backtest_model(returns, f, sampler=samplers[name] if samplers else None)
                         for name, f in forecasts.items()}).T
