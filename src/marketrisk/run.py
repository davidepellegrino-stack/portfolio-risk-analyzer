"""End-to-end run: python -m marketrisk.run"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import attribution, backtest, config as cfg, data, models, plots, stress


def main(refresh: bool = False) -> None:
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prices, asset_returns = data.load_market_data(refresh)
    weights = np.full(len(cfg.TICKERS), 1 / len(cfg.TICKERS))
    port = data.portfolio_returns(asset_returns, weights)
    print(f"{len(port)} daily returns, {port.index[0].date()} to {port.index[-1].date()}, in {cfg.BASE_CCY}")

    forecasts = models.run_all(port)
    oos = next(iter(forecasts.values())).index
    print(f"Out-of-sample: {len(oos)} days from {oos[0].date()}")

    results = backtest.backtest_all(port, forecasts)
    results.to_csv(cfg.TABLES_DIR / "backtest_summary.csv", float_format="%.4f")
    pd.concat(forecasts, axis=1).to_csv(cfg.TABLES_DIR / "forecasts.csv", float_format="%.6f")
    _, _, garch_params = models.garch_models(port)
    garch_params.to_csv(cfg.TABLES_DIR / "garch_parameters.csv", float_format="%.4f")

    scen = stress.historical_scenarios(asset_returns, weights, cfg.STRESS_SCENARIOS)
    worst = stress.worst_losses(port)
    svar = pd.Series(stress.stressed_var(port), name="value")
    comp_var = attribution.component_var(asset_returns.loc[oos], weights)
    comp_es = attribution.component_es(asset_returns.loc[oos], weights)
    scen.to_csv(cfg.TABLES_DIR / "stress_scenarios.csv", float_format="%.2f")
    worst.to_csv(cfg.TABLES_DIR / "worst_losses.csv", float_format="%.2f")
    svar.to_csv(cfg.TABLES_DIR / "stressed_var.csv")
    comp_var.join(comp_es.drop(columns="weight_%")).to_csv(
        cfg.TABLES_DIR / "risk_contributions.csv", float_format="%.3f")

    plots.plot_var_backtests(port, forecasts, cfg.FIGURES_DIR / "var_backtests.png")
    plots.plot_traffic_light(port, forecasts, cfg.FIGURES_DIR / "traffic_light.png")
    plots.plot_risk_contributions(comp_var, comp_es, cfg.FIGURES_DIR / "risk_contributions.png")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\nBacktest\n", results.T, "\n\nScenarios\n", scen, "\n\nWorst losses\n", worst,
          "\n\nStressed VaR\n", svar, "\n\nContributions\n", comp_var.join(comp_es.drop(columns="weight_%")),
          "\n\nGARCH params (first, last)\n", garch_params.iloc[[0, -1]], sep="")


if __name__ == "__main__":
    import sys
    main(refresh="--refresh" in sys.argv)
