"""
Portfolio Risk Analyzer — main pipeline.
Runs the full analysis end to end:
  1. Fetch data
  2. Compute risk metrics (VaR, CVaR, volatility, drawdown)
  3. Optimise portfolio (efficient frontier, Max Sharpe, Min Variance)
  4. Generate all charts and summary tables
"""
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config import (TICKERS, CONFIDENCE_LEVEL, OUTPUT_TABLES,
                    OUTPUT_FIGS, DATA_DIR, ROLLING_WINDOW)
from risk_metrics import (
    var_historical, var_parametric, var_monte_carlo,
    cvar_historical, cvar_parametric,
    rolling_volatility, correlation_matrix, max_drawdown, portfolio_stats
)
from portfolio import (
    simulate_random_portfolios, max_sharpe_portfolio,
    min_variance_portfolio, equal_weight_portfolio,
    portfolio_performance, summarise_portfolios
)
from visualizations import (
    plot_cumulative_returns, plot_rolling_volatility,
    plot_correlation_heatmap, plot_var_comparison,
    plot_drawdown, plot_efficient_frontier
)


def main():
    print("=" * 65)
    print("  PORTFOLIO RISK ANALYZER")
    print("=" * 65)

    # ── 1. LOAD DATA ──────────────────────────────────────────────────
    prices_path  = os.path.join(DATA_DIR, "prices.csv")
    returns_path = os.path.join(DATA_DIR, "returns.csv")

    if not os.path.exists(prices_path):
        print("Data not found — fetching from Yahoo Finance...")
        from fetch_data import fetch
        prices, returns = fetch()
    else:
        prices  = pd.read_csv(prices_path,  index_col=0, parse_dates=True)
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(returns)} trading days "
              f"({returns.index[0].date()} to {returns.index[-1].date()})")

    returns = returns.dropna()
    tickers = list(returns.columns)
    n = len(tickers)
    eq_weights = np.ones(n) / n

    print(f"Assets: {', '.join(tickers)}\n")

    # ── 2. PORTFOLIO RETURNS ──────────────────────────────────────────
    port_returns = returns @ eq_weights

    # ── 3. RISK METRICS ──────────────────────────────────────────────
    print("Computing risk metrics...")

    var_h  = var_historical(port_returns, CONFIDENCE_LEVEL)
    var_p  = var_parametric(port_returns, CONFIDENCE_LEVEL)
    var_mc = var_monte_carlo(returns, eq_weights, CONFIDENCE_LEVEL)
    cvar_h = cvar_historical(port_returns, CONFIDENCE_LEVEL)
    cvar_p = cvar_parametric(port_returns, CONFIDENCE_LEVEL)
    stats  = portfolio_stats(port_returns)
    dd     = max_drawdown(port_returns)

    print(f"\n{'─'*50}")
    print(f"  EQUAL-WEIGHT PORTFOLIO RISK SUMMARY ({CONFIDENCE_LEVEL:.0%})")
    print(f"{'─'*50}")
    print(f"  Annualised Return:      {stats['ann_return']*100:>8.2f}%")
    print(f"  Annualised Volatility:  {stats['ann_volatility']*100:>8.2f}%")
    print(f"  Sharpe Ratio:           {stats['sharpe_ratio']:>8.4f}")
    print(f"  Skewness:               {stats['skewness']:>8.4f}")
    print(f"  Excess Kurtosis:        {stats['excess_kurtosis']:>8.4f}")
    print(f"  Max Drawdown:           {stats['max_drawdown']*100:>8.2f}%")
    print(f"{'─'*50}")
    print(f"  VaR Historical (1-day): {var_h*100:>8.4f}%")
    print(f"  VaR Parametric (1-day): {var_p*100:>8.4f}%")
    print(f"  VaR Monte Carlo (10-d): {var_mc*100:>8.4f}%")
    print(f"  CVaR Historical:        {cvar_h*100:>8.4f}%")
    print(f"  CVaR Parametric:        {cvar_p*100:>8.4f}%")
    print(f"{'─'*50}\n")

    # Save risk summary
    risk_table = pd.DataFrame([{
        "Metric": "VaR Historical (1-day)",   "Value (%)": round(var_h*100, 4)},
        {"Metric": "VaR Parametric (1-day)",  "Value (%)": round(var_p*100, 4)},
        {"Metric": "VaR Monte Carlo (10-day)","Value (%)": round(var_mc*100, 4)},
        {"Metric": "CVaR Historical",         "Value (%)": round(cvar_h*100, 4)},
        {"Metric": "CVaR Parametric",         "Value (%)": round(cvar_p*100, 4)},
        {"Metric": "Ann. Return",             "Value (%)": round(stats["ann_return"]*100, 4)},
        {"Metric": "Ann. Volatility",         "Value (%)": round(stats["ann_volatility"]*100, 4)},
        {"Metric": "Sharpe Ratio",            "Value (%)": round(stats["sharpe_ratio"], 4)},
        {"Metric": "Max Drawdown",            "Value (%)": round(stats["max_drawdown"]*100, 4)},
    ])
    risk_table.to_csv(os.path.join(OUTPUT_TABLES, "risk_summary.csv"), index=False)

    # ── 4. PORTFOLIO OPTIMISATION ─────────────────────────────────────
    print("Optimising portfolios...")
    simulated  = simulate_random_portfolios(returns)
    max_sharpe = max_sharpe_portfolio(returns)
    min_var    = min_variance_portfolio(returns)
    eq_port    = equal_weight_portfolio(returns)

    print(f"\n{'─'*50}")
    print("  PORTFOLIO COMPARISON")
    print(f"{'─'*50}")
    summary = summarise_portfolios(returns)
    print(summary[["Portfolio","Ann. Return (%)","Ann. Volatility (%)","Sharpe Ratio"]].to_string(index=False))
    print()

    summary.to_csv(os.path.join(OUTPUT_TABLES, "portfolio_comparison.csv"), index=False)

    # ── 5. VISUALISATIONS ─────────────────────────────────────────────
    print("Generating charts...")
    plot_cumulative_returns(returns)
    plot_rolling_volatility(returns, ROLLING_WINDOW)
    plot_correlation_heatmap(returns)
    plot_var_comparison(port_returns, var_h, var_p, var_mc, cvar_h)
    plot_drawdown(port_returns)
    plot_efficient_frontier(simulated, max_sharpe, min_var, eq_port)

    print(f"\nAll outputs saved to outputs/")
    print("  figures/ — 6 charts")
    print("  tables/  — risk_summary.csv, portfolio_comparison.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
