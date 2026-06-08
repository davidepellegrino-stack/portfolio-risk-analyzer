# Portfolio Risk Analyzer

A quantitative portfolio risk analysis tool built in Python. Computes Value-at-Risk using three distinct methodologies, Expected Shortfall, drawdown analysis, and mean-variance portfolio optimization with efficient frontier visualization.

Built as part of a self-directed quantitative finance project alongside M2 Financial Economics studies at Université Paris 1 Panthéon-Sorbonne.

## Features

### Risk Metrics
- **Value-at-Risk (VaR)** via three methods:
  - Historical simulation (empirical quantile)
  - Variance-covariance / parametric (normal distribution assumption)
  - Monte Carlo simulation (10-day horizon, 10,000 paths, Cholesky decomposition for correlated draws)
- **Expected Shortfall (CVaR)** — mean loss beyond the VaR threshold, both historical and parametric
- **Maximum drawdown** with peak, trough, and recovery date detection
- **Rolling volatility** (annualised, configurable window)
- **Pairwise correlation matrix**

### Portfolio Optimization
- **Efficient frontier** via Monte Carlo simulation of 10,000 random portfolios
- **Maximum Sharpe ratio portfolio** (SLSQP constrained optimization)
- **Global minimum variance portfolio**
- **Equal-weight benchmark** for comparison

### Visualizations
Six publication-quality charts saved automatically:

| Chart | Description |
|---|---|
| `cumulative_returns.png` | Growth of €1 invested in each asset |
| `rolling_volatility.png` | Annualised rolling volatility (60-day window) |
| `correlation_heatmap.png` | Pairwise correlation heatmap |
| `var_comparison.png` | Return distribution with VaR & CVaR overlaid |
| `drawdown.png` | Underwater equity curve with max drawdown annotation |
| `efficient_frontier.png` | Frontier scatter coloured by Sharpe ratio |

## Portfolio

European-tilted multi-asset portfolio (configurable in `src/config.py`):

| Ticker | Name |
|---|---|
| AAPL | Apple |
| MSFT | Microsoft |
| MC.PA | LVMH |
| SAN.PA | Sanofi |
| SIE.DE | Siemens |
| BNP.PA | BNP Paribas |
| TTE.PA | TotalEnergies |
| ASML | ASML |

## Quick Start

```bash
git clone https://github.com/davidepellegrino-stack/portfolio-risk-analyzer.git
cd portfolio-risk-analyzer
pip install -r requirements.txt

# Fetch data and run full analysis
python src/main.py
```

To change the portfolio, edit `TICKERS` in `src/config.py`.

## Project Structure

```
portfolio-risk-analyzer/
├── src/
│   ├── config.py           # All parameters and paths
│   ├── fetch_data.py       # Download prices from Yahoo Finance
│   ├── risk_metrics.py     # VaR, CVaR, drawdown, volatility
│   ├── portfolio.py        # Mean-variance optimization
│   ├── visualizations.py   # All charts
│   └── main.py             # End-to-end pipeline
├── data/                   # Downloaded price/return CSVs
├── outputs/
│   ├── figures/            # PNG charts
│   └── tables/             # CSV result tables
├── requirements.txt
└── README.md
```

## Methodology Notes

**VaR methods comparison:** Historical simulation makes no distributional assumption but is sensitive to the sample period. The parametric approach is computationally efficient but underestimates tail risk in heavy-tailed return distributions (excess kurtosis is typically positive for equities). Monte Carlo simulation allows for non-normal distributions and multi-day horizons through Cholesky-decomposed correlated draws.

**Optimization:** The efficient frontier is traced via random Dirichlet-weighted portfolios. The Max Sharpe and Min Variance portfolios are found via SLSQP constrained optimization with long-only constraints (weights ∈ [0,1], Σw = 1).

## Requirements

- Python 3.9+
- yfinance, pandas, numpy, scipy, matplotlib, seaborn
