"""
All charts for the portfolio risk analyzer.
  1. Cumulative returns
  2. Rolling volatility
  3. Correlation heatmap
  4. VaR comparison (3 methods)
  5. Drawdown chart
  6. Efficient frontier with optimised portfolios
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_FIGS, CONFIDENCE_LEVEL, ROLLING_WINDOW, TICKERS

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = ["#2c3e50", "#e74c3c", "#3498db", "#27ae60",
          "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]


def plot_cumulative_returns(returns: pd.DataFrame):
    """Cumulative log return for each asset."""
    cumulative = (1 + returns).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, col in enumerate(cumulative.columns):
        label = TICKERS.get(col, col)
        ax.plot(cumulative.index, cumulative[col],
                color=COLORS[i % len(COLORS)], label=label, linewidth=1.5)

    ax.set_title("Cumulative Returns", fontweight="bold", fontsize=13)
    ax.set_ylabel("Growth of €1")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "cumulative_returns.png"), bbox_inches="tight")
    plt.close()
    print("Saved: cumulative_returns.png")


def plot_rolling_volatility(returns: pd.DataFrame, window: int = ROLLING_WINDOW):
    """Annualised rolling volatility for each asset."""
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, col in enumerate(rolling_vol.columns):
        label = TICKERS.get(col, col)
        ax.plot(rolling_vol.index, rolling_vol[col],
                color=COLORS[i % len(COLORS)], label=label,
                linewidth=1.2, alpha=0.85)

    ax.set_title(f"Rolling {window}-Day Annualised Volatility (%)",
                 fontweight="bold", fontsize=13)
    ax.set_ylabel("Volatility (%)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "rolling_volatility.png"), bbox_inches="tight")
    plt.close()
    print("Saved: rolling_volatility.png")


def plot_correlation_heatmap(returns: pd.DataFrame):
    """Correlation matrix heatmap with asset names."""
    corr = returns.corr()
    labels = [TICKERS.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=-1, vmax=1, center=0,
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 9}
    )
    ax.set_title("Pairwise Correlation Matrix", fontweight="bold", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("Saved: correlation_heatmap.png")


def plot_var_comparison(portfolio_returns: pd.Series,
                        var_hist: float, var_param: float, var_mc: float,
                        cvar_hist: float):
    """Return distribution with VaR and CVaR overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(portfolio_returns * 100, bins=80, color="#3498db",
            alpha=0.6, edgecolor="white", linewidth=0.3, label="Daily returns")

    ax.axvline(-var_hist * 100,  color="#e74c3c",  lw=2,
               linestyle="--", label=f"VaR Historical ({CONFIDENCE_LEVEL:.0%}): {-var_hist*100:.2f}%")
    ax.axvline(-var_param * 100, color="#f39c12",  lw=2,
               linestyle=":",  label=f"VaR Parametric ({CONFIDENCE_LEVEL:.0%}): {-var_param*100:.2f}%")
    ax.axvline(-var_mc * 100,    color="#9b59b6",  lw=2,
               linestyle="-.", label=f"VaR Monte Carlo (10-day): {-var_mc*100:.2f}%")
    ax.axvline(-cvar_hist * 100, color="#2c3e50",  lw=2,
               linestyle="-",  label=f"CVaR (ES) Historical: {-cvar_hist*100:.2f}%")

    ax.set_title("Portfolio Return Distribution with VaR & CVaR",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "var_comparison.png"), bbox_inches="tight")
    plt.close()
    print("Saved: var_comparison.png")


def plot_drawdown(returns: pd.Series, label: str = "Equal-Weight Portfolio"):
    """Underwater equity curve showing drawdown over time."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(cumulative.index, cumulative, color="#2c3e50", linewidth=1.5)
    ax1.set_title(f"{label} — Cumulative Performance & Drawdown",
                  fontweight="bold", fontsize=13)
    ax1.set_ylabel("Portfolio Value (€)")

    ax2.fill_between(drawdown.index, drawdown, 0, color="#e74c3c", alpha=0.6)
    ax2.plot(drawdown.index, drawdown, color="#e74c3c", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Annotate max drawdown
    min_dd = drawdown.min()
    min_date = drawdown.idxmin()
    ax2.annotate(f"Max DD: {min_dd:.1f}%",
                 xy=(min_date, min_dd),
                 xytext=(min_date, min_dd - 3),
                 fontsize=9, color="#c0392b", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "drawdown.png"), bbox_inches="tight")
    plt.close()
    print("Saved: drawdown.png")


def plot_efficient_frontier(simulated: pd.DataFrame,
                            max_sharpe: dict, min_var: dict, eq_weight: dict):
    """Efficient frontier scatter with optimised portfolios highlighted."""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Random portfolios coloured by Sharpe ratio
    sc = ax.scatter(
        simulated["volatility"] * 100,
        simulated["return"] * 100,
        c=simulated["sharpe"], cmap="viridis",
        s=4, alpha=0.5, linewidths=0
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # Optimised portfolios
    for p, color, marker, size in [
        (max_sharpe,  "#e74c3c", "*", 350),
        (min_var,     "#f39c12", "D", 150),
        (eq_weight,   "#2c3e50", "o", 150),
    ]:
        ax.scatter(
            p["volatility"] * 100, p["return"] * 100,
            color=color, marker=marker, s=size,
            zorder=5, edgecolors="white", linewidths=0.8,
            label=f"{p['label']} (Sharpe: {p['sharpe']:.2f})"
        )

    ax.set_title("Efficient Frontier — Mean-Variance Optimisation",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("Annualised Return (%)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "efficient_frontier.png"), bbox_inches="tight")
    plt.close()
    print("Saved: efficient_frontier.png")
