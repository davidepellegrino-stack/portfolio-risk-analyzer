"""Figures for the validation report."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import exceedances, rolling_exceedances

INK, GRID, RED, BLUE = "#1f2933", "#d9dee3", "#c0392b", "#1f5a94"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})


def plot_var_backtests(returns: pd.Series, forecasts: dict[str, pd.DataFrame], path) -> None:
    fig, axes = plt.subplots(len(forecasts), 1, figsize=(10, 2.3 * len(forecasts)), sharex=True)
    for ax, (name, f) in zip(axes, forecasts.items()):
        r = returns.loc[f.index]
        hits = exceedances(r, f["var_99"])
        ax.bar(r.index, 100 * r, width=1.0, color="#9aa5b1", linewidth=0)
        ax.plot(f.index, -100 * f["var_99"], color=BLUE, linewidth=0.9)
        ax.scatter(r.index[hits], 100 * r[hits], color=RED, s=12, zorder=3)
        ax.set_title(f"{name}: {int(hits.sum())} exceedances in {len(r)} days "
                     f"(expected {0.01 * len(r):.0f})", loc="left", fontsize=9, color=INK)
        ax.set_ylabel("daily return, %")
    fig.suptitle("Daily P&L (grey), 99% VaR forecast (blue), exceedances (red)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_traffic_light(returns: pd.Series, forecasts: dict[str, pd.DataFrame], path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhspan(0, 4.5, color="#2e8b57", alpha=0.10, linewidth=0)
    ax.axhspan(4.5, 9.5, color="#e0a100", alpha=0.13, linewidth=0)
    ax.axhspan(9.5, 30, color=RED, alpha=0.10, linewidth=0)
    top = 10
    for name, f in forecasts.items():
        roll = rolling_exceedances(exceedances(returns.loc[f.index], f["var_99"]))
        ax.plot(roll.index, roll, linewidth=1.2, label=name)
        top = max(top, roll.max())
    ax.set_ylim(0, top + 1.5)
    ax.set_ylabel("exceedances in trailing 250 days")
    ax.set_title("Basel traffic light: green 0-4, yellow 5-9, red 10+", loc="left", color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_contributions(comp_var: pd.DataFrame, comp_es: pd.DataFrame, path) -> None:
    df = pd.DataFrame({"Weight": comp_var["weight_%"],
                       "Share of 99% VaR (normal)": comp_var["share_of_var_%"],
                       "Share of 97.5% ES (historical)": comp_es["share_of_es_%"]})
    df = df.sort_values("Share of 97.5% ES (historical)")
    ax = df.plot.barh(figsize=(8, 4.5), color=["#9aa5b1", BLUE, RED], width=0.8)
    ax.set_xlabel("% of portfolio")
    ax.set_title("Equal weights, unequal risk", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower right")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)
