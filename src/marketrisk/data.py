"""Prices -> EUR prices on one calendar -> simple returns -> portfolio P&L."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def download_prices(start: str = cfg.START_DATE, end: str = cfg.END_DATE) -> pd.DataFrame:
    """Adjusted closes for the portfolio plus EURUSD, straight from Yahoo Finance."""
    import yfinance as yf

    symbols = list(cfg.TICKERS) + [cfg.FX_TICKER]
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    return raw["Close"][symbols]


def to_base_currency(prices: pd.DataFrame, currencies: dict[str, str],
                     eurusd: pd.Series) -> pd.DataFrame:
    """Convert USD prices to EUR. EURUSD is USD per EUR, so EUR price = USD price / EURUSD."""
    out = prices.copy()
    for ticker, ccy in currencies.items():
        if ccy == "USD":
            out[ticker] = prices[ticker] / eurusd
        elif ccy != "EUR":
            raise ValueError(f"unsupported currency {ccy} for {ticker}")
    return out


def align_calendar(prices: pd.DataFrame, calendar: pd.Series, max_fill: int = 3) -> pd.DataFrame:
    """Keep the days on which the calendar market traded.

    A market closed on such a day keeps its last price (zero return), which is how
    a risk system marks a position when no new price exists. Longer gaps are left
    as NaN so that a data problem is not silently filled.
    """
    days = calendar.dropna().index
    return prices.ffill(limit=max_fill).reindex(days)


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().iloc[1:]


def portfolio_returns(asset_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Daily return of a portfolio held at fixed weights.

    The portfolio return is the weighted sum of SIMPLE asset returns. Log returns
    add up over time but not across assets, so they must not be used here.
    Applying today's weights to the whole history gives the P&L today's portfolio
    would have had, which is the series a risk model for today's portfolio needs.
    """
    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    return (asset_returns * weights).sum(axis=1).rename("portfolio")


def load_market_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EUR prices and simple returns, cached in data/ after the first download."""
    cache = cfg.DATA_DIR / "prices_raw.csv"
    if cache.exists() and not refresh:
        raw = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        raw = download_prices()
        cfg.DATA_DIR.mkdir(exist_ok=True)
        raw.to_csv(cache)

    tickers = list(cfg.TICKERS)
    filled = raw.ffill(limit=3)
    eur = to_base_currency(filled[tickers], cfg.TICKERS, filled[cfg.FX_TICKER])
    eur = align_calendar(eur, raw[cfg.CALENDAR_TICKER]).dropna()
    return eur, simple_returns(eur)
