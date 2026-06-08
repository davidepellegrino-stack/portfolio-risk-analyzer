"""
Download adjusted close prices from Yahoo Finance.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import TICKERS, START_DATE, END_DATE, DATA_DIR

def fetch():
    tickers = list(TICKERS.keys())
    print(f"Downloading {len(tickers)} assets: {', '.join(tickers)}")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=True)

    prices = raw["Close"] if len(tickers) > 1 else raw["Close"].to_frame(tickers[0])
    prices = prices.dropna(how="all")

    # Log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    prices.to_csv(os.path.join(DATA_DIR, "prices.csv"))
    log_returns.to_csv(os.path.join(DATA_DIR, "returns.csv"))

    print(f"\nSaved {len(prices)} trading days ({prices.index[0].date()} to {prices.index[-1].date()})")
    print(f"Assets with complete data: {list(prices.columns)}")
    return prices, log_returns

if __name__ == "__main__":
    fetch()
