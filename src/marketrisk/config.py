"""All choices that affect a number in the results live here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

# Portfolio: eight large caps, equal weight, reported in EUR.
# ASML here is the Nasdaq listing, so it is quoted in USD.
TICKERS = {
    "AAPL": "USD", "MSFT": "USD", "ASML": "USD",
    "MC.PA": "EUR", "SAN.PA": "EUR", "SIE.DE": "EUR",
    "BNP.PA": "EUR", "TTE.PA": "EUR",
}
BASE_CCY = "EUR"
FX_TICKER = "EURUSD=X"          # USD per 1 EUR
CALENDAR_TICKER = "MC.PA"       # Euronext Paris trading days define the calendar

START_DATE = "2015-01-01"
END_DATE = "2026-10-31"         # fixed so the published results can be reproduced

# Confidence levels, written as tail probabilities.
VAR_ALPHA = 0.01                # 99% VaR: the level of the Basel backtest
ES_ALPHA = 0.025                # 97.5% ES: the FRTB internal-models measure

# Estimation
BURN_IN = 1000                  # observations before the first forecast (all models)
HS_WINDOW = 500                 # historical simulation look-back
NORMAL_WINDOW = 250             # rolling-sigma look-back
EWMA_LAMBDA = 0.94              # RiskMetrics daily decay
GARCH_WINDOW = 1000             # rolling window for GARCH estimation
GARCH_REFIT = 20                # re-estimate GARCH parameters every 20 trading days

# Historical stress windows (inclusive dates).
STRESS_SCENARIOS = {
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-09-30"),
    "April 2025 tariff shock": ("2025-04-02", "2025-04-08"),
}
