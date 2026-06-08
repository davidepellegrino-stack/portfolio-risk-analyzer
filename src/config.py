"""
Configuration for the portfolio risk analyzer.
"""
import os

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
OUTPUT_TABLES = os.path.join(PROJECT_ROOT, "outputs", "tables")
OUTPUT_FIGS   = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGS, exist_ok=True)

# ── PORTFOLIO (European-tilted multi-asset) ──────────────────────────────
TICKERS = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "MC.PA": "LVMH",
    "SAN.PA":"Sanofi",
    "SIE.DE":"Siemens",
    "BNP.PA":"BNP Paribas",
    "TTE.PA":"TotalEnergies",
    "ASML":  "ASML",
}

EQUAL_WEIGHTS = {t: 1.0 / len(TICKERS) for t in TICKERS}

# ── PARAMETERS ───────────────────────────────────────────────────────────
START_DATE       = "2018-01-01"
END_DATE         = "2026-06-01"
CONFIDENCE_LEVEL = 0.95
MC_SIMULATIONS   = 10_000
MC_HORIZON       = 10            # Trading days
ROLLING_WINDOW   = 60            # Days
RISK_FREE_RATE   = 0.03          # Annualised
TRADING_DAYS     = 252
