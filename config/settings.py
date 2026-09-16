"""
config/settings.py
Central configuration for GeoSignal pipeline.
All API keys are loaded from environment variables (never hardcode secrets).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW      = BASE_DIR / "data" / "raw"
DATA_PROC     = BASE_DIR / "data" / "processed"
RESULTS_DIR   = BASE_DIR / "results"
CONFIG_DIR    = BASE_DIR / "config"

# Ensure directories exist
for _dir in [DATA_RAW, DATA_PROC, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
FRED_API_KEY        = os.getenv("FRED_API_KEY", "")
GOOGLE_PROJECT_ID   = os.getenv("GOOGLE_CLOUD_PROJECT", "")  # for BigQuery/GDELT 2.0

# ── GDELT Settings ────────────────────────────────────────────────────────────
GDELT_BASE_URL     = "http://data.gdeltproject.org/events/"
GDELT_GKG_URL      = "http://data.gdeltproject.org/gkg/"

# ── UCDP Settings ─────────────────────────────────────────────────────────────
UCDP_API_BASE      = "https://ucdpapi.pcr.uu.se/api"

# ── Temporal Windows ──────────────────────────────────────────────────────────
IMPACT_WINDOWS     = [1, 3, 7, 14, 30]     # days post-event (configurable)
ESTIMATION_WINDOW  = 120                  # trading days before event for market model
EVENT_WINDOW       = (-5, 30)             # (pre, post) trading days around event

# ── BM25 / TF-IDF ─────────────────────────────────────────────────────────────
BM25_TOP_N         = 10                   # top news articles per event
MIN_COSINE_SIM     = 0.05                 # threshold for article-event alignment (calibrated for GKG metadata proxy)
NEWS_TEMPORAL_BAND = 3                    # days around event for news retrieval

# ── Goldstein Scale ───────────────────────────────────────────────────────────
# Maps CAMEO root codes → Goldstein conflict intensity scores
GOLDSTEIN_MAP_PATH = CONFIG_DIR / "cameo_goldstein_map.csv"

# ── Sector Taxonomy ───────────────────────────────────────────────────────────
GICS_MAP_PATH      = CONFIG_DIR / "gics_taxonomy.csv"
NIC_MAP_PATH       = CONFIG_DIR / "nic2008_taxonomy.csv"
ENTITY_RULES_PATH  = CONFIG_DIR / "entity_rules.jsonl"

# ── NSE Sectoral Tickers (yfinance) ───────────────────────────────────────────
NSE_SECTOR_TICKERS = {
    "Auto"          : "^CNXAUTO",
    "Bank"          : "^NSEBANK",
    "FMCG"          : "^CNXFMCG",
    "IT"            : "^CNXIT",
    "Metal"         : "^CNXMETAL",
    "Energy"        : "^CNXENERGY",
    "Pharma"        : "^CNXPHARMA",
    "Realty"        : "^CNXREALTY",
    "Infra"         : "^CNXINFRA",
    "Media"         : "^CNXMEDIA",
}
NIFTY50_TICKER     = "^NSEI"             # Broad market index

# ── Granger Causality ─────────────────────────────────────────────────────────
GRANGER_MAX_LAGS   = 12
ADF_SIGNIFICANCE   = 0.05
