import os
import csv
import datetime as dt
import pandas as pd
import numpy as np

# Silence HF tokenizer fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.universe.load_universe import load_universe
from src.universe.aliases import primary_name
from src.data.fetch_prices import get_prices
from src.news.fetch_news import get_company_news, get_local_news
from src.nlp.sentiment import score_texts
from src.features.aggregate_sentiment import aggregate_daily_sentiment as agg_news
from src.features.aggregate_reddit import aggregate_daily_reddit as agg_reddit
from src.portfolio.positions import load_positions, save_positions, open_position, close_position
from src.data.short_interest import get_short_interest_pct
from src.ingest.store import get_db, query_reddit

# ================= Settings =================
UNIVERSE_CSV         = "src/universe/tickers.csv"
LOOKBACK_DAYS        = 180
MIN_PRICE            = 5.0
MIN_AVG_VOL_20D      = 2_000_000

TOP_K_FOR_NEWS       = 15            # only fetch news for top momentum names
TOP_N_BUYS           = 10
TOP_N_SHORTS         = 5             # bottom-K for shorts

NEWS_LOOKBACK_HRS    = 96
REDDIT_LOOKBACK_HRS  = 72

# Exits (for open LONG positions)
HOLD_DAYS_MAX        = 3             # days
STOP_LOSS_LONG       = 0.08          # -8%
TAKE_PROFIT_LONG     = 0.05          # +5%
SENT_EXIT_MIN        = 0.20          # blended sentiment below this → SELL

# Weights for final ranking (higher → more bullish)
W_MOM                = 0.30          # momentum (z)
W_RS                 = 0.30          # relative strength vs sector (z)
W_NEWS               = 0.25          # news sentiment (0..1)
W_REDDIT             = 0.15          # reddit sentiment (0..1)
W_SQUEEZE            = 0.10          # short-interest × momentum boost

# Portfolio behavior
ALLOW_SHORT_TRADES   = False         # if True, open short positions (qty=-1) when action="SHORT"
# ===========================================

# Minimal sector map; extend as needed (fallback to SPY)
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "AVGO": "XLK",
    # Add more tickers as needed
}

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - np.nanmean(s)) / np.nanstd(s)

# ...rest of your pipeline code...
