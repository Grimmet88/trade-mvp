# src/pipeline.py - REVISED
import os, csv, datetime as dt
import pandas as pd
import numpy as np
import logging

from src.universe.load_universe import load_universe
from src.data.fetch_prices import get_prices
from src.news.fetch_news import get_company_news
from src.nlp.sentiment import score_texts
from src.features.aggregate_sentiment import aggregate_daily_sentiment as agg_news
from src.scrapers.fetch_reddit import fetch_reddit_posts
from src.features.aggregate_reddit import tag_tickers, aggregate_daily_reddit as agg_reddit
from src.portfolio.positions import load_positions, save_positions, open_position, close_position
from src.data.short_interest import get_short_interest_pct

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- Settings ----------------
UNIVERSE_CSV       = "src/universe/tickers.csv"
LOOKBACK_DAYS      = 180
MIN_PRICE          = 5.0
MIN_AVG_VOL_20D    = 2_000_000

TOP_K_FOR_NEWS     = 15           # fetch news only for top momentum names
TOP_N_BUYS         = 10

# Reddit collection
SUBREDDITS          = ["stocks", "investing", "wallstreetbets"]
REDDIT_LOOKBACK_HRS = 72
REDDIT_LIMIT_PERSUB = 200

# Exit rules
HOLD_DAYS_MAX   = 3               # days
STOP_LOSS       = 0.08            # -8%
TAKE_PROFIT     = 0.05            # +5%
SENT_EXIT_MIN   = 0.20            # blended sentiment below this → SELL (using [0, 1] scale, 0.5 is neutral)

# Weights for final ranking (MUST sum to 1.0)
W_MOM     = 0.30
W_RS      = 0.30
W_NEWS    = 0.20    # Adjusted from 0.25 to accommodate W_SQUEEZE
W_REDDIT  = 0.10    # Adjusted from 0.15 to accommodate W_SQUEEZE
W_SQUEEZE = 0.10    # Squeeze factor weight
# TOTAL SUM = 1.00
# ------------------------------------------

# Minimal sector map; extend as needed (fallback SPY)
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "AVGO": "XLK",
    "META": "XLC", "GOOGL": "XLC", "GOOG": "XLC", "NFLX": "XLC",
    "ADBE": "XLK", "CRM": "XLK", "CSCO": "XLK", "ORCL": "XLK", "TXN": "XLK", "QCOM": "XLK",
    "JPM": "XLF", "V": "XLF", "MA": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF", "PYPL": "XLF",
    "LLY": "XLV", "MRK": "XLV", "TMO": "XLV", "UNH": "XLV", "PFE": "XLV",
    "AMZN": "XLY", "TSLA": "XLY", "COST": "XLY", "MCD": "XLY", "NKE": "XLY", "WMT": "XLY", "HD": "XLY",
    "PEP": "XLP", "KO": "XLP",
    "CAT": "XLI", "GE": "XLI",
    "XOM": "XLE", "CVX": "XLE",
}

os.makedirs("data", exist_ok=True)

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / sd if (sd and sd > 0) else s * 0

def main():
    logging.info("--- Starting Prediction Pipeline ---")
    
    # 1) Universe
    tickers = load_universe(UNIVERSE_CSV)
    if not tickers:
        logging.error("Universe is empty. Exiting.")
        raise SystemExit("Universe is empty. Add tickers to src/universe/tickers.csv")

    # 2) Price data
    close, volume = get_prices(tickers, lookback_days=LOOKBACK_DAYS)
    today = close.index.max()
    if pd.isna(today):
        logging.error("No price data returned. Exiting.")
        raise SystemExit("No price data returned.")

    logging.info(f"Processing date: {today.date()}. {len(close.columns)} tickers loaded.")

    ret5 = close.pct_change(5)
    avg_vol20 = volume.rolling(20).mean()

    last_close = close.loc[today].dropna()
    last_avg_vol20 = avg_vol20.loc[today].reindex(last_close.index)

    # Screen tickers
    screened = last_close[(last_close > MIN_PRICE) & (last_avg_vol20 > MIN_AVG_VOL_20D)].index.tolist()
    if len(screened) < 10:
        logging.warning("Low number of screened tickers. Using all available tickers.")
        screened = last_close.index.tolist()

    # Momentum
    r5_today = ret5.loc[today].reindex(screened).dropna()
    ranked_by_r5 = r5_today.sort_values(ascending=False)
    r5_z_all = zscore(ranked_by_r5)  # keep for squeeze

    # 3) News sentiment (48h window for top-K momentum names)
    candidates = r5_today.sort_values(ascending=False).head(TOP_K_FOR_NEWS).index.tolist()
    logging.info(f"Fetching news for top {len(candidates)} momentum stocks.")
    
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=48)

    news_rows = []
    for t in candidates:
        try:
            arts = get_company_news(t, start, end, page_size=20)
        except Exception as e:
            logging.warning(f"News fetch failed for {t}: {e}")
            arts = []
        titles = [a.get("title") or "" for a in arts]
        # score_texts now returns a list of continuous scores [-1, 1]
        scored = score_texts(titles, max_len=128, batch_size=16) if titles else []
        for a, sc in zip(arts, scored):
            news_rows.append({
                "ticker": t,
                "publishedAt": a.get("publishedAt"),
                "title": a.get("title"),
                "sent_score": float(sc), # Note: `sc` is the float score, not a tuple
            })

    news_df = pd.DataFrame(news_rows)
    if not news_df.empty:
        news_daily = agg_news(news_df)
        # FIX: sentiment aggregation is now in the 'sent_norm' column
        news_today = news_daily[news_daily["date"] == dt.datetime.utcnow().date()].set_index("ticker")
        logging.info(f"News sentiment aggregated for {len(news_today)} tickers.")
    else:
        news_today = pd.DataFrame(columns=["sent_norm","sent_mean","n_news"]).set_index(pd.Index([]))

    # 4) Reddit sentiment
    term_map = {t: [t, f"{t} stock"] for t in screened}
    reddit_posts = fetch_reddit_posts(SUBREDDITS, lookback_hours=REDDIT_LOOKBACK_HRS, limit_per_sub=REDDIT_LIMIT_PERSUB)
    reddit_df = pd.DataFrame(reddit_posts)

    def _clip(s, n): return (s or "")[:n]

    if not reddit_df.empty:
        tagged
