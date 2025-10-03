# src/pipeline.py - REVISED WITH FIXES AND LOGGING
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
from src.reporting.log_trades import log_closed_trade # <<< NEW IMPORT

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
SENT_EXIT_MIN   = 0.20            # blended sentiment below this → SELL (using [0, 1] normalized scale)

# Weights for final ranking (Normalized to sum to 1.0)
W_MOM     = 0.30
W_RS      = 0.30
W_NEWS    = 0.20    # Adjusted from 0.25
W_REDDIT  = 0.10    # Adjusted from 0.15
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

    ret5 = close.pct_change(5)
    avg_vol20 = volume.rolling(20).mean()

    last_close = close.loc[today].dropna()
    last_avg_vol20 = avg_vol20.loc[today].reindex(last_close.index)

    # Screen tickers
    screened = last_close[(last_close > MIN_PRICE) & (last_avg_vol20 > MIN_AVG_VOL_20D)].index.tolist()
    if len(screened) < 10:
        screened = last_close.index.tolist()

    # Momentum
    r5_today = ret5.loc[today].reindex(screened).dropna()
    ranked_by_r5 = r5_today.sort_values(ascending=False)
    r5_z_all = zscore(ranked_by_r5)  # keep for squeeze

    # 3) News sentiment (48h window for top-K momentum names)
    candidates = r5_today.sort_values(ascending=False).head(TOP_K_FOR_NEWS).index.tolist()
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=48)

    news_rows = []
    for t in candidates:
        try:
            arts = get_company_news(t, start, end, page_size=20)
        except Exception:
            arts = []
        titles = [a.get("title") or "" for a in arts]
        # score_texts now returns a list of continuous scores [-1, 1]
        scored = score_texts(titles, max_len=128, batch_size=16) if titles else []
        for a, sc in zip(arts, scored):
            news_rows.append({
                "ticker": t,
                "publishedAt": a.get("publishedAt"),
                "title": a.get("title"),
                "sent_score": float(sc),
            })

    news_df = pd.DataFrame(news_rows)
    if not news_df.empty:
        news_daily = agg_news(news_df)
        # Fix: use the correct column name 'sent_norm'
        news_today = news_daily[news_daily["date"] == dt.datetime.utcnow().date()].set_index("ticker")
    else:
        news_today = pd.DataFrame(columns=["sent_norm","sent_mean","n_news"]).set_index(pd.Index([]))

    # 4) Reddit sentiment
    term_map = {t: [t, f"{t} stock"] for t in screened}
    reddit_posts = fetch_reddit_posts(SUBREDDITS, lookback_hours=REDDIT_LOOKBACK_HRS, limit_per_sub=REDDIT_LIMIT_PERSUB)
    reddit_df = pd.DataFrame(reddit_posts)

    def _clip(s, n): return (s or "")[:n]

    if not reddit_df.empty:
        tagged = tag_tickers(reddit_df, term_map)
        texts = [f"{_clip(t,160)}. {_clip(x,540)}"
                 for t, x in zip(tagged["title"].fillna(""), tagged["selftext"].fillna(""))]
        # score_texts now returns a list of continuous scores [-1, 1]
        scored = score_texts(texts, max_len=128, batch_size=16) if texts else []
        if scored:
            tagged = tagged.assign(
                sent_score=[float(s) for s in scored]
            )
        reddit_daily = agg_reddit(tagged)
        # Fix: use the correct column name 'sent_reddit_norm'
        reddit_today = reddit_daily[reddit_daily["date"] == dt.datetime.utcnow().date()].set_index("ticker")
    else:
        reddit_today = pd.DataFrame(columns=["sent_reddit_norm","sent_reddit_mean","n_reddit"]).set_index(pd.Index([]))

    # 5) Relative Strength vs sector ETF (5d)
    sector_etfs = sorted(set(SECTOR_MAP.get(t, "SPY") for t in screened) | {"SPY"})
    sec_close, _ = get_prices(sector_etfs, lookback_days=LOOKBACK_DAYS)
    ret5_sec = sec_close.pct_change(5)

    rs5 = {}
    for t in screened:
        stock_r5 = float(r5_today.get(t, np.nan)) if t in r5_today.index else np.nan
        etf = SECTOR_MAP.get(t, "SPY")
        if etf in ret5_sec.columns and today in ret5_sec.index:
            sec_r5 = float(ret5_sec.loc[today, etf])
        else:
            sec_r5 = float(ret5_sec.loc[today, "SPY"]) if "SPY" in ret5_sec.columns else 0.0
        rs5[t] = stock_r5 - sec_r5
    rs5_series = pd.Series(rs5).dropna()
    rs5_z = zscore(rs5_series).reindex(r5_today.index).fillna(0.0)

    # 6) Short interest & Squeeze score
    si_pct = get_short_interest_pct(screened)                             # 0..1
    si_z = zscore(si_pct).reindex(r5_today.index).fillna(0.0)
    squeeze_score = si_z.clip(lower=0) * r5_z_all.reindex(r5_today.index).fillna(0.0).clip(lower=0)

    # 7) Combine scores (using W_MOM to W_SQUEEZE summing to 1.0)
    # s_news and s_redd now contain the [0, 1] normalized score (0.5=neutral)
    s_news = pd.Series(0.0, index=r5_today.index)
    n_news = pd.Series(0, index=r5_today.index, dtype=int)
    if not news_today.empty:
        # FIX: Use 'sent_norm'
        s_news.update(news_today.get("sent_norm", pd.Series(dtype=float)))
        n_news.update(news_today.get("n_news", pd.Series(dtype=int)))

    s_redd = pd.Series(0.0, index=r5_today.index)
    n_redd = pd.Series(0, index=r5_today.index, dtype=int)
    if not reddit_today.empty:
        # FIX: Use 'sent_reddit_norm'
        s_redd.update(reddit_today.get("sent_reddit_norm", pd.Series(dtype=float)))
        n_redd.update(reddit_today.get("n_reddit", pd.Series(dtype=int)))

    combined = (
        W_MOM     * r5_z_all.reindex(r5_today.index).fillna(0.0) +
        W_RS      * rs5_z.reindex(r5_today.index).fillna(0.0) +
        W_NEWS    * s_news.reindex(r5_today.index).fillna(0.0) +
        W_REDDIT  * s_redd.reindex(r5_today.index).fillna(0.0) +
        W_SQUEEZE * squeeze_score.fillna(0.0)
    )

    buy_set = set(combined.sort_values(ascending=False).head(TOP_N_BUYS).index.tolist())

    # 8) BUY/HOLD rows
    rows_base = []
    for t in screened:
        lc = float(last_close.get(t, 0.0))
        r5 = float(r5_today.get(t, 0.0)) if t in r5_today.index else 0.0
        sn = float(s_news.get(t, 0.0))
        sr = float(s_redd.get(t, 0.0))
        nn = int(n_news.get(t, 0))
        nr = int(n_redd.get(t, 0))
        rs = float(rs5.get(t, 0.0))
        si = float(si_pct.get(t, 0.0))
        sq = float(squeeze_score.get(t, 0.0))

        action = "BUY" if t in buy_set else "HOLD"
        # Simplified confidence based on combined score ranking
        conf = max(0.0, min(0.99, (float(combined.get(t, 0)) / combined.max()) * 0.9))

        reasons = (
            f"r5={r5:.3f}, RS5={rs:.3f}, news_norm={sn:.2f} (n={nn}), "
            f"reddit_norm={sr:.2f} (n={nr}), short%={si:.3f}, sq={sq:.3f}, "
            f"score={float(combined.get(t,0)):.3f}"
        )

        feats = {
            "ret_5d": round(r5, 4),
            "rel_strength_5d": round(rs, 4),
            "short_pct_float": round(si, 3),
            "squeeze_score": round(sq, 3),
            "sent_news_norm": round(sn, 2), "n_news": nn,
            "sent_reddit_norm": round(sr, 2), "n_reddit": nr,
            "score": round(float(combined.get(t, 0)), 3)
        }

        rows_base.append([
            dt.date.today().isoformat(), t, action, 0, lc, 0, 0,
            round(conf, 2), reasons, str(feats)
        ])

    # 9) SELL rules
    positions = load_positions()
    sell_rows = []
    if not positions.empty:
        for _, pos in positions.iterrows():
            t = str(pos["ticker"])
            if t not in last_close.index:
                continue
            
            # Ensure entry_date is a datetime object for calculation and logging
            entry_dt = dt.date.fromisoformat(str(pos["entry_date"]))
            
            entry = float(pos["entry_price"])
            price = float(last_close[t])
            pnl = (price - entry) / entry
            days_held = (dt.date.today() - entry_dt).days
            r5 = float(r5_today.get(t, 0.0)) if t in r5_today.index else 0.0
            sn = float(s_news.get(t, 0.0))
            sr = float(s_redd.get(t, 0.0))
            si = float(si_pct.get(t, 0.0)) if t in si_pct.index else 0.0
            sq = float(squeeze_score.get(t, 0.0)) if t in squeeze_score.index else 0.0

            blended_sent = (0.6 * sn + 0.4 * sr)

            triggers, do_sell = [], False
            if pnl <= -STOP_LOSS:             do_sell=True; triggers.append(f"stop {-STOP_LOSS*100:.0f}%")
            if pnl >=  TAKE_PROFIT:           do_sell=True; triggers.append(f"take +{TAKE_PROFIT*100:.0f}%")
            if r5 <= 0:                       do_sell=True; triggers.append("momentum<=0")
            if blended_sent < SENT_EXIT_MIN:
                                             do_sell=True; triggers.append(f"sent_norm<{SENT_EXIT_MIN:.2f}")
            if days_held >= HOLD_DAYS_MAX:    do_sell=True; triggers.append(f"time>{HOLD_DAYS_MAX}d")

            if do_sell:
                exit_features = str({
                    "pnl": round(float(pnl), 4),
                    "r5": round(r5, 4),
                    "news_norm": round(sn, 2),
                    "reddit_norm": round(sr, 2),
                    "short_pct_float": round(si, 3),
                    "squeeze_score": round(sq, 3)
                })
                
                # NEW: Log the closed trade before removing it from positions
                log_closed_trade(
                    t,
                    entry_dt,
                    entry,
                    dt.date.today(),
                    price,
                    int(pos.get("qty", 1)),
                    pnl,
                    "; ".join(triggers),
                    exit_features
                )

                sell_rows.append([
                    dt.date.today().isoformat(), t, "SELL", int(pos.get("qty", 1)), price, 0, 0, 0.80,
                    "; ".join(triggers),
                    exit_features
                ])
                positions = close_position(positions, t)

    # 10) Prevent duplicate buys
    open_set = set(positions["ticker"].tolist()) if not positions.empty else set()
    for row in rows_base:
        if row[1] in open_set and row[2] == "BUY":
            row[2] = "HOLD"

    # 11) Write signals
    header = ["date","ticker","action","qty","entry_price","stop","take_profit","confidence","reasons","features_json"]
    with open("data/signals_latest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for sr in sell_rows: w.writerow(sr)
        for br in rows_base: w.writerow(br)

    # 12) Update positions
    updated = positions.copy()
    for row in rows_base:
        d, t, action, _, entry_price, *_ = row
        if action == "BUY":
            updated = open_position(updated, t, 1, float(entry_price), d)
    save_positions(updated)

    logging.info("--- Pipeline Finished ---")
    print("✅ signals_updated.",
          f"Sells: {', '.join([r[1] for r in sell_rows]) or 'None'} |",
          f"Buys: {', '.join(sorted(list(buy_set))) or 'None'}")

if __name__ == "__main__":
    main()
