import os, csv, datetime as dt
import pandas as pd
import numpy as np

from src.universe.load_universe import load_universe
from src.data.fetch_prices import get_prices
from src.news.fetch_news import get_company_news
from src.nlp.sentiment import score_texts
from src.features.aggregate_sentiment import aggregate_daily_sentiment

os.makedirs("data", exist_ok=True)

# -------- Settings you can tweak --------
UNIVERSE_CSV = "src/universe/tickers.csv"
LOOKBACK_DAYS = 180
MIN_PRICE = 5.0
MIN_AVG_VOL_20D = 2_000_000
TOP_K_FOR_NEWS = 30
TOP_N_BUYS = 10
# ---------------------------------------

# 1) Universe
TICKERS = load_universe(UNIVERSE_CSV)
if not TICKERS:
    raise SystemExit("Universe is empty. Add tickers to src/universe/tickers.csv")

# 2) Prices & basic features
close, volume = get_prices(TICKERS, lookback_days=LOOKBACK_DAYS)
today = close.index.max()
ret5 = close.pct_change(5)
avg_vol20 = volume.rolling(20).mean()

# Last values per ticker
last_close = close.loc[today].dropna()
last_avg_vol20 = avg_vol20.loc[today].reindex(last_close.index)

# 3) Liquidity & price screen
screened = last_close[(last_close > MIN_PRICE) & (last_avg_vol20 > MIN_AVG_VOL_20D)].index.tolist()

# If the screen is too strict, fall back to all that have prices today
if len(screened) < 10:
    screened = last_close.index.tolist()

# 4) Rank by 5-day momentum
r5_today = ret5.loc[today].reindex(screened).dropna()
ranked_by_r5 = r5_today.sort_values(ascending=False)

# 5) Fetch news + FinBERT for the top momentum names only (rate-limit friendly)
candidates = ranked_by_r5.head(TOP_K_FOR_NEWS).index.tolist()
end = dt.datetime.utcnow()
start = end - dt.timedelta(hours=48)

all_rows = []
for t in candidates:
    arts = get_company_news(t, start, end, page_size=50)
    titles = [a.get("title") or "" for a in arts]
    scored = score_texts(titles) if titles else []
    for a, sc in zip(arts, scored):
        all_rows.append({
            "ticker": t,
            "publishedAt": a.get("publishedAt"),
            "title": a.get("title"),
            "url": a.get("url"),
            "sent_label": sc[0],
            "sent_score": float(sc[1]),
        })

articles_df = pd.DataFrame(all_rows)
daily_sent = aggregate_daily_sentiment(articles_df)
sent_today = daily_sent[daily_sent["date"] == dt.datetime.utcnow().date()].set_index("ticker") \
             if not daily_sent.empty else pd.DataFrame(columns=["sent_pos","sent_neg","sent_mean","n_news"])

# 6) Scoring: combine standardized momentum + sentiment
def zscore(s: pd.Series):
    s = s.astype(float)
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / sd if sd > 0 else s*0

r5_z = zscore(ranked_by_r5)

# Sentiment mean (default 0 if missing), aligned to r5_z index
smean = pd.Series(0.0, index=r5_z.index)
nnews = pd.Series(0, index=r5_z.index, dtype=int)
if not sent_today.empty:
    smean.update(sent_today["sent_mean"])
    nnews.update(sent_today["n_news"])

# Combined score: 50% momentum (z), 50% sentiment (raw 0..1)
combined = 0.5 * r5_z + 0.5 * smean

# 7) Decide actions: BUY top N by combined score (with minimal guards)
buys = combined.sort_values(ascending=False).head(TOP_N_BUYS).index.tolist()

# 8) Output rows for all screened tickers (BUY/HOLD)
rows = [["date","ticker","action","qty","entry_price","stop","take_profit","confidence","reasons","features_json"]]
for t in screened:
    lc = float(last_close[t]) if t in last_close else 0.0
    r5 = float(r5_today[t]) if t in r5_today.index else 0.0
    sm = float(smean[t]) if t in smean.index else 0.0
    nn = int(nnews[t]) if t in nnews.index else 0
    action = "BUY" if t in buys else "HOLD"
    # confidence blends normalized momentum + sentiment (clip 0..0.99)
    conf = max(0.0, min(0.99, (max(0, r5)*10)*0.4 + sm*0.6))
    reasons = f"price>{MIN_PRICE}, avgVol20>{MIN_AVG_VOL_20D}, r5={r5:.3f}, sent_mean={sm:.2f}, n_news={nn}, rank={combined.rank(ascending=False).get(t, float('nan')):.0f}"
    feats = {"ret_5d": round(r5,4), "sent_mean": round(sm,2), "n_news": nn, "score": round(float(combined.get(t, 0)),3)}
    rows.append([dt.date.today().isoformat(), t, action, 0, lc, 0, 0, round(conf,2), reasons, str(feats)])

with open("data/signals_latest.csv","w",newline="") as f:
    csv.writer(f).writerows(rows)

print("✅ signals_updated. Buys:", ", ".join(buys) if buys else "None")
