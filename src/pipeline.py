import os, csv, datetime as dt
import pandas as pd

from src.data.fetch_prices import get_prices
from src.news.fetch_news import get_company_news
from src.nlp.sentiment import score_texts
from src.features.aggregate_sentiment import aggregate_daily_sentiment

os.makedirs("data", exist_ok=True)

TICKERS = ["AAPL","MSFT","NVDA"]

# 1) Prices & 5d momentum
close = get_prices(TICKERS, lookback_days=120)
ret5 = close.pct_change(5)
today = close.index.max()

# 2) News last 48h → FinBERT sentiment
end = dt.datetime.utcnow()
start = end - dt.timedelta(hours=48)

all_rows = []
for t in TICKERS:
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

# Pull the most recent sentiment row per ticker (today UTC date)
sent_today = daily_sent[daily_sent["date"] == dt.datetime.utcnow().date()].set_index("ticker") \
             if not daily_sent.empty else pd.DataFrame(columns=["sent_pos","sent_neg","sent_mean","n_news"])

# 3) Build signals
rows = [["date","ticker","action","qty","entry_price","stop","take_profit","confidence","reasons","features_json"]]
for t in TICKERS:
    last_close = float(close.loc[today, t])
    r5 = float(ret5.loc[today, t])
    smean = float(sent_today.loc[t,"sent_mean"]) if t in sent_today.index else 0.0
    nnews = int(sent_today.loc[t,"n_news"]) if t in sent_today.index else 0

    # Rule: buy only if BOTH momentum and sentiment are positive enough
    buy = (r5 > 0) and (smean >= 0.60) and (nnews >= 2)
    action = "BUY" if buy else "HOLD"

    # Confidence: blend normalized r5 + sentiment mean
    conf = max(0.0, min(0.99, (r5*10)*0.4 + smean*0.6))

    reason = []
    reason.append("5d momentum > 0" if r5>0 else "5d momentum ≤ 0")
    reason.append(f"sent_mean={smean:.2f}")
    reason.append(f"n_news={nnews}")
    reasons = "; ".join(reason)

    feats = {"ret_5d": round(r5,4), "sent_mean": round(smean,2), "n_news": nnews}
    rows.append([
        dt.date.today().isoformat(), t, action, 0, last_close, 0, 0, round(conf,2), reasons, str(feats)
    ])

with open("data/signals_latest.csv","w",newline="") as f:
    csv.writer(f).writerows(rows)

print("✅ signals_updated:", " | ".join(f"{r[1]}:{r[2]}" for r in rows[1:]))
