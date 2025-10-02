import os, csv, datetime as dt
import pandas as pd
from src.data.fetch_prices import get_prices

os.makedirs("data", exist_ok=True)

TICKERS = ["AAPL","MSFT","NVDA"]  # start small

# 1) Fetch prices (last ~60 calendar days)
close = get_prices(TICKERS, lookback_days=90)

# 2) Compute 5-day returns and today’s last close
ret5 = close.pct_change(5)
today = close.index.max()
rows = [["date","ticker","action","qty","entry_price","stop","take_profit","confidence","reasons","features_json"]]

for t in TICKERS:
    last_close = float(close.loc[today, t])
    r5 = float(ret5.loc[today, t])
    action = "BUY" if r5 > 0 else "HOLD"
    confidence = min(max(r5 * 10, 0), 0.99)  # map small returns to 0..~1
    reason = "5d momentum > 0" if action == "BUY" else "No positive 5d momentum"
    features = {"ret_5d": round(r5, 4)}
    rows.append([
        dt.date.today().isoformat(),
        t, action, 0, last_close, 0, 0, round(confidence, 2), reason, str(features)
    ])

with open("data/signals_latest.csv","w",newline="") as f:
    csv.writer(f).writerows(rows)

print("✅ signals_updated:", " | ".join(f"{r[1]}:{r[2]}" for r in rows[1:]))
