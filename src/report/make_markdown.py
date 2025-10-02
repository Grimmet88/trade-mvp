import pandas as pd, datetime as dt, os
os.makedirs("reports", exist_ok=True)
df = pd.read_csv("data/signals_latest.csv")
ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
lines = [
  "# Daily Trade Signals",
  f"_Generated: {ts}_",
  "",
  "| Date | Ticker | Action | Qty | Entry | Stop | Target | Conf | Reason |",
  "|---|---|---|---:|---:|---:|---:|---:|---|",
]
for _, r in df.iterrows():
    lines.append(f"| {r.date} | {r.ticker} | {r.action} | {int(r.qty)} | {float(r.entry_price):.2f} | {float(r.stop):.2f} | {float(r.take_profit):.2f} | {float(r.confidence):.2f} | {r.reasons} |")
open("reports/daily_signals.md", "w").write("\n".join(lines))
print("✅ Wrote reports/daily_signals.md")
