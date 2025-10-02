import pandas as pd, datetime as dt, os, html

os.makedirs("reports", exist_ok=True)
df = pd.read_csv("data/signals_latest.csv")

# Basic sorting: BUYs first, then by confidence desc
df["__rank"] = (df["action"]!="BUY").astype(int)
df = df.sort_values(["__rank","confidence"], ascending=[True, False])

ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

def row(r):
    return f"""
      <tr>
        <td>{html.escape(str(r['date']))}</td>
        <td>{html.escape(str(r['ticker']))}</td>
        <td><strong>{html.escape(str(r['action']))}</strong></td>
        <td style="text-align:right">{int(r['qty'])}</td>
        <td style="text-align:right">{float(r['entry_price']):.2f}</td>
        <td style="text-align:right">{float(r['stop']):.2f}</td>
        <td style="text-align:right">{float(r['take_profit']):.2f}</td>
        <td style="text-align:right">{float(r['confidence']):.2f}</td>
        <td>{html.escape(str(r['reasons']))}</td>
      </tr>
    """

rows_html = "\n".join(row(r) for _, r in df.iterrows())

html_doc = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Daily Trade Signals</title>
<style>
  body {{ font: 16px/1.4 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 24px; }}
  h1 {{ margin: 0 0 6px; }}
  .sub {{ color:#666; margin-bottom:20px; }}
  table {{ width:100%; border-collapse: collapse; }}
  th, td {{ border-bottom: 1px solid #eee; padding: 8px 6px; }}
  th {{ text-align:left; background:#fafafa; position:sticky; top:0; }}
  tr:hover td {{ background:#fffdf0; }}
  .buy {{ background:#e8fff0; }}
</style>
<h1>Daily Trade Signals</h1>
<div class="sub">Generated {ts}</div>
<table>
  <thead>
    <tr>
      <th>Date</th><th>Ticker</th><th>Action</th><th style="text-align:right">Qty</th>
      <th style="text-align:right">Entry</th><th style="text-align:right">Stop</th>
      <th style="text-align:right">Target</th><th style="text-align:right">Conf</th><th>Reason</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
<p style="margin-top:20px;color:#777">Source: data/signals_latest.csv</p>
</html>
"""
open("reports/index.html","w").write(html_doc)
print("✅ Wrote reports/index.html")
