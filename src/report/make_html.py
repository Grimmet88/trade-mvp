# src/report/make_html.py
import os
import html
import datetime as dt
import pandas as pd
import yfinance as yf

OUT_DIR = "reports"
ARCHIVE = True         # also save reports/YYYY-MM-DD.html
SPARK_DAYS = 30        # days of closes shown in sparkline
SPARK_WIDTH = 120
SPARK_HEIGHT = 28
SPARK_PAD = 4

def fetch_sparkline_series(ticker: str, days: int = SPARK_DAYS):
    """Return last ~days closes (list). Uses extra buffer to cover weekends/holidays."""
    end = dt.date.today()
    start = end - dt.timedelta(days=days * 2)
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
    )
    if df.empty or "Close" not in df.columns:
        return []
    return df["Close"].tail(days).tolist()

def to_svg_sparkline(series, width=SPARK_WIDTH, height=SPARK_HEIGHT, pad=SPARK_PAD):
    """Return a tiny inline SVG sparkline for a numeric series."""
    if not series or len(series) < 2:
        return ""
    lo, hi = min(series), max(series)
    span = (hi - lo) or 1e-9
    xs = [pad + i * (width - 2 * pad) / (len(series) - 1) for i in range(len(series))]
    ys = [height - pad - (v - lo) * (height - 2 * pad) / span for v in series]
    path = " ".join(f"L{xs[i]:.1f},{ys[i]:.1f}" for i in range(1, len(series)))
    first = f"M{xs[0]:.1f},{ys[0]:.1f}"
    cx, cy = xs[-1], ys[-1]
    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <path d="{first} {path}" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="2" fill="currentColor"/>
</svg>
""".strip()

def conf_bar(conf: float):
    """Simple horizontal confidence bar 0..1."""
    try:
        c = float(conf)
    except Exception:
        c = 0.0
    pct = max(0, min(99, int(round(c * 100))))
    return f"""
<div class="bar">
  <div class="fill" style="width:{pct}%;"></div>
  <span class="pct">{c:.2f}</span>
</div>
""".strip()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = "data/signals_latest.csv"
    if not os.path.exists(csv_path):
        raise SystemExit("Missing data/signals_latest.csv — run the pipeline first.")

    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required = {"ticker","action","entry_price","confidence","reasons","features_json"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"CSV missing columns: {sorted(required - set(df.columns))}")

    # Sort: SELL first, then BUY by confidence (desc), then HOLD
    df["__rank"] = df["action"].map({"SELL": 0, "BUY": 1}).fillna(2)
    df = df.sort_values(["__rank", "confidence"], ascending=[True, False])

    # Collect sparklines for all tickers present
    spark_by_ticker = {}
    for t in df["ticker"].unique().tolist():
        try:
            series = fetch_sparkline_series(str(t), days=SPARK_DAYS)
            spark_by_ticker[t] = to_svg_sparkline(series)
        except Exception:
            spark_by_ticker[t] = ""

    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    buy_count = int((df["action"] == "BUY").sum())
    sell_count = int((df["action"] == "SELL").sum())

    # Build table rows
    rows_html = []
    for _, r in df.iterrows():
        klass = "sell" if r.action == "SELL" else ("buy" if r.action == "BUY" else "")
        spark = spark_by_ticker.get(r.ticker, "")
        rows_html.append(f"""
<tr class="{klass}">
  <td class="sticky">{html.escape(str(r.ticker))}</td>
  <td>{html.escape(str(r.action))}</td>
  <td class="num">{float(r.entry_price):.2f}</td>
  <td class="num">{float(r.confidence):.2f}{conf_bar(float(r.confidence))}</td>
  <td class="spark">{spark}</td>
  <td class="left">{html.escape(str(r.reasons))}</td>
  <td class="left small">{html.escape(str(r.features_json))}</td>
</tr>""")

    # Full HTML
    html_doc = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Daily Trade Signals</title>
<style>
  :root {{
    --bg:#0b0c10; --card:#111317; --ink:#e6e8eb; --muted:#9aa3ad;
    --buy:#0f5132; --buy-bg:#0a3622; --sell:#842029; --sell-bg:#2c0b0e; --hold-bg:#14161a;
    --bar:#2a2e36; --fill:#4aa3ff;
    font-synthesis-weight:none;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 24px; font: 15px/1.5 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; color:var(--ink); background:var(--bg); }}
  h1 {{ margin: 0 0 6px; font-size: 22px; }}
  .sub {{ color:var(--muted); margin:0 0 18px; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin-left:8px; background:#1c1f25; color:var(--muted); }}
  .wrap {{ border:1px solid #1a1d23; background:var(--card); border-radius:14px; padding:12px; box-shadow: 0 10px 30px rgba(0,0,0,.25); }}
  table {{ width:100%; border-collapse: collapse; }}
  th, td {{ border-bottom:1px solid #1a1d23; padding:10px 8px; vertical-align:middle; }}
  th {{ text-align:left; position:sticky; top:0; background:var(--card); z-index:1; }}
  tr:hover td {{ background:#171a20; }}
  tr.buy td {{ background: var(--buy-bg); }}
  tr.sell td {{ background: var(--sell-bg); }}
  td.sticky {{ position: sticky; left: 0; background:inherit; font-weight:700; }}
  td.num {{ text-align:right; white-space:nowrap; }}
  td.spark svg {{ display:block; }}
  td.left {{ text-align:left; }}
  td.small {{ color:var(--muted); font-size: 12px; }}
  .bar {{ position:relative; height:8px; background:var(--bar); border-radius:6px; overflow:hidden; margin-top:6px; }}
  .fill {{ position:absolute; left:0; top:0; bottom:0; background:var(--fill); }}
  .pct {{ position:absolute; right:6px; top:-18px; font-size:10px; color:var(--muted); }}
  .legend {{ color:var(--muted); font-size:13px; margin-top:10px; }}
</style>

<h1>Daily Trade Signals <span class="pill">{sell_count} SELL · {buy_count} BUY</span></h1>
<div class="sub">Generated {ts}</div>

<div class="wrap">
  <table>
    <thead>
      <tr>
        <th>Ticker</th>
        <th>Action</th>
        <th>Entry</th>
        <th>Confidence</th>
        <th>30d</th>
        <th>Reason</th>
        <th>Features</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
  <div class="legend">SELL rows appear first, then BUY (by confidence), then HOLD. Sparklines show the last ~30 trading days (auto-adjusted close).</div>
</div>
</html>"""

    # Write outputs
    index_path = os.path.join(OUT_DIR, "index.html")
    with open(index_path, "w") as f:
        f.write(html_doc)
    if ARCHIVE:
        d = dt.date.today().isoformat()
        with open(os.path.join(OUT_DIR, f"{d}.html"), "w") as f:
            f.write(html_doc)
    print("✅ Wrote", index_path)

if __name__ == "__main__":
    main()
