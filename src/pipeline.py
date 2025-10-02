# src/pipeline.py
import os, csv, datetime as dt
import pandas as pd
import numpy as np

from src.universe.load_universe import load_universe
from src.data.fetch_prices import get_prices
from src.news.fetch_news import get_company_news
from src.nlp.sentiment import score_texts
from src.features.aggregate_sentiment import aggregate_daily_sentiment
from src.portfolio.positions import load_positions, save_positions, open_position, close_position

# ---------------- Settings ----------------
UNIVERSE_CSV    = "src/universe/tickers.csv"
LOOKBACK_DAYS   = 180
MIN_PRICE       = 5.0
MIN_AVG_VOL_20D = 2_000_000

TOP_K_FOR_NEWS  = 15               # limit API usage
TOP_N_BUYS      = 10

# Exit rules
HOLD_DAYS_MAX   = 3                # time exit
STOP_LOSS       = 0.08             # -8%
TAKE_PROFIT     = 0.05             # +5%
SENT_EXIT_MIN   = 0.20             # weak sentiment
# ------------------------------------------

os.makedirs("data", exist_ok=True)

def zscore(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / sd if (sd and sd > 0) else s*0

def main():
    # 1) Universe + data
    tickers = load_universe(UNIVERSE_CSV)
    close, volume = get_prices(tickers, lookback_days=LOOKBACK_DAYS)
    today = close.index.max()
    if pd.isna(today):
        raise SystemExit("No price data.")

    ret5 = close.pct_change(5)
    avg_vol20 = volume.rolling(20).mean()

    last_close = close.loc[today].dropna()
    last_avg_vol20 = avg_vol20.loc[today].reindex(last_close.index)

    # 2) Screen
    screened = last_close[(last_close > MIN_PRICE) & (last_avg_vol20 > MIN_AVG_VOL_20D)].index.tolist()
    if len(screened) < 10:
        screened = last_close.index.tolist()

    # 3) Rank by momentum
    r5_today = ret5.loc[today].reindex(screened).dropna()
    ranked_by_r5 = r5_today.sort_values(ascending=False)

    # 4) News + sentiment for top-K
    candidates = ranked_by_r5.head(TOP_K_FOR_NEWS).index.tolist()
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=48)

    all_rows = []
    for t in candidates:
        try:
            arts = get_company_news(t, start, end, page_size=20)
        except Exception:
            arts = []
        titles = [a.get("title") or "" for a in arts]
        scored = score_texts(titles) if titles else []
        for a, sc in zip(arts, scored):
            all_rows.append({
                "ticker": t,
                "publishedAt": a.get("publishedAt"),
                "title": a.get("title"),
                "sent_label": sc[0],
                "sent_score": float(sc[1]),
            })

    articles_df = pd.DataFrame(all_rows)
    daily_sent = aggregate_daily_sentiment(articles_df)
    sent_today = (
        daily_sent[daily_sent["date"] == dt.datetime.utcnow().date()].set_index("ticker")
        if not daily_sent.empty else
        pd.DataFrame(columns=["sent_pos","sent_neg","sent_mean","n_news"])
    )

    # 5) Combined score
    r5_z = zscore(ranked_by_r5)
    smean = pd.Series(0.0, index=r5_z.index)
    nnews = pd.Series(0, index=r5_z.index, dtype=int)
    if not sent_today.empty:
        smean.update(sent_today.get("sent_mean", pd.Series(dtype=float)))
        nnews.update(sent_today.get("n_news", pd.Series(dtype=int)))
    combined = 0.5 * r5_z + 0.5 * smean
    buys_ranked = combined.sort_values(ascending=False)
    buy_set = set(buys_ranked.head(TOP_N_BUYS).index.tolist())

    # 6) Base BUY/HOLD rows for screened set
    base_rows = []
    for t in screened:
        lc = float(last_close.get(t, 0.0))
        r5 = float(r5_today.get(t, 0.0)) if t in r5_today.index else 0.0
        sm = float(smean.get(t, 0.0)) if t in smean.index else 0.0
        nn = int(nnews.get(t, 0)) if t in nnews.index else 0
        action = "BUY" if t in buy_set else "HOLD"
        conf = max(0.0, min(0.99, (max(0, r5)*10)*0.4 + sm*0.6))
        reasons = f"r5={r5:.3f}, sent_mean={sm:.2f}, n_news={nn}, score={float(combined.get(t, 0.0)):.3f}"
        feats = {"ret_5d": round(r5,4), "sent_mean": round(sm,2), "n_news": nn, "score": round(float(combined.get(t, 0.0)),3)}
        base_rows.append([dt.date.today().isoformat(), t, action, 0, lc, 0, 0, round(conf,2), reasons, str(feats)])

    # 7) SELL logic for current positions
    positions = load_positions()
    sell_rows = []
    if not positions.empty:
        for _, pos in positions.iterrows():
            t = str(pos["ticker"])
            if t not in last_close.index:
                continue
            entry_price = float(pos["entry_price"])
            price = float(last_close[t])
            pnl = (price - entry_price) / entry_price
            days_held = (dt.date.today() - dt.date.fromisoformat(str(pos["entry_date"]))).days
            r5 = float(r5_today.get(t, 0.0)) if t in r5_today.index else 0.0
            sm = float(smean.get(t, 0.0)) if t in smean.index else 0.0

            triggers, do_sell = [], False
            if pnl <= -STOP_LOSS:   do_sell, triggers = True, triggers+[f"stop {-STOP_LOSS*100:.0f}%"]
            if pnl >=  TAKE_PROFIT: do_sell, triggers = True, triggers+[f"take +{TAKE_PROFIT*100:.0f}%"]
            if r5 <= 0:             do_sell, triggers = True, triggers+["momentum<=0"]
            if sm < SENT_EXIT_MIN:  do_sell, triggers = True, triggers+[f"sent<{SENT_EXIT_MIN:.2f}"]
            if days_held >= HOLD_DAYS_MAX: do_sell, triggers = True, triggers+[f"time>{HOLD_DAYS_MAX}d"]

            if do_sell:
                sell_rows.append([
                    dt.date.today().isoformat(), t, "SELL", int(pos.get("qty", 1)), price, 0, 0, 0.80,
                    "; ".join(triggers),
                    str({"pnl": round(float(pnl),4), "ret_5d": round(r5,4), "sent_mean": round(sm,2)})
                ])
                positions = close_position(positions, t)

    # 8) Prevent new BUYs for already-open tickers
    open_set = set(positions["ticker"].tolist()) if not positions.empty else set()
    for row in base_rows:
        if row[1] in open_set and row[2] == "BUY":
            row[2] = "HOLD"

    # 9) Write signals (SELL rows first)
    header = ["date","ticker","action","qty","entry_price","stop","take_profit","confidence","reasons","features_json"]
    with open("data/signals_latest.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for sr in sell_rows: w.writerow(sr)
        for br in base_rows: w.writerow(br)

    # 10) Update ledger with fresh BUYs (qty=1 for paper)
    updated = positions.copy()
    for row in base_rows:
        d, t, action, _, entry_price, *_ = row
        if action == "BUY":
            updated = open_position(updated, t, 1, float(entry_price), d)
    save_positions(updated)

    print("✅ signals_updated.",
          f"Sells: {', '.join([r[1] for r in sell_rows]) or 'None'} |",
          f"Buys: {', '.join(sorted(list(buy_set)))}")

if __name__ == "__main__":
    main()

