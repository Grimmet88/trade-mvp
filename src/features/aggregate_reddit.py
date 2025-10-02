# src/features/aggregate_reddit.py
import pandas as pd
import datetime as dt
from typing import Dict, List
import re

def _compile_map(ticker_to_terms: Dict[str, List[str]]):
    """Build a case-insensitive regex per ticker to match any of its terms."""
    pat = {}
    for t, terms in ticker_to_terms.items():
        escaped = [re.escape(x) for x in terms if x]
        if not escaped:
            continue
        pat[t] = re.compile(r"(?i)\b(" + "|".join(escaped) + r")\b")
    return pat

def tag_tickers(posts_df: pd.DataFrame, ticker_to_terms: Dict[str, List[str]]) -> pd.DataFrame:
    """Return rows tagged with first matching ticker based on title/selftext."""
    if posts_df.empty:
        return posts_df.assign(ticker=None)
    pats = _compile_map(ticker_to_terms)
    texts = (posts_df["title"].fillna("") + " " + posts_df["selftext"].fillna("")).tolist()
    tickers = []
    for txt in texts:
        hit = None
        for t, rgx in pats.items():
            if rgx.search(txt):
                hit = t
                break
        tickers.append(hit)
    out = posts_df.copy()
    out["ticker"] = tickers
    return out.dropna(subset=["ticker"])

def aggregate_daily_reddit(tagged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: cols [ticker, created_utc, sent_label, sent_score]
    Output: per-ticker per-UTC-date: sent_reddit_mean, reddit_pos, reddit_neg, n_reddit
    """
    if tagged_df.empty:
        return pd.DataFrame(columns=["ticker","date","sent_reddit_mean","reddit_pos","reddit_neg","n_reddit"])
    df = tagged_df.copy()
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.date
    return (df.groupby(["ticker","date"])
              .apply(lambda g: pd.Series({
                  "sent_reddit_mean": g["sent_score"].mean() if len(g) else 0.0,
                  "reddit_pos": (g["sent_label"]=="positive").mean() if len(g) else 0.0,
                  "reddit_neg": (g["sent_label"]=="negative").mean() if len(g) else 0.0,
                  "n_reddit": len(g),
              }))
              .reset_index())

