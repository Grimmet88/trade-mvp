# src/features/aggregate_reddit.py
import pandas as pd
import datetime as dt
from typing import Dict, List
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _compile_map(ticker_to_terms: Dict[str, List[str]]):
    """Build a case-insensitive regex per ticker to match any of its terms."""
    pat = {}
    for t, terms in ticker_to_terms.items():
        # Ensure terms are escaped for regex safety
        escaped = [re.escape(x) for x in terms if x]
        if not escaped:
            continue
        # \b ensures whole-word match; (?i) makes it case-insensitive
        pat[t] = re.compile(r"(?i)\b(" + "|".join(escaped) + r")\b")
    return pat

def tag_tickers(posts_df: pd.DataFrame, ticker_to_terms: Dict[str, List[str]]) -> pd.DataFrame:
    """Return rows tagged with first matching ticker based on title/selftext."""
    if posts_df.empty:
        return posts_df.assign(ticker=None)
        
    pats = _compile_map(ticker_to_terms)
    # Combine title and selftext into a single searchable column
    texts = (posts_df["title"].fillna("") + " " + posts_df["selftext"].fillna("")).tolist()
    
    tickers = []
    
    # NOTE: This loop is O(N_posts * N_tickers) and is the current performance bottleneck.
    for txt in texts:
        hit = None
        for t, rgx in pats.items():
            if rgx.search(txt):
                hit = t
                break
        tickers.append(hit)
        
    out = posts_df.copy()
    out["ticker"] = tickers
    
    # Return only posts that successfully matched a ticker
    return out.dropna(subset=["ticker"])


def aggregate_daily_reddit(tagged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: cols [ticker, created_utc, sent_score (continuous [-1, 1])]
    Output: per-ticker per-UTC-date: sent_reddit_mean (raw), sent_reddit_norm ([0, 1]), n_reddit
    """
    output_cols = ["ticker", "date", "sent_reddit_mean", "sent_reddit_norm", "n_reddit"]
    if tagged_df.empty:
        return pd.DataFrame(columns=output_cols)
        
    df = tagged_df.copy()
    
    # 1. Convert timestamp (seconds since epoch) to UTC date object
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.date
    
    # Ensure score is numeric
    df['sent_score'] = pd.to_numeric(df['sent_score'], errors='coerce') 

    # 2. Group and calculate raw mean and count
    agg_results = df.groupby(["ticker", "date"]).agg(
        sent_reddit_mean=('sent_score', 'mean'), # This is the raw [-1, 1] mean
        n_reddit=('sent_score', 'count')
    ).reset_index()

    # 3. Normalize the raw mean [-1, 1] into the [0, 1] range
    # Formula: (x + 1) / 2
    agg_results['sent_reddit_norm'] = (agg_results['sent_reddit_mean'] + 1) / 2
    agg_results['sent_reddit_norm'] = agg_results['sent_reddit_norm'].clip(0, 1) # Cap for safety

    logging.info(f"Aggregated Reddit sentiment for {len(agg_results)} date-ticker pairs.")

    return agg_results[output_cols]
