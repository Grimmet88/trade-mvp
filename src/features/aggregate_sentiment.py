# src/features/aggregate_sentiment.py
import pandas as pd
import numpy as np

def aggregate_daily_sentiment(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates sentiment scores (now continuous [-1, 1]) daily by ticker.
    
    EXPECTS cols: ticker, publishedAt, sent_score (the continuous [-1, 1] value).

    Returns:
        DataFrame with aggregated daily metrics, including a [0, 1] normalized mean.
    """
    
    # Define the expected output structure for the pipeline if the input is empty.
    output_cols = ["ticker", "date", "sent_raw_mean", "sent_norm", "n_news"]
    if articles_df.empty:
        return pd.DataFrame(columns=output_cols)
        
    df = articles_df.copy()
    
    # 1. Prepare data (ensure date and score types are correct)
    df["date"] = pd.to_datetime(df["publishedAt"]).dt.date
    # Ensure the score column is the new continuous score from src/nlp/sentiment.py
    df['sent_score'] = pd.to_numeric(df['sent_score'], errors='coerce') 

    # 2. Group by ticker and date and calculate raw mean and count
    agg_results = df.groupby(["ticker", "date"]).agg(
        sent_raw_mean=('sent_score', 'mean'),
        n_news=('sent_score', 'count')
    ).reset_index()

    # 3. Normalize the raw mean [-1, 1] into the [0, 1] range
    # Formula: (x + 1) / 2 
    # This aligns with the pipeline's original assumption of sentiment being [0, 1]
    agg_results['sent_norm'] = (agg_results['sent_raw_mean'] + 1) / 2
    agg_results['sent_norm'] = agg_results['sent_norm'].clip(0, 1) # Cap for safety

    # Rename to match the pipeline's expectations (optional, but good practice)
    agg_results.rename(columns={'sent_raw_mean': 'sent_mean'}, inplace=True)
    
    return agg_results[output_cols]
