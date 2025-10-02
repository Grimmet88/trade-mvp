import pandas as pd

def aggregate_daily_sentiment(articles_df: pd.DataFrame):
    # Requires cols: ticker, publishedAt, title, sent_label, sent_score
    if articles_df.empty:
        return pd.DataFrame(columns=["ticker","date","sent_pos","sent_neg","sent_mean","n_news"])
    df = articles_df.copy()
    df["date"] = pd.to_datetime(df["publishedAt"]).dt.date
    return (df.groupby(["ticker","date"])
              .apply(lambda g: pd.Series({
                  "sent_pos": (g["sent_label"]=="positive").mean() if len(g) else 0.0,
                  "sent_neg": (g["sent_label"]=="negative").mean() if len(g) else 0.0,
                  "sent_mean": g["sent_score"].mean() if len(g) else 0.0,
                  "n_news": len(g)
              }))
              .reset_index())
