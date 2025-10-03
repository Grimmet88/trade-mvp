import os
import pandas as pd
import numpy as np

# Silence HF tokenizer fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Example: Load ticker universe
def load_universe(csv_path):
    return pd.read_csv(csv_path)["ticker"].tolist()

# Example: Fetch prices (stub, replace with your logic)
def get_prices(tickers, lookback_days=180):
    # Replace with your own data source!
    dates = pd.date_range(end=pd.Timestamp.today(), periods=lookback_days)
    prices = pd.DataFrame(index=dates, columns=tickers)
    prices = prices.fillna(np.random.uniform(10, 200, size=prices.shape))  # Random price stub
    return prices

# Momentum z-score
def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return (s - np.nanmean(s)) / np.nanstd(s)

def main():
    UNIVERSE_CSV = "src/universe/tickers.csv"
    TOP_N_BUYS = 10

    # 1. Load tickers
    tickers = load_universe(UNIVERSE_CSV)

    # 2. Fetch prices
    prices = get_prices(tickers)

    # 3. Calculate momentum (simple: last price - mean)
    momentum = prices.iloc[-1] - prices.mean()
    ranked = zscore(momentum).sort_values(ascending=False)

    # 4. Select top candidates
    top_candidates = ranked.head(TOP_N_BUYS)
    print("Top Candidates:")
    print(top_candidates)

    # Optionally return for Streamlit use
    return top_candidates

if __name__ == "__main__":
    main()
