import pandas as pd
from pathlib import Path

def load_universe(path="src/universe/tickers.csv"):
    df = pd.read_csv(path)
    # Clean, uppercase, and drop dupes
    tickers = sorted({str(t).strip().upper() for t in df['ticker'].dropna().tolist()})
    return [t for t in tickers if t]  # no empties
