import os
import pandas as pd
import datetime as dt

PATH = "data/positions.csv"

def load_positions():
    """Load positions ledger from CSV (create empty if missing)."""
    if not os.path.exists(PATH):
        cols = ["ticker", "qty", "entry_price", "entry_date"]
        pd.DataFrame(columns=cols).to_csv(PATH, index=False)
    return pd.read_csv(PATH)

def save_positions(df: pd.DataFrame):
    """Save the positions ledger to CSV."""
    df.to_csv(PATH, index=False)

def open_position(positions, ticker, qty, price, date):
    """Add a new open position if not already held."""
    if any(positions["ticker"] == ticker):
        return positions
    new = pd.DataFrame([{
        "ticker": ticker,
        "qty": qty,
        "entry_price": price,
        "entry_date": str(date)
    }])
    return pd.concat([positions, new], ignore_index=True)

def close_position(positions, ticker):
    """Remove a position (after SELL)."""
    return positions[positions["ticker"] != ticker].reset_index(drop=True)

