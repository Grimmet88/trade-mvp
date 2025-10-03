import pandas as pd
import numpy as np
import datetime as dt
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION FOR POSITION SIZING ---
# NOTE: These are simulated values for the backtest/paper trading environment.
PORTFOLIO_EQUITY = 100_000.0  # Simulated total capital in USD
RISK_PER_TRADE = 0.01        # Percentage of equity to risk per trade (1%)

POSITIONS_FILE = "data/positions.csv"
os.makedirs("data", exist_ok=True)

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def load_positions() -> pd.DataFrame:
    """Loads the current open positions from the CSV file."""
    try:
        if os.path.exists(POSITIONS_FILE):
            df = pd.read_csv(POSITIONS_FILE, parse_dates=["entry_date"])
            # Ensure the 'qty' is always an integer for safety
            if 'qty' in df.columns:
                df['qty'] = df['qty'].astype(int)
            return df
        else:
            # Create an empty DataFrame if the file doesn't exist
            return pd.DataFrame(columns=["ticker", "qty", "entry_price", "entry_date"])
    except Exception as e:
        logging.error(f"Error loading positions: {e}")
        return pd.DataFrame(columns=["ticker", "qty", "entry_price", "entry_date"])

def save_positions(df: pd.DataFrame):
    """Saves the current open positions DataFrame to the CSV file."""
    if not df.empty:
        df.to_csv(POSITIONS_FILE, index=False)
    else:
        # If the DataFrame is empty, ensure the file is cleared or removed
        if os.path.exists(POSITIONS_FILE):
             os.remove(POSITIONS_FILE)
        logging.info("Positions saved (empty set).")

# ----------------------------------------------------------------------
# Position Sizing Logic (NEW)
# ----------------------------------------------------------------------

def calculate_position_size(entry_price: float, stop_loss_pct: float) -> int:
    """
    Calculates the maximum quantity (qty) to buy based on the fixed risk amount
    (RISK_PER_TRADE) and the trade's stop-loss distance.

    Args:
        entry_price: The current price of the stock.
        stop_loss_pct: The percentage below entry_price where the stop loss is set (e.g., 0.08 for 8%).
    
    Returns:
        The integer quantity (qty) of shares to buy.
    """
    if entry_price <= 0 or stop_loss_pct <= 0:
        return 0

    # 1. Calculate the maximum USD amount we are willing to lose on this trade.
    max_risk_usd = PORTFOLIO_EQUITY * RISK_PER_TRADE

    # 2. Calculate the USD loss per share (Risk Distance per Share)
    # The stop loss is stop_loss_pct * entry_price
    risk_per_share = entry_price * stop_loss_pct

    if risk_per_share <= 0:
        return 0
    
    # 3. Calculate the quantity (Position Size = Total Risk / Risk per Share)
    qty_float = max_risk_usd / risk_per_share
    
    # 4. Return the quantity rounded down to the nearest whole share
    qty = int(np.floor(qty_float))
    
    # Ensure quantity is at least 1, provided the risk is within budget
    if qty < 1 and max_risk_usd > risk_per_share:
        qty = 1
        
    logging.info(f"Sizing: Risk=${max_risk_usd:.2f}, Risk/Share=${risk_per_share:.2f}. Calculated Qty: {qty}")
    return max(0, qty)

# ----------------------------------------------------------------------
# Trade Execution
# ----------------------------------------------------------------------

def open_position(df: pd.DataFrame, ticker: str, qty: int, entry_price: float, date_str: str) -> pd.DataFrame:
    """Opens a new position, adding it to the DataFrame."""
    if qty <= 0:
        logging.warning(f"Attempted to open position for {ticker} with zero quantity.")
        return df

    new_position = {
        "ticker": ticker,
        "qty": qty,
        "entry_price": entry_price,
        "entry_date": dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    }
    
    # Use pandas.concat to add the new row
    new_row = pd.DataFrame([new_position])
    return pd.concat([df, new_row], ignore_index=True)

def close_position(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Closes an existing position by removing it from the DataFrame."""
    initial_len = len(df)
    
    # Find the index of the position to close
    index_to_close = df[df["ticker"] == ticker].index
    
    if index_to_close.empty:
        logging.warning(f"Attempted to close position for {ticker}, but no open position was found.")
        return df

    # Remove the row(s) corresponding to the ticker
    df_updated = df.drop(index_to_close)

    if len(df_updated) < initial_len:
        logging.info(f"Position for {ticker} closed.")
    
    return df_updated
