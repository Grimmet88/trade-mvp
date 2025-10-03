import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import logging

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_prices(tickers: list, lookback_days: int = 180):
    """
    Fetches historical stock prices and volume data using yfinance.
    
    Args:
        tickers: List of ticker symbols.
        lookback_days: Number of days to look back from today.

    Returns:
        A tuple of (Close prices DataFrame, Volume DataFrame).
    """
    if not tickers:
        logging.warning("Ticker list is empty.")
        return pd.DataFrame(), pd.DataFrame()

    # Set end date to tomorrow's date. yfinance 'end' is exclusive.
    # This reliably fetches data up to and including the last full market day.
    end_date = date.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days + 1) # +1 day buffer

    try:
        data = yf.download(
            tickers=tickers,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            auto_adjust=True,
            threads=True,
            group_by='ticker' # FORCES MultiIndex for consistent parsing
        )
    except Exception as e:
        logging.error(f"Failed to download prices for {len(tickers)} tickers: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    # 1. Check for empty results after download
    if data.empty or (len(tickers) > 1 and not isinstance(data.columns, pd.MultiIndex)):
         # If group_by='ticker' is used, data columns should always be MultiIndex.
         # For a single ticker, the columns are (Adj Close, Close, High, Low, Open, Volume).
         # For multiple tickers, the columns are (Feature, Ticker).
         logging.warning("Price data download returned no results or unexpected format.")
         return pd.DataFrame(), pd.DataFrame()

    # 2. Extract and Normalize (Simplified due to group_by='ticker' standardization)
    if len(tickers) == 1:
        # Handle single ticker case without MultiIndex
        close = pd.DataFrame(data['Close']).rename(columns={'Close': tickers[0]})
        volume = pd.DataFrame(data['Volume']).rename(columns={'Volume': tickers[0]})
    else:
        # Multi-ticker case
        close = data.xs('Close', axis=1, level=1, drop_level=True).copy()
        volume = data.xs('Volume', axis=1, level=1, drop_level=True).copy()

    # Ensure index is datetime and sort for safety
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    
    # Remove any rows where all values are NaN (e.g., if a ticker was added recently)
    close.dropna(axis=0, how='all', inplace=True)
    volume.dropna(axis=0, how='all', inplace=True)
    
    logging.info(f"Successfully fetched price data from {close.index.min().date()} to {close.index.max().date()}")
    
    return close.sort_index(), volume.sort_index()
