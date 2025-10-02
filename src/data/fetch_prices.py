import yfinance as yf
import pandas as pd
from datetime import date, timedelta

def get_prices(tickers, lookback_days=180):
    end = date.today()
    start = end - timedelta(days=lookback_days)
    data = yf.download(tickers, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=True, threads=True)
    # Normalize
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'].copy()
        volume = data['Volume'].copy()
    else:
        close = pd.DataFrame(data['Close']).rename(columns={'Close': tickers[0]})
        volume = pd.DataFrame(data['Volume']).rename(columns={'Volume': tickers[0]})
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    return close.sort_index(), volume.sort_index()
