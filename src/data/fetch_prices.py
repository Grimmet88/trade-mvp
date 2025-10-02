import yfinance as yf
import pandas as pd
from datetime import date, timedelta

def get_prices(tickers, lookback_days=60):
    end = date.today()
    start = end - timedelta(days=lookback_days)
    data = yf.download(tickers, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=True)
    # yfinance returns a MultiIndex when multiple tickers; normalize to tidy frame
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'].copy()
    else:
        close = pd.DataFrame(data['Close']).rename(columns={'Close': tickers[0]})
    close.index = pd.to_datetime(close.index)
    return close.sort_index()
