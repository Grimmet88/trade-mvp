import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_universe(path: str = "src/universe/tickers.csv") -> list[str]:
    """
    Loads ticker symbols from a CSV file, cleans them, and returns a sorted list.
    
    Args:
        path: Path to the CSV file containing the 'ticker' column.

    Returns:
        A sorted list of unique, uppercase ticker symbols.
    """
    file_path = Path(path)
    
    if not file_path.exists():
        logging.error(f"Universe file not found: {path}")
        # Return an empty list instead of crashing the pipeline
        return []
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error reading CSV file {path}: {e}")
        return []

    if 'ticker' not in df.columns:
        logging.error(f"Required column 'ticker' not found in {path}. Columns found: {df.columns.tolist()}")
        return []

    # Clean, uppercase, and drop duplicates using a set comprehension
    try:
        # Use a list comprehension to ensure we only get valid strings
        raw_tickers = df['ticker'].dropna().astype(str).tolist()
        
        tickers = sorted({t.strip().upper() for t in raw_tickers if t.strip()})
        
        logging.info(f"Loaded {len(tickers)} unique tickers from {path}.")
        return tickers

    except Exception as e:
        logging.error(f"Error processing 'ticker' column data: {e}")
        return []
