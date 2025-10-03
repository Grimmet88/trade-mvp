import pandas as pd
import numpy as np
import datetime as dt
import os
import logging
import random
import json

# --- 1. CONFIGURATION AND IMPORTS ---
# Assuming project structure where subdirectories are on the Python path
try:
    from reporting.log_trades import log_closed_trade
    # NOTE: Added calculate_position_size and RISK_PER_TRADE import
    from portfolio.positions import load_positions, save_positions, open_position, close_position, calculate_position_size, RISK_PER_TRADE
except ImportError as e:
    logging.error(f"Failed to import a module. Ensure all files are in place: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Trading Strategy Parameters ---
STOP_LOSS = 0.08      # 8% below entry price
TAKE_PROFIT = 0.15    # 15% above entry price
MAX_OPEN_POSITIONS = 5

# --- Data Paths ---
SIMULATED_DATA_FILE = "data/simulated_market_data.csv"
SENTIMENT_DATA_FILE = "data/signals_sentiment.csv"
SIGNALS_OUTPUT_FILE = "data/signals_latest.csv"
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# --- 2. Data Simulation (Placeholder for real data fetching) ---

def load_and_prepare_data():
    """Simulates loading the required market and feature data."""
    try:
        # Create dummy market data if it doesn't exist
        if not os.path.exists(SIMULATED_DATA_FILE):
            logging.info("Creating dummy market data for simulation.")
            tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'JPM', 'V', 'NVDA', 'PYPL', 'CRM']
            data = []
            today = dt.date.today()
            for i in range(10): # 10 simulated days
                date = today - dt.timedelta(days=i)
                for t in tickers:
                    # Simulated closing price (lc) and simulated technical indicator (RSI)
                    lc = random.uniform(50, 200)
                    rsi = random.uniform(30, 70)
                    momentum = random.uniform(-0.05, 0.05)
                    data.append([date, t, lc, rsi, momentum])
            
            df_mkt = pd.DataFrame(data, columns=['date', 'ticker', 'lc', 'rsi', 'momentum'])
            df_mkt.to_csv(SIMULATED_DATA_FILE, index=False)
        
        # Load the simulated data
        df_mkt = pd.read_csv(SIMULATED_DATA_FILE)
        
        # Create dummy sentiment data if it doesn't exist
        if not os.path.exists(SENTIMENT_DATA_FILE):
            logging.info("Creating dummy sentiment data.")
            df_sent = pd.DataFrame({
                'ticker': df_mkt['ticker'].unique(),
                'sentiment_score': [random.uniform(-0.5, 0.5) for _ in range(len(df_mkt['ticker'].unique()))]
            })
            df_sent.to_csv(SENTIMENT_DATA_FILE, index=False)
            
        df_sent = pd.read_csv(SENTIMENT_DATA_FILE)
        
        return df_mkt, df_sent
    except Exception as e:
        logging.error(f"Error in data preparation: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 3. Feature Calculation (Placeholder for advanced feature engineering) ---

def calculate_features(df_mkt: pd.DataFrame) -> dict:
    """Calculates signals based on market data."""
    # Use the latest day's data for current signals
    latest_date = df_mkt['date'].max()
    df_latest = df_mkt[df_mkt['date'] == latest_date]
    
    signals = {}
    for _, row in df_latest.iterrows():
        ticker = row['ticker']
        # Feature 1: Simple RSI check
        rsi_signal = 1 if row['rsi'] < 40 else 0
        
        # Feature 2: Momentum check
        momentum_signal = 1 if row['momentum'] > 0.02 else 0
        
        signals[ticker] = {
            'lc': row['lc'],
            'rsi_signal': rsi_signal,
            'momentum_signal': momentum_signal
        }
        
    return signals

# --- 4. Signal Combination and Weights ---

def combine_signals(signals: dict, df_sentiment: pd.DataFrame) -> dict:
    """Combines multiple feature signals with external sentiment using weights."""
    
    # Define weights for each signal (must sum to 1.0)
    # NOTE: Weights corrected to sum to 1.0
    weights = {
        'rsi_signal': 0.3,
        'momentum_signal': 0.3,
        'sentiment_score': 0.4 
    }
    
    # Normalize sentiment scores to be positive/negative indicators
    sentiment_map = df_sentiment.set_index('ticker')['sentiment_score'].to_dict()
    
    combined_scores = {}
    for ticker, feats in signals.items():
        # Get sentiment (default to 0 if missing)
        sentiment_val = sentiment_map.get(ticker, 0.0)
        
        # Normalize sentiment to a 0-1 range for combination
        # Assuming sentiment range is -0.5 to 0.5, we shift it: (sentiment + 0.5)
        normalized_sentiment = (sentiment_val + 0.5)
        
        # Calculate combined score (weighted average)
        score = (
            (feats['rsi_signal'] * weights['rsi_signal']) + 
            (feats['momentum_signal'] * weights['momentum_signal']) +
            (normalized_sentiment * weights['sentiment_score'])
        )
        combined_scores[ticker] = score
        
        # Add sentiment and combined score to the features dictionary for logging
        feats['sentiment_score'] = sentiment_val
        feats['combined_score'] = score
        
    return combined_scores

# --- 5. Trading Decision Logic ---

def generate_decisions(combined_scores: dict, current_positions: pd.DataFrame) -> tuple[set, set]:
    """Determines which tickers to BUY, SELL, or HOLD."""
    
    buy_set = set()
    sell_set = set()
    
    # 5.1. Identify Sell Signals (Stop Loss / Take Profit)
    current_prices = {t: v['lc'] for t, v in signals.items()}
    
    for _, pos in current_positions.iterrows():
        t = pos['ticker']
        entry = pos['entry_price']
        price = current_prices.get(t)
        qty = pos['qty']
        
        if price is None:
            logging.warning(f"Could not get current price for open position {t}. Skipping risk check.")
            continue
        
        # Calculate PnL percentage
        pnl_pct = (price - entry) / entry
        
        # Calculate stop and target prices
        stop_price = entry * (1 - STOP_LOSS)
        target_price = entry * (1 + TAKE_PROFIT)
        
        triggers = []
        
        # Check Stop Loss
        if price <= stop_price:
            triggers.append("STOP_LOSS")
            
        # Check Take Profit
        if price >= target_price:
            triggers.append("TAKE_PROFIT")

        # Execute Sell if triggered
        if triggers:
            sell_set.add(t)
            pnl = (price - entry) * qty
            
            # --- LOG THE CLOSED TRADE (Integration from log_trades.py) ---
            # NOTE: We log the trade BEFORE removing the position from the DataFrame
            log_closed_trade(t, pos["entry_date"], entry, dt.date.today(), price, int(pos.get("qty", 0)), pnl, pnl_pct, "; ".join(triggers))
            logging.info(f"SELL signal for {t} triggered by: {', '.join(triggers)}. PnL: {pnl:.2f} USD ({pnl_pct*100:.2f}%)")


    # 5.2. Identify Buy Signals
    
    # Sort potential buys by score and filter out current holdings
    potential_buys = {t: score for t, score in combined_scores.items() if t not in current_positions['ticker'].values}
    
    # High score threshold (e.g., top 30% of max possible score)
    BUY_THRESHOLD = 0.8 * sum(weights.values()) # Max score is sum of weights (1.0)
    
    ranked_buys = sorted(potential_buys.items(), key=lambda item: item[1], reverse=True)
    
    # Filter by score threshold and max capacity
    for t, score in ranked_buys:
        if score >= BUY_THRESHOLD and len(current_positions) + len(buy_set) < MAX_OPEN_POSITIONS:
            buy_set.add(t)
            
    return buy_set, sell_set

# --- 6. Main Execution Function ---

def main():
    logging.info("--- Starting Daily Trading Pipeline ---")
    
    # 1. Load Data
    df_mkt, df_sent = load_and_prepare_data()
    if df_mkt.empty:
        logging.error("No market data available. Exiting.")
        return

    # 2. Get Features (Signals)
    global signals # Make signals accessible globally for combined_signals
    signals = calculate_features(df_mkt)
    
    # 3. Load Current Positions
    current_positions = load_positions()
    logging.info(f"Loaded {len(current_positions)} open positions.")
    
    # 4. Combine Signals
    combined_scores = combine_signals(signals, df_sent)
    
    # 5. Determine Trading Decisions (Note: Sell signals are logged inside this function)
    buy_set, sell_set = generate_decisions(combined_scores, current_positions)
    
    logging.info(f"Decisions: BUY {list(buy_set)}, SELL {list(sell_set)}")
    
    # 6. Generate Signal Report Rows (Including all tickers)
    
    rows_base = []
    
    for t, feats in signals.items():
        lc = feats['lc']
        
        action = "HOLD"
        reasons = ""
        conf = 0.0
        
        # Confidence is calculated based on the combined score relative to the maximum possible score (1.0)
        conf = max(0.0, min(0.99, (combined_scores.get(t, 0.0)))) 

        # Current Price Check
        if t in current_positions['ticker'].values:
            action = "HOLD" # Already open, will be closed by SELL logic if triggered
            reasons = "Position currently open."
        
        # BUY signal logic
        elif t in buy_set:
            # --- NEW SIZING LOGIC ---
            # We use the STOP_LOSS constant from the pipeline settings for the sizing calculation
            qty = calculate_position_size(lc, STOP_LOSS)
            
            # If sizing returns 0, it means the trade is too large/risky for the capital, so we don't open it.
            if qty == 0:
                action = "HOLD"
                reasons = f"BUY signal cancelled. Position size is zero due to entry price ({lc:.2f}) vs Risk Budget ({RISK_PER_TRADE*100:.0f}% of portfolio)."
                logging.info(f"Skipping BUY for {t}: Position size calculated as zero based on risk budget.")
            else:
                action = "BUY"
                reasons = f"RSI={feats['rsi_signal']}, Momentum={feats['momentum_signal']}, Sentiment={feats['sentiment_score']:.2f}"
        
        # SELL signal logic (for the report only, closure happens in generate_decisions)
        elif t in sell_set:
            action = "SELL"
            reasons = "Triggered Stop Loss or Take Profit (Check Trade History)."

        # Default HOLD signal for the report
        else:
             reasons = "No strong signal or position currently open/closed."
             qty = 0 # Default quantity for non-buy signals

        # Calculate Stop and Target Prices for the report (only needed for BUY signals)
        stop_price = round(lc * (1 - STOP_LOSS), 2) if action == "BUY" else 0.0
        take_price = round(lc * (1 + TAKE_PROFIT), 2) if action == "BUY" else 0.0
        
        # Append signal row
        rows_base.append([
            dt.date.today().isoformat(), t, action, qty, lc, stop_price, take_price,
            round(conf, 2), reasons, json.dumps(feats)
        ])

    # 7. Write Daily Signals Report
    
    # Convert list of rows to DataFrame and write to file
    df_signals = pd.DataFrame(rows_base, columns=[
        "date", "ticker", "action", "qty", "entry_price", "stop", "take_profit", 
        "confidence", "reasons", "features"
    ])
    df_signals.to_csv(SIGNALS_OUTPUT_FILE, index=False)
    logging.info(f"Wrote daily signals to {SIGNALS_OUTPUT_FILE}")
    
    # Generate the Markdown Report (Placeholder logic from user's initial script)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
      "# Daily Trade Signals",
      f"_Generated: {ts}_",
      "",
      "| Date | Ticker | Action | Qty | Entry | Stop | Target | Conf | Reason |",
      "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df_signals.iterrows():
        lines.append(f"| {r['date']} | {r['ticker']} | {r['action']} | {int(r['qty'])} | {float(r['entry_price']):.2f} | {float(r['stop']):.2f} | {float(r['take_profit']):.2f} | {float(r['confidence']):.2f} | {r['reasons'][:50]}... |")
    open("reports/daily_signals.md", "w").write("\n".join(lines))
    print("✅ Wrote reports/daily_signals.md")
    
    # 8. Update Open Positions (After potential SELL trades have been logged)
    
    updated = current_positions.copy()
    
    # First, process sales (removes from the DataFrame)
    for t in sell_set:
        updated = close_position(updated, t)
        
    # Second, process buys (adds to the DataFrame)
    for row in rows_base:
        # Unpack the row, ensuring 'qty' is captured from the 4th element (index 3)
        d, t, action, qty, entry_price, *_ = row
        if action == "BUY":
            # Use the calculated quantity (qty)
            updated = open_position(updated, t, int(qty), float(entry_price), d)

    # 9. Save Final Positions
    save_positions(updated)
    logging.info(f"Finished pipeline. New open positions count: {len(updated)}")

if __name__ == "__main__":
    main()
