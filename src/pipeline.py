import os, csv, datetime as dt

# Make sure the data folder exists
os.makedirs("data", exist_ok=True)

# Create a fake signal row (just a placeholder so you can test the pipeline)
today = dt.date.today().isoformat()
rows = [
    ["date","ticker","action","qty","entry_price","stop","take_profit","confidence","reasons","features_json"],
    [today,"AAPL","HOLD",0,0,0,0,0.50,"Bootstrap test","{}"]
]

with open("data/signals_latest.csv","w",newline="") as f:
    csv.writer(f).writerows(rows)

print("✅ Wrote data/signals_latest.csv")


