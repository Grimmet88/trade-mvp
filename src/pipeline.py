import pandas as pd
import numpy as np
from src.news.rss import fetch_multiple_feeds
from textblob import TextBlob

TICKER_COMPANY_MAP = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "META": "Meta",
    "GOOGL": "Google",
    "GOOG": "Google",
    "TSLA": "Tesla",
    "AVGO": "Broadcom",
    "LLY": "Eli Lilly",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "MA": "Mastercard",
    "COST": "Costco",
    "NFLX": "Netflix",
    "ADBE": "Adobe",
    "CRM": "Salesforce",
    "PEP": "PepsiCo",
    "KO": "Coca-Cola",
    "AMD": "AMD",
    "NKE": "Nike",
    "MCD": "McDonald's",
    "CSCO": "Cisco",
    "ORCL": "Oracle",
    "TXN": "Texas Instruments",
    "TMO": "Thermo Fisher",
    "MRK": "Merck",
    "UNH": "UnitedHealth",
    "WMT": "Walmart",
    "HD": "Home Depot",
    "LIN": "Linde",
    "ABNB": "Airbnb",
    "INTU": "Intuit",
    "IBM": "IBM",
    "PYPL": "PayPal",
    "QCOM": "Qualcomm",
    "PFE": "Pfizer",
    "SBUX": "Starbucks",
    "CAT": "Caterpillar",
    "GE": "General Electric"
}

def load_universe(csv_path):
    df = pd.read_csv(csv_path)
    return df["ticker"].tolist()

def get_prices(tickers, lookback_days=60):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=lookback_days)
    prices = pd.DataFrame(
        np.random.uniform(100, 300, size=(lookback_days, len(tickers))),
        index=dates, columns=tickers
    )
    return prices

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return (s - np.nanmean(s)) / np.nanstd(s)

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity  # [-1.0, 1.0]

def mentions_ticker(article, tickers, ticker_company_map):
    text = (article.get("title", "") or "") + " " + (article.get("summary", "") or "")
    mentions = []
    for ticker in tickers:
        company = ticker_company_map.get(ticker, "")
        # Check for exact ticker symbol (case sensitive) or company name (case insensitive)
        if ticker in text or company.lower() in text.lower():
            mentions.append(ticker)
    return mentions

def main():
    tickers = load_universe("src/data/tickers.csv")
    prices = get_prices(tickers)
    momentum = prices.iloc[-1] - prices.mean()
    ranked = zscore(momentum).sort_values(ascending=False)

    rss_urls = [
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.marketwatch.com/rss/topstories"
    ]
    articles = fetch_multiple_feeds(rss_urls)
    print(f"Fetched {len(articles)} articles!")

    news_results = []
    for a in articles:
        sentiment = analyze_sentiment(str(a.get("title", "")) + " " + str(a.get("summary", "")))
        mentioned = mentions_ticker(a, tickers, TICKER_COMPANY_MAP)
        if mentioned:
            news_results.append({
                "title": a["title"],
                "link": a["link"],
                "sentiment": sentiment,
                "tickers": mentioned
            })

    print(f"Found {len(news_results)} relevant articles with ticker mentions!")
    for nr in news_results[:5]:
        print(f"{nr['title']} | {nr['tickers']} | Sentiment: {nr['sentiment']:.2f} | {nr['link']}")

    print("\nTop Candidates:")
    print(ranked.head(5))

    return ranked, news_results

if __name__ == "__main__":
    main()
