import pandas as pd
import numpy as np
from src.news.rss import fetch_multiple_feeds
from textblob import TextBlob
import re
from collections import defaultdict

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
    text_lower = text.lower()
    mentions = []
    for ticker in tickers:
        company = ticker_company_map.get(ticker, "").lower()
        ticker_pattern = re.compile(rf"\b{ticker.lower()}\b")
        if ticker_pattern.search(text_lower) or company in text_lower:
            mentions.append(ticker)
        elif company and any(word in text_lower for word in company.split()):
            mentions.append(ticker)
    return list(set(mentions))

def aggregate_sentiment(news_results, tickers):
    ticker_sentiments = defaultdict(list)
    for nr in news_results:
        for ticker in nr["tickers"]:
            ticker_sentiments[ticker].append(nr["sentiment"])
    avg_sentiment = {k: (sum(v)/len(v) if v else 0) for k, v in ticker_sentiments.items()}
    for ticker in tickers:
        avg_sentiment.setdefault(ticker, 0)
    return avg_sentiment

def get_action(momentum, sentiment, momentum_thresh=0.5, sentiment_thresh=0.1):
    if momentum > momentum_thresh and sentiment > sentiment_thresh:
        return "Buy"
    elif momentum < -momentum_thresh and sentiment < -sentiment_thresh:
        return "Sell"
    else:
        return "Hold"

def main(return_for_streamlit=True):
    tickers = load_universe("src/data/tickers.csv")
    prices = get_prices(tickers)
    momentum = prices.iloc[-1] - prices.mean()
    ranked = zscore(momentum).sort_values(ascending=False)

    rss_urls = [
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.marketwatch.com/rss/topstories",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://www.bloomberg.com/feed/podcast/etf-report.xml",
        "https://seekingalpha.com/market_currents.xml",
        "https://www.investing.com/rss/news_25.rss",
        "https://www.ft.com/?format=rss",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://www.nasdaq.com/feed/rssoutbound?category=Stock-Market-News"
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

    avg_sentiment = aggregate_sentiment(news_results, tickers)
    decisions = {}
    for ticker in tickers:
        momentum_score = ranked.get(ticker, 0)
        sentiment_score = avg_sentiment.get(ticker, 0)
        action = get_action(momentum_score, sentiment_score)
        decisions[ticker] = {
            "momentum": momentum_score,
            "sentiment": sentiment_score,
            "action": action
        }

    print(f"Found {len(news_results)} relevant articles with ticker mentions!")
    for nr in news_results[:5]:
        print(f"{nr['title']} | {nr['tickers']} | Sentiment: {nr['sentiment']:.2f} | {nr['link']}")

    print("\nBuy/Sell/Hold Signals:")
    for ticker, info in decisions.items():
        print(f"{ticker}: {info['action']} (Momentum: {info['momentum']:.2f}, Sentiment: {info['sentiment']:.2f})")

    print("\nTop Candidates:")
    print(ranked.head(5))

    # Always return three values for Streamlit/dashboard
    return ranked, news_results, decisions

if __name__ == "__main__":
    # Unpack all three return values for direct script usage
    ranked, news_results, decisions = main()
    print("Ranked:", ranked.head())
    print("Sample News Results:", news_results[:2])
    print("Sample Decisions:", {k: decisions[k] for k in list(decisions.keys())[:2]})
