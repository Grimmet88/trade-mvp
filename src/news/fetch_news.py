import os, requests, datetime as dt
from dotenv import load_dotenv
load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
BASE = "https://newsapi.org/v2/everything"

def get_company_news(query: str, from_dt: dt.datetime, to_dt: dt.datetime, page_size=50):
    if not NEWSAPI_KEY:
        return []  # No key configured → graceful empty
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "from": from_dt.isoformat(timespec="seconds"),
        "to": to_dt.isoformat(timespec="seconds"),
    }
    r = requests.get(BASE, params=params, headers={"X-Api-Key": NEWSAPI_KEY}, timeout=30)
    r.raise_for_status()
    return r.json().get("articles", [])
