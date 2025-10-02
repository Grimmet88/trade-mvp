# src/scrapers/fetch_reddit.py
import os, time, datetime as dt
from typing import List, Dict
import praw
from dotenv import load_dotenv

load_dotenv()

CID  = os.getenv("REDDIT_CLIENT_ID")
CSEC = os.getenv("REDDIT_CLIENT_SECRET")
UA   = os.getenv("REDDIT_USER_AGENT", "trade-mvp/1.0")

def _client():
    if not (CID and CSEC and UA):
        return None
    return praw.Reddit(
        client_id=CID,
        client_secret=CSEC,
        user_agent=UA,
        ratelimit_seconds=2,  # be polite
    )

def fetch_reddit_posts(subreddits: List[str], lookback_hours: int = 72, limit_per_sub: int = 200) -> List[Dict]:
    """
    Pull recent posts from given subreddits. Returns list of dicts with
    keys: source, subreddit, created_utc, title, selftext, url.
    """
    r = _client()
    if r is None:
        return []

    cutoff = dt.datetime.utcnow().timestamp() - lookback_hours * 3600
    out: List[Dict] = []
    for sub in subreddits:
        try:
            for p in r.subreddit(sub).new(limit=limit_per_sub):
                if float(p.created_utc) < cutoff:
                    continue
                out.append({
                    "source": "reddit",
                    "subreddit": sub,
                    "created_utc": float(p.created_utc),
                    "title": p.title or "",
                    "selftext": p.selftext or "",
                    "url": p.url or "",
                })
            time.sleep(0.5)  # small pause between subs
        except Exception:
            # swallow errors so one bad sub doesn't kill the run
            continue
    return out

