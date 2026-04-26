"""
scraping/news_scraper.py
========================
Fetches recent financial news headlines for a given ticker from multiple
free sources (no API keys required):

  1. yfinance  → ticker.get_news()     — most reliable, always available
  2. Google News RSS                   — broad financial news
  3. Moneycontrol RSS                  — India-specific (RELIANCE.NS etc.)
  4. Economic Times Markets RSS        — India-specific fallback

Returns a flat list of dicts:
  [{ "title": str, "source": str, "url": str, "published": str }, ...]
"""

import os
import sys
import time
import re
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    import xml.etree.ElementTree as ET
    _XML_OK = True
except ImportError:
    _XML_OK = False


# ---------------------------------------------------------------------------
# RSS parser (minimal — no feedparser dependency)
# ---------------------------------------------------------------------------
def _parse_rss(xml_text: str, source_name: str, max_items: int = 15) -> list:
    """Parse a simple RSS 2.0 feed and return article dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
        # Handle both direct <channel> and namespaced roots
        channel = root.find("channel") or root
        for item in channel.findall("item")[:max_items]:
            title  = (item.findtext("title") or "").strip()
            link   = (item.findtext("link")  or "").strip()
            pubdate= (item.findtext("pubDate") or
                      item.findtext("{http://purl.org/dc/elements/1.1/}date") or
                      datetime.now(timezone.utc).isoformat())
            if title:
                articles.append({
                    "title":     title,
                    "source":    source_name,
                    "url":       link,
                    "published": pubdate,
                })
    except Exception as e:
        pass
    return articles


def _fetch_rss(url: str, source_name: str, timeout: int = 8) -> list:
    """Fetch and parse an RSS feed URL."""
    if not _REQUESTS_OK:
        return []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return _parse_rss(resp.text, source_name)
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Source 1: yfinance built-in news
# ---------------------------------------------------------------------------
def fetch_yfinance_news(ticker: str, max_articles: int = 20) -> list:
    """
    Use yfinance's .get_news() or .news attribute to pull recent headlines.
    Handles both old (dict with 'title') and new (nested 'content') formats.
    """
    articles = []
    try:
        t    = yf.Ticker(ticker)
        news = getattr(t, "get_news", None)
        news = news() if callable(news) else t.news
        if not news:
            return []

        for item in news[:max_articles]:
            # New yfinance format: item["content"]["title"]
            if "content" in item and isinstance(item["content"], dict):
                content   = item["content"]
                title     = content.get("title", "")
                url       = (content.get("canonicalUrl", {}) or {}).get("url", "")
                provider  = (content.get("provider",    {}) or {}).get("displayName", "Yahoo Finance")
                pub_raw   = content.get("pubDate", "")
            else:
                # Old format: flat dict
                title    = item.get("title", "")
                url      = item.get("link",  item.get("url", ""))
                provider = item.get("publisher", "Yahoo Finance")
                pub_raw  = item.get("providerPublishTime", "")
                if isinstance(pub_raw, (int, float)):
                    pub_raw = datetime.fromtimestamp(pub_raw, tz=timezone.utc).isoformat()

            if title:
                articles.append({
                    "title":     title.strip(),
                    "source":    provider,
                    "url":       url,
                    "published": str(pub_raw),
                })
    except Exception as e:
        print(f"[NewsScraper] yfinance news error: {e}")
    return articles


# ---------------------------------------------------------------------------
# Source 2: Google News RSS (company name search)
# ---------------------------------------------------------------------------
def fetch_google_news(query: str, max_articles: int = 15) -> list:
    """Fetch Google News RSS for a search query."""
    encoded = query.replace(" ", "+").replace("&", "%26")
    url = (f"https://news.google.com/rss/search?"
           f"q={encoded}+stock&hl=en-IN&gl=IN&ceid=IN:en")
    return _fetch_rss(url, "Google News", max_articles)


# ---------------------------------------------------------------------------
# Source 3: Moneycontrol RSS (NSE stocks)
# ---------------------------------------------------------------------------
_MONEYCONTROL_RSS = {
    "markets":  "https://www.moneycontrol.com/rss/latestnews.xml",
    "business": "https://www.moneycontrol.com/rss/business.xml",
}

def fetch_moneycontrol_news(max_articles: int = 15) -> list:
    articles = []
    for name, url in _MONEYCONTROL_RSS.items():
        articles.extend(_fetch_rss(url, f"Moneycontrol/{name}", max_articles // 2))
        time.sleep(0.3)
    return articles[:max_articles]


# ---------------------------------------------------------------------------
# Source 4: Economic Times RSS
# ---------------------------------------------------------------------------
_ET_RSS = "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"

def fetch_et_markets_news(max_articles: int = 15) -> list:
    return _fetch_rss(_ET_RSS, "Economic Times", max_articles)


# ---------------------------------------------------------------------------
# Master fetch function
# ---------------------------------------------------------------------------
def fetch_all_news(
    ticker:      str,
    company:     str = "",
    max_total:   int = 30,
    sources:     list = None,
) -> list:
    """
    Fetch news from all available sources, deduplicate by title, and return
    up to max_total articles sorted by recency.

    Parameters
    ----------
    ticker    : Yahoo Finance ticker (e.g. "RELIANCE.NS")
    company   : human-readable company name for Google News query
    max_total : maximum articles to return
    sources   : list of source names to use (None = all)
                options: ["yfinance", "google", "moneycontrol", "et"]

    Returns
    -------
    list of dicts: [{"title", "source", "url", "published"}, ...]
    """
    all_sources = sources or ["yfinance", "google", "moneycontrol", "et"]
    articles    = []

    query = company or ticker.split(".")[0]

    if "yfinance" in all_sources:
        yfin = fetch_yfinance_news(ticker, max_articles=20)
        print(f"[NewsScraper] yfinance: {len(yfin)} articles")
        articles.extend(yfin)

    if "google" in all_sources:
        goog = fetch_google_news(query, max_articles=15)
        print(f"[NewsScraper] Google News: {len(goog)} articles")
        articles.extend(goog)

    if "moneycontrol" in all_sources:
        mc = fetch_moneycontrol_news(max_articles=15)
        print(f"[NewsScraper] Moneycontrol: {len(mc)} articles")
        articles.extend(mc)

    if "et" in all_sources:
        et = fetch_et_markets_news(max_articles=15)
        print(f"[NewsScraper] Economic Times: {len(et)} articles")
        articles.extend(et)

    # Deduplicate by normalised title
    seen    = set()
    unique  = []
    for art in articles:
        key = re.sub(r"\W+", " ", art["title"].lower()).strip()
        if key not in seen and len(key) > 10:
            seen.add(key)
            unique.append(art)

    print(f"[NewsScraper] Total unique articles: {len(unique)}")
    return unique[:max_total]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = fetch_all_news("RELIANCE.NS", company="Reliance Industries")
    for r in results[:5]:
        print(f"  [{r['source']}] {r['title'][:80]}")
