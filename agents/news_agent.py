"""
agents/news_agent.py
=====================
News Sentiment Agent — scrapes recent headlines for the configured ticker
and runs FinBERT (or distilbert as fallback) to produce a sentiment score.

Output schema (compatible with Orchestrator / MCP Connector)
-------------------------------------------------------------
{
    "sentiment_label":  "Positive" | "Negative" | "Neutral",
    "sentiment_score":  float ∈ [-1.0, 1.0],
    "headline_summary": str,           # top 3 headlines joined
    "num_articles":     int,
    "source_breakdown": {              # articles per source
        "yfinance": int, "google": int, ...
    },
    "model_used":       str,           # "finbert" | "distilbert" | "keyword"
}

Sentiment models tried in order:
  1. ProsusAI/finbert  — financial domain, outputs positive/negative/neutral
  2. distilbert-base-uncased-finetuned-sst-2-english — general, POSITIVE/NEGATIVE
  3. Keyword fallback  — rule-based, no model needed (always works)
"""

import os
import sys
import re
from collections import Counter
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TICKER
from scraping.news_scraper import fetch_all_news

# ---------------------------------------------------------------------------
# Keyword fallback (no ML needed)
# ---------------------------------------------------------------------------
_POSITIVE_WORDS = {
    "profit", "gain", "growth", "rise", "rally", "surge", "beat", "record",
    "strong", "positive", "upgrade", "buy", "bullish", "expand", "dividend",
    "revenue", "earnings", "outperform", "exceeds", "wins", "acquires",
    "partnership", "launch", "milestone", "recovery", "optimistic",
}
_NEGATIVE_WORDS = {
    "loss", "fall", "drop", "decline", "slump", "miss", "weak", "downgrade",
    "sell", "bearish", "contract", "layoff", "cut", "debt", "risk", "concern",
    "fraud", "penalty", "fine", "lawsuit", "volatile", "crash", "worry",
    "disappoints", "below", "negative", "reduce",
}

def _keyword_sentiment(texts: list) -> tuple:
    """Rule-based fallback sentiment. Returns (label, score ∈ [-1,1])."""
    pos_count = neg_count = 0
    for text in texts:
        words = set(re.sub(r"[^\w\s]", " ", text.lower()).split())
        pos_count += len(words & _POSITIVE_WORDS)
        neg_count += len(words & _NEGATIVE_WORDS)
    total = pos_count + neg_count
    if total == 0:
        return "Neutral", 0.0
    score = (pos_count - neg_count) / total
    label = "Positive" if score > 0.1 else ("Negative" if score < -0.1 else "Neutral")
    return label, round(score, 4)


# ---------------------------------------------------------------------------
# FinBERT / DistilBERT sentiment
# ---------------------------------------------------------------------------
def _load_sentiment_pipeline():
    """
    Try to load FinBERT (financial domain). Falls back to DistilBERT.
    Returns (pipeline, model_name) or (None, "keyword").
    """
    try:
        from transformers import pipeline
        print("[NewsAgent] Loading ProsusAI/finbert …")
        pipe = pipeline(
            "text-classification",
            model     = "ProsusAI/finbert",
            tokenizer = "ProsusAI/finbert",
            device    = -1,   # CPU (GPU if available via device_map="auto")
        )
        print("[NewsAgent] FinBERT loaded ✅")
        return pipe, "finbert"
    except Exception as e:
        print(f"[NewsAgent] FinBERT unavailable ({e}), trying DistilBERT …")

    try:
        from transformers import pipeline
        pipe = pipeline(
            "sentiment-analysis",
            model  = "distilbert-base-uncased-finetuned-sst-2-english",
            device = -1,
        )
        print("[NewsAgent] DistilBERT loaded ✅")
        return pipe, "distilbert"
    except Exception as e:
        print(f"[NewsAgent] DistilBERT unavailable ({e}), using keyword fallback.")

    return None, "keyword"


def _run_pipeline(pipe, model_name: str, texts: list) -> tuple:
    """
    Run the loaded pipeline on text chunks and aggregate to one score.

    FinBERT outputs:  positive / negative / neutral
    DistilBERT:       POSITIVE / NEGATIVE  (no neutral)

    Returns (label, score ∈ [-1, 1])
    """
    if not texts:
        return "Neutral", 0.0

    # Truncate texts to model max length (512 tokens ≈ 400 words)
    clean_texts = [t[:512] for t in texts if len(t.strip()) > 10]
    if not clean_texts:
        return "Neutral", 0.0

    try:
        results = pipe(clean_texts, truncation=True, max_length=512, batch_size=8)
    except Exception as e:
        print(f"[NewsAgent] Inference error: {e}")
        return _keyword_sentiment(texts)

    scores = []
    for r in results:
        label = r["label"].lower()
        conf  = float(r["score"])
        if label in ("positive",):
            scores.append(+conf)
        elif label in ("negative",):
            scores.append(-conf)
        else:
            scores.append(0.0)

    if not scores:
        return "Neutral", 0.0

    avg_score = float(sum(scores) / len(scores))
    label = "Positive" if avg_score > 0.1 else ("Negative" if avg_score < -0.1 else "Neutral")
    return label, round(avg_score, 4)


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------
def _build_summary(articles: list, max_headlines: int = 3) -> str:
    """Join the top headlines into a 1-2 sentence summary."""
    titles = [a["title"] for a in articles[:max_headlines]]
    if not titles:
        return "No recent news available."
    return " | ".join(titles)


# ---------------------------------------------------------------------------
# Main News Agent
# ---------------------------------------------------------------------------
class NewsAgent:
    """
    Scrapes recent financial news and runs sentiment analysis.

    Usage
    -----
        agent  = NewsAgent()
        result = agent.run()
        # result["sentiment_score"] ∈ [-1, 1]
    """

    def __init__(
        self,
        ticker:       str  = TICKER,
        company:      str  = "Reliance Industries",
        max_articles: int  = 30,
        sources:      list = None,   # None = all available
        lazy_load:    bool = True,   # load model on first call, not at init
    ):
        self.ticker       = ticker
        self.company      = company
        self.max_articles = max_articles
        self.sources      = sources
        self._pipe        = None
        self._model_name  = None

        if not lazy_load:
            self._pipe, self._model_name = _load_sentiment_pipeline()

    def _ensure_model(self):
        if self._pipe is None and self._model_name is None:
            self._pipe, self._model_name = _load_sentiment_pipeline()

    # ── Main run ──────────────────────────────────────────────────────────
    def run(self) -> dict:
        """
        Fetch news + run sentiment. Returns dict matching Orchestrator schema.
        """
        print(f"\n[NewsAgent] Fetching news for {self.ticker} ({self.company}) …")

        # Step 1: Fetch headlines from all sources
        articles = fetch_all_news(
            ticker    = self.ticker,
            company   = self.company,
            max_total = self.max_articles,
            sources   = self.sources,
        )

        if not articles:
            print("[NewsAgent] ⚠️  No articles fetched — returning neutral sentiment.")
            return self._neutral_result(0)

        # Step 2: Load model (lazy)
        self._ensure_model()

        # Step 3: Run sentiment
        titles = [a["title"] for a in articles]

        if self._pipe is not None:
            label, score = _run_pipeline(self._pipe, self._model_name, titles)
        else:
            label, score = _keyword_sentiment(titles)
            self._model_name = "keyword"

        # Step 4: Source breakdown
        source_counts = dict(Counter(a["source"].split("/")[0] for a in articles))

        # Step 5: Summary
        summary = _build_summary(articles, max_headlines=3)

        result = {
            "sentiment_label":  label,
            "sentiment_score":  score,
            "headline_summary": summary,
            "num_articles":     len(articles),
            "source_breakdown": source_counts,
            "model_used":       self._model_name or "keyword",
        }

        print(f"\n[NewsAgent] Sentiment: {label}  Score: {score:+.3f}  "
              f"Articles: {len(articles)}  Model: {result['model_used']}")
        print(f"[NewsAgent] Summary: {summary[:120]}…")
        return result

    @staticmethod
    def _neutral_result(n: int) -> dict:
        return {
            "sentiment_label":  "Neutral",
            "sentiment_score":  0.0,
            "headline_summary": "No recent news available.",
            "num_articles":     n,
            "source_breakdown": {},
            "model_used":       "none",
        }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent  = NewsAgent(ticker=TICKER, company="Reliance Industries")
    result = agent.run()
    print("\n── Result ──")
    for k, v in result.items():
        if k != "source_breakdown":
            print(f"  {k:<22}: {v}")
