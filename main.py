"""
main.py
=======
Prediction-only entry point for the Stock Market LLM pipeline.

Workflow (single run, no training):
  1. Download RELIANCE.NS data for the last 15 years ending on the most
     recent trading day available from yfinance.
  2. Engineer technical features on the downloaded data.
  3. Load pre-trained iTransformer and TFT checkpoints from saved_models/.
  4. Predict closing prices for the next 5 trading days from today.
  5. Fetch news sentiment via NewsAgent.
  6. Synthesise everything through OrchestratorAgent and save a JSON result.

Usage
-----
  python main.py          # standard predict run
  python main.py --no-log # skip MCP run logging
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config.settings import (
    TICKER, MODEL_DIR, FEAT_CSV, RAW_CSV,
    ALL_FEATURES, SENTIMENT_CACHE_PATH,
)
from agents.orchestrator_agent import OrchestratorAgent

SENTIMENT_CACHE_TTL = 24 * 3600   # seconds


# ---------------------------------------------------------------------------
# STEP 0 — Download last 15 years of data ending on the latest available date
# ---------------------------------------------------------------------------
def prepare_data() -> None:
    """
    Download RELIANCE.NS OHLCV from yfinance for the last 15 years
    (start = today minus 15 years, end = latest available trading day).
    Then engineer all technical features and save to FEAT_CSV.
    Always re-downloads to ensure the most current data is used.
    """
    from utils.data_loader import download_data
    from utils.feature_engineering import engineer_features
    import yfinance as yf

    # Determine start date: exactly 15 years before today
    today = datetime.today()
    start_date = (today - timedelta(days=15 * 365 + 4)).strftime("%Y-%m-%d")

    # Find the actual latest date available from yfinance (last trading day)
    print(f"[DataPipeline] Probing latest available date for {TICKER} …")
    probe = yf.download(TICKER, period="5d", interval="1d",
                        auto_adjust=False, progress=False)
    if probe.empty:
        raise RuntimeError(
            f"[DataPipeline] Cannot reach yfinance for {TICKER}. "
            "Check internet connection and ticker symbol."
        )

    if isinstance(probe.columns, pd.MultiIndex):
        probe.columns = probe.columns.get_level_values(0)
    probe.index.name = "date"
    probe = probe.reset_index()
    probe["date"] = pd.to_datetime(probe["date"])
    latest_date = probe["date"].max().strftime("%Y-%m-%d")
    print(f"[DataPipeline] Latest available date: {latest_date}")
    print(f"[DataPipeline] Download range: {start_date} → {latest_date}")

    # Download the full 15-year window; force=True so we always get fresh data
    raw_df = download_data(
        ticker=TICKER,
        start=start_date,
        end=latest_date,
        force=True,
    )
    print(f"[DataPipeline] Downloaded {len(raw_df):,} rows")

    # Build and save features
    engineer_features(raw_df, save_path=FEAT_CSV)
    print(f"[DataPipeline] ✅ Features saved → {FEAT_CSV}")


# ---------------------------------------------------------------------------
# Sentiment cache helpers
# ---------------------------------------------------------------------------
def _load_sentiment_cache() -> dict | None:
    if not os.path.exists(SENTIMENT_CACHE_PATH):
        return None
    try:
        with open(SENTIMENT_CACHE_PATH) as f:
            cached = json.load(f)
        saved_at = datetime.fromisoformat(
            cached.get("_cached_at", "2000-01-01T00:00:00+00:00")
        )
        age = (datetime.now(timezone.utc) - saved_at).total_seconds()
        if age < SENTIMENT_CACHE_TTL:
            print(f"[Main] News cache hit ({age/3600:.1f}h old)")
            return {k: v for k, v in cached.items() if not k.startswith("_")}
        print(f"[Main] News cache stale ({age/3600:.1f}h old) — re-scraping.")
        return None
    except Exception as e:
        print(f"[Main] News cache unreadable ({e}) — re-scraping.")
        return None


def _save_sentiment_cache(news_output: dict):
    try:
        to_save = dict(news_output)
        to_save["_cached_at"] = datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.dirname(SENTIMENT_CACHE_PATH), exist_ok=True)
        with open(SENTIMENT_CACHE_PATH, "w") as f:
            json.dump(to_save, f, indent=2)
        print(f"[Main] News sentiment cached → {SENTIMENT_CACHE_PATH}")
    except Exception as e:
        print(f"[Main] Warning: could not cache news output ({e})")


# ---------------------------------------------------------------------------
# News agent
# ---------------------------------------------------------------------------
def _get_news() -> dict:
    cached = _load_sentiment_cache()
    if cached is not None:
        return cached

    print("[Main] Running News Agent …")
    try:
        from agents.news_agent import NewsAgent
        news_output = NewsAgent().run()
        _save_sentiment_cache(news_output)
        return news_output
    except ImportError:
        print("[Main] ⚠️  News agent not found — using neutral fallback.")
    except Exception as e:
        print(f"[Main] ⚠️  News agent error ({e}) — using neutral fallback.")

    return {
        "sentiment_label":  "Neutral",
        "sentiment_score":  0.0,
        "headline_summary": "No news data available.",
        "num_articles":     0,
    }


# ---------------------------------------------------------------------------
# Run iTransformer — load saved model, infer next 5 days
# ---------------------------------------------------------------------------
def _run_itransformer() -> dict:
    from agents.itransformer_agent import iTransformerAgent
    print("\n[Main] Running iTransformer Agent (inference only) …")
    agent = iTransformerAgent()
    # infer() loads the saved checkpoint and predicts without training
    return agent.infer(force_download=False)


# ---------------------------------------------------------------------------
# Run TFT — load saved model, infer next 5 days
# ---------------------------------------------------------------------------
def _run_tft() -> dict:
    print("\n[Main] Running TFT Agent (inference only) …")
    try:
        from agents.tft_agent import TFTAgent
        agent = TFTAgent()
        # retrain=False → loads saved checkpoint
        return agent.run(force_download=False, retrain=False)
    except Exception as e:
        print(f"[Main] ⚠️  TFT agent error ({e}) — using fallback.")
        import traceback; traceback.print_exc()
        # Return neutral fallback so orchestrator can still run
        return {
            "model":             "TFT",
            "predicted_prices":  [],
            "trend":             "Unknown",
            "confidence":        0.0,
            "important_features": [],
            "last_actual_price": 0.0,
            "val_mae":           0.0,
            "val_rmse":          0.0,
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(log_runs: bool = True):
    print("\n" + "█" * 64)
    print("  STOCK MARKET LLM — PREDICT (RELIANCE.NS)")
    print("  Loading saved models · Next 5 trading days")
    print("█" * 64)

    # ── STEP 1: Download fresh 15-year data and engineer features ─────────
    print("\n[Main] STEP 1 — Downloading fresh data & engineering features")
    prepare_data()

    # ── STEP 2: iTransformer inference ────────────────────────────────────
    print("\n[Main] STEP 2 — iTransformer inference")
    itx_output = _run_itransformer()

    # ── STEP 3: TFT inference ─────────────────────────────────────────────
    print("\n[Main] STEP 3 — TFT inference")
    tft_output = _run_tft()

    # ── STEP 4: News sentiment ────────────────────────────────────────────
    print("\n[Main] STEP 4 — News sentiment")
    news_output = _get_news()

    # ── STEP 5: Orchestrate final prediction ──────────────────────────────
    print("\n[Main] STEP 5 — Orchestrating final prediction")
    orchestrator = OrchestratorAgent(
        ticker="RELIANCE.NS",
        company="Reliance Industries Limited",
        log_runs=log_runs,
    )
    final = orchestrator.run(itx_output, tft_output, news_output)
    OrchestratorAgent.save(final)

    print("\n" + "█" * 64)
    print("  PREDICTION COMPLETE")
    print("█" * 64)
    return final


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stock Market LLM — Predict Only")
    p.add_argument("--no-log", action="store_true", help="Disable MCP run logging")
    args = p.parse_args()
    main(log_runs=not args.no_log)
