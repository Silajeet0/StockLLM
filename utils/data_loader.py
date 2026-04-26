"""
utils/data_loader.py
====================
Unified Yahoo Finance data downloader for both iTransformer and TFT pipelines.

Two download modes (both produce identical output):
  • start/end  dates  — your iTransformer style  e.g. "2010-01-01" → today
  • period/interval   — TFT style                e.g. "15y" / "1d"

Both modes hit the same cache — if the CSV already exists the download
is skipped unless force=True.

Output columns: date, open, high, low, close, adj_close, volume
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    TICKER,
    START_DATE, END_DATE,   # iTransformer style
    PERIOD, INTERVAL,       # TFT style
    RAW_CSV,                # canonical save path  (RAW_DATA_PATH is an alias)
)

# Columns we keep from Yahoo Finance
_KEEP_COLS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


# ---------------------------------------------------------------------------
def download_data(
    ticker:     str        = TICKER,
    start:      str        = START_DATE,
    end:        str | None = END_DATE,
    period:     str | None = None,
    interval:   str        = INTERVAL,
    save_path:  str        = RAW_CSV,
    force:      bool       = False,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance and cache it as a CSV.

    Priority: if `period` is provided it overrides start/end (TFT style).
              Otherwise start/end are used (iTransformer style).

    Parameters
    ----------
    ticker    : Yahoo Finance ticker symbol  (e.g. "RELIANCE.NS")
    start     : start date "YYYY-MM-DD"  — used when period is None
    end       : end date "YYYY-MM-DD" or None → today
    period    : yfinance period string e.g. "15y", "5y", "max"
                if set, overrides start/end
    interval  : bar size — "1d" for daily (default)
    save_path : path to write/read the cached CSV
    force     : if True, re-download even if the cached CSV exists

    Returns
    -------
    pd.DataFrame  columns: date, open, high, low, close, adj_close, volume
    """

    # ── Cache check ────────────────────────────────────────────────────────
    if os.path.exists(save_path) and not force:
        print(f"[DataLoader] Cache hit: {save_path}")
        return load_raw(save_path)

    # ── Determine download parameters ──────────────────────────────────────
    if period is not None:
        # TFT style — period overrides start/end
        print(f"[DataLoader] Downloading {ticker}  period={period}  interval={interval} …")
        raw = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=False, progress=True)
    else:
        # iTransformer style — explicit date range
        end_str = end or datetime.today().strftime("%Y-%m-%d")
        print(f"[DataLoader] Downloading {ticker}  {start} → {end_str} …")
        raw = yf.download(ticker, start=start, end=end_str,
                          auto_adjust=False, progress=True)

    if raw.empty:
        raise ValueError(
            f"[DataLoader] No data returned for ticker '{ticker}'. "
            "Check the symbol and your internet connection."
        )

    # ── Normalise columns ──────────────────────────────────────────────────
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    # Reset index so 'date' becomes a regular column
    raw.index.name = "date"
    raw = raw.reset_index()
    raw = raw.rename(columns={"adj_close": "adj_close",   # already correct
                               "adj close": "adj_close"})  # old yfinance format
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    # Keep only the columns we need
    keep = [c for c in _KEEP_COLS if c in raw.columns]
    raw  = raw[keep].dropna().reset_index(drop=True)

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    raw.to_csv(save_path, index=False)
    print(f"[DataLoader] Saved {len(raw):,} rows → {save_path}")
    return raw


# ---------------------------------------------------------------------------
def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    """
    Load the cached raw CSV from disk (no network call).
    Used by TFT pipeline and for quick re-runs without re-downloading.

    Returns
    -------
    pd.DataFrame  sorted by date, columns: date, open, high, low, close, adj_close, volume
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[DataLoader] Loaded {len(df):,} rows  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # iTransformer style (start/end)
    df1 = download_data(force=True)
    print(df1.tail())

    # TFT style (period)
    df2 = download_data(period=PERIOD, force=False)   # will hit cache
    print(df2.tail())
