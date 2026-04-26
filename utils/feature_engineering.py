"""
utils/feature_engineering.py
==============================
Unified feature engineering for both iTransformer and TFT pipelines.

What's computed
---------------
Technical indicators  (past features — unknown in future):
    rsi, macd, macd_signal, macd_hist,
    sma_20, ema_20,
    bb_upper, bb_middle, bb_lower, atr,
    returns, volatility

Time / calendar features  (known future inputs — TFT decoder only):
    day_of_week, day_of_month, month, week_of_year

Output CSV columns (matches ALL_FEATURES in settings.py):
    date,
    open, high, low, close, adj_close, volume,   ← raw OHLCV
    rsi, macd, macd_signal, macd_hist,            ← momentum
    sma_20, ema_20,                               ← trend
    bb_upper, bb_middle, bb_lower, atr,           ← volatility
    returns, volatility,                          ← risk
    day_of_week, day_of_month, month, week_of_year ← calendar

Indicator library fallback chain (in priority order)
----------------------------------------------------
  1. ta          (pip install ta)           ← preferred, used if available
  2. pandas-ta   (pip install pandas-ta)   ← second choice
  3. pure pandas/numpy                     ← always works, no extra install
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import FEAT_CSV, FEATURE_COLS

warnings.filterwarnings("ignore")

# ── Detect available TA libraries once at import time ─────────────────────
try:
    from ta.momentum   import RSIIndicator
    from ta.trend      import MACD, SMAIndicator, EMAIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    _USE_TA = True
except ImportError:
    _USE_TA = False

try:
    import pandas_ta as pta
    _USE_PTA = True
except ImportError:
    _USE_PTA = False


# ============================================================
# Individual indicator functions  (TFT codebase style)
# Each function takes a DataFrame, adds column(s), returns df.
# ============================================================

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    c = df["close"]
    if _USE_TA:
        df["rsi"] = RSIIndicator(close=c, window=window).rsi()
    elif _USE_PTA:
        df["rsi"] = df.ta.rsi(length=window)
    else:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(window).mean()
        loss  = (-delta.clip(upper=0)).rolling(window).mean()
        rs    = gain / (loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame,
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    c = df["close"]
    if _USE_TA:
        m = MACD(close=c, window_fast=fast, window_slow=slow, window_sign=signal)
        df["macd"]        = m.macd()
        df["macd_signal"] = m.macd_signal()
        df["macd_hist"]   = m.macd_diff()
    elif _USE_PTA:
        m = df.ta.macd(fast=fast, slow=slow, signal=signal)
        df["macd"]        = m[f"MACD_{fast}_{slow}_{signal}"]
        df["macd_signal"] = m[f"MACDs_{fast}_{slow}_{signal}"]
        df["macd_hist"]   = m[f"MACDh_{fast}_{slow}_{signal}"]
    else:
        ema_fast          = c.ewm(span=fast,   adjust=False).mean()
        ema_slow          = c.ewm(span=slow,   adjust=False).mean()
        df["macd"]        = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]
    return df


def add_moving_averages(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    c = df["close"]
    if _USE_TA:
        df["sma_20"] = SMAIndicator(close=c, window=window).sma_indicator()
        df["ema_20"] = EMAIndicator(close=c, window=window).ema_indicator()
    else:
        df["sma_20"] = c.rolling(window).mean()
        df["ema_20"] = c.ewm(span=window, adjust=False).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, std: int = 2) -> pd.DataFrame:
    c = df["close"]
    if _USE_TA:
        bb = BollingerBands(close=c, window=window, window_dev=std)
        df["bb_upper"]  = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"]  = bb.bollinger_lband()
    elif _USE_PTA:
        b = df.ta.bbands(length=window, std=std)
        df["bb_upper"]  = b[f"BBU_{window}_{float(std):.1f}"]
        df["bb_middle"] = b[f"BBM_{window}_{float(std):.1f}"]
        df["bb_lower"]  = b[f"BBL_{window}_{float(std):.1f}"]
    else:
        df["bb_middle"] = c.rolling(window).mean()
        _std            = c.rolling(window).std()
        df["bb_upper"]  = df["bb_middle"] + std * _std
        df["bb_lower"]  = df["bb_middle"] - std * _std
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    if _USE_TA:
        df["atr"] = AverageTrueRange(
            high=h, low=l, close=c, window=window
        ).average_true_range()
    elif _USE_PTA:
        df["atr"] = df.ta.atr(length=window)
    else:
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window).mean()
    return df


def add_returns_and_volatility(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    df["returns"]    = df["close"].pct_change()
    log_ret          = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = log_ret.rolling(vol_window).std()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features required by the TFT decoder.
    These are KNOWN in advance (future dates are predictable).
    """
    dt = pd.to_datetime(df["date"])
    df["day_of_week"]  = dt.dt.dayofweek                          # 0=Mon … 6=Sun
    df["day_of_month"] = dt.dt.day                                # 1–31
    df["month"]        = dt.dt.month                              # 1–12
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)     # 1–52
    return df


# ============================================================
# Master functions
# ============================================================

def build_features(
    df:        pd.DataFrame,
    save_path: str = FEAT_CSV,
) -> pd.DataFrame:
    """
    iTransformer entry point.
    Computes all technical indicators, selects FEATURE_COLS, saves CSV.

    Parameters
    ----------
    df        : raw OHLCV DataFrame  (date, open, high, low, close, adj_close, volume)
    save_path : where to write the processed CSV  (None = don't save)

    Returns
    -------
    pd.DataFrame  with columns  [date] + FEATURE_COLS, NaN rows dropped
    """
    return _run_pipeline(df, save_path=save_path, add_time=False,
                         col_filter=FEATURE_COLS, tag="FeatureEng/iTx")


def engineer_features(
    raw:       pd.DataFrame,
    save_path: str = FEAT_CSV,
) -> pd.DataFrame:
    """
    TFT entry point  (matches original TFT function name exactly).
    Computes all technical indicators AND time/calendar features, saves CSV.

    Parameters
    ----------
    raw       : raw OHLCV DataFrame  (date, open, high, low, close, adj_close, volume)
    save_path : where to write the processed CSV

    Returns
    -------
    pd.DataFrame  with ALL_FEATURES columns + date, NaN rows dropped
    """
    from config.settings import ALL_FEATURES
    return _run_pipeline(raw, save_path=save_path, add_time=True,
                         col_filter=None, tag="FeatEng/TFT")


def _run_pipeline(
    df:         pd.DataFrame,
    save_path:  str,
    add_time:   bool,
    col_filter: list,
    tag:        str,
) -> pd.DataFrame:
    """
    Shared implementation used by both entry points.

    Parameters
    ----------
    add_time   : if True, add day_of_week / month / week_of_year etc.
    col_filter : if not None, keep only these columns in final output
    """
    print(f"[{tag}] Computing technical indicators …")
    df = df.copy().sort_values("date").reset_index(drop=True)
    df.columns = df.columns.str.lower()

    # ── Ensure adj_close exists ────────────────────────────────────────────
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # ── Apply all indicators ───────────────────────────────────────────────
    df = add_rsi(df)
    df = add_macd(df)
    df = add_moving_averages(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_returns_and_volatility(df)

    if add_time:
        df = add_time_features(df)

    # ── Drop NaN warm-up rows ──────────────────────────────────────────────
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"[{tag}] Dropped {before - len(df)} warm-up rows  "
          f"→ {len(df):,} rows remain")

    # ── Select and reorder columns ─────────────────────────────────────────
    if col_filter is not None:
        available = [c for c in col_filter if c in df.columns]
        df = df[["date"] + available]
    # else: keep all computed columns (TFT path)

    df = df.reset_index(drop=True)

    # ── Save ───────────────────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[{tag}] Saved {len(df):,} rows → {save_path}")

    print(f"[{tag}] Final shape: {df.shape}  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ============================================================
if __name__ == "__main__":
    from utils.data_loader import download_data
    raw  = download_data()

    print("\n── iTransformer features ──")
    itx_df = build_features(raw)
    print("Columns:", itx_df.columns.tolist())

    print("\n── TFT features ──")
    tft_df = engineer_features(raw)
    print("Columns:", tft_df.columns.tolist())
