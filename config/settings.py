"""
config/settings.py
==================
Unified central configuration for the entire Stock Market LLM pipeline.

TICKER GENERALIZATION
---------------------
The active ticker is resolved in this order:
  1. STOCK_TICKER environment variable   (set by main.py before any import)
  2. Fallback default: "RELIANCE.NS"

To analyze a different stock:
  • CLI:        python main.py --ticker TCS.NS --company "Tata Consultancy Services"
  • Shell env:  export STOCK_TICKER=HDFCBANK.NS && python main.py

All file paths (raw CSV, feature CSV, model checkpoints, scalers) are
automatically derived from the ticker slug so runs never collide.

Edit ONLY this file to change any behaviour across the whole system.
"""

import os

# ============================================================
# TICKER CONFIGURATION
# ============================================================
# Fixed to Reliance Industries Ltd (NSE)
TICKER = "RELIANCE.NS"

# Slug used in file names:  "RELIANCE.NS" → "reliance_ns"
#                           "HDFCBANK.NS" → "hdfcbank_ns"
#                           "TCS.NS"      → "tcs_ns"
_SLUG  = TICKER.replace(".", "_").replace("-", "_").lower()

# ============================================================
# PATHS
# ============================================================
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")

RAW_DIR   = os.path.join(DATA_DIR, "raw")
PROC_DIR  = os.path.join(DATA_DIR, "processed")

# TFT codebase aliases (kept for backward-compat — same folders)
YAHOO_RAW_DIR  = RAW_DIR
YAHOO_PROC_DIR = PROC_DIR

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_DIR   = os.path.join(MODEL_DIR, "mcp_logs")

for _d in (RAW_DIR, PROC_DIR, MODEL_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ── File paths (all ticker-specific) ─────────────────────────────────────
RAW_CSV          = os.path.join(RAW_DIR,   f"{_SLUG}_raw.csv")
FEAT_CSV         = os.path.join(PROC_DIR,  f"{_SLUG}_features.csv")

# iTransformer artefacts
ITX_SCALER_PATH  = os.path.join(MODEL_DIR, f"{_SLUG}_itransformer_scaler.pkl")
ITX_MODEL_CKPT   = os.path.join(MODEL_DIR, f"{_SLUG}_itransformer_best.pt")
ITX_METRICS_PATH = os.path.join(MODEL_DIR, f"{_SLUG}_itransformer_metrics.json")

# TFT artefacts
TFT_SCALER_PATH  = os.path.join(MODEL_DIR, f"{_SLUG}_tft_scaler.pkl")
TFT_MODEL_CKPT   = os.path.join(MODEL_DIR, f"{_SLUG}_tft_best.pt")
TFT_METRICS_PATH = os.path.join(MODEL_DIR, f"{_SLUG}_tft_metrics.json")

# Final orchestrator output
FINAL_OUTPUT_PATH = os.path.join(MODEL_DIR, f"{_SLUG}_final_prediction.json")
SENTIMENT_CACHE_PATH = os.path.join(MODEL_DIR, f"{_SLUG}_sentiment_cache.json")

# Legacy flat aliases — keeps every unchanged import in either codebase working
SCALER_PATH         = ITX_SCALER_PATH
MODEL_CKPT          = ITX_MODEL_CKPT
METRICS_PATH        = ITX_METRICS_PATH
RAW_DATA_PATH       = RAW_CSV
PROCESSED_DATA_PATH = FEAT_CSV
MODEL_SAVE_PATH     = TFT_MODEL_CKPT
SCALER_SAVE_PATH    = TFT_SCALER_PATH

# ============================================================
# DATA SOURCE
# ============================================================
START_DATE  = "2010-01-01"    # ~15 years of daily data
END_DATE    = None             # None → today
PERIOD      = "15y"
INTERVAL    = "1d"

# ============================================================
# SEQUENCE PARAMETERS  (shared by both models)
# ============================================================
SEQ_LEN  = 60   # look-back window / encoder input length
PRED_LEN = 5    # forecast horizon in trading days

LOOKBACK = SEQ_LEN
HORIZON  = PRED_LEN

# ============================================================
# TRAIN / VAL SPLIT
# ============================================================
TRAIN_RATIO = 0.80

# ============================================================
# FEATURES
# ============================================================

PAST_FEATURES = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd", "macd_signal", "macd_hist",
    "sma_20", "ema_20",
    "bb_upper", "bb_middle", "bb_lower",
    "atr", "returns", "volatility",
]  # 17 features

FUTURE_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
]  # 4 features

# iTransformer: close placed FIRST so TARGET_IDX = 0
FEATURE_COLS = [
    "close",
    "open", "high", "low", "volume",
    "rsi", "macd", "macd_signal", "macd_hist",
    "sma_20", "ema_20",
    "bb_upper", "bb_middle", "bb_lower", "atr",
    "returns", "volatility",
    "adj_close",
]  # 18 features

ALL_FEATURES = PAST_FEATURES + FUTURE_FEATURES   # 21

TARGET_COL  = "close"
TARGET      = "close"
TARGET_IDX  = FEATURE_COLS.index(TARGET_COL)    # 0  (iTransformer)
CLOSE_IDX   = PAST_FEATURES.index("close")      # 3  (TFT)
N_VARS      = len(FEATURE_COLS)                  # 18

# ============================================================
# iTransformer HYPER-PARAMETERS
# ============================================================
ITRANSFORMER_CFG = dict(
    seq_len    = SEQ_LEN,
    pred_len   = PRED_LEN,
    n_vars     = N_VARS,
    d_model    = 128,
    n_heads    = 4,
    n_layers   = 3,
    d_ff       = 256,
    dropout    = 0.15,
    target_idx = TARGET_IDX,
)

# ============================================================
# TFT HYPER-PARAMETERS
# ============================================================
TFT_CFG = dict(
    hidden_size       = 64,
    lstm_layers       = 2,
    num_heads         = 4,
    dropout           = 0.1,
    n_past_features   = len(PAST_FEATURES),
    n_future_features = len(FUTURE_FEATURES),
    lookback          = SEQ_LEN,
    horizon           = PRED_LEN,
)

HIDDEN_SIZE = TFT_CFG["hidden_size"]
LSTM_LAYERS = TFT_CFG["lstm_layers"]
NUM_HEADS   = TFT_CFG["num_heads"]
DROPOUT     = TFT_CFG["dropout"]

# ============================================================
# TRAINING HYPER-PARAMETERS  (shared defaults)
# ============================================================
TRAIN_CFG = dict(
    epochs        = 50,
    batch_size    = 64,
    lr            = 1e-4,
    weight_decay  = 1e-4,
    patience      = 10,
    grad_clip     = 1.0,
    num_workers   = 0,
    pin_memory    = True,
)

BATCH_SIZE  = TRAIN_CFG["batch_size"]
MAX_EPOCHS  = TRAIN_CFG["epochs"]
LR          = TRAIN_CFG["lr"]
PATIENCE    = TRAIN_CFG["patience"]
