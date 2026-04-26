"""
utils/preprocessing.py
========================
Unified preprocessing for both models.

┌──────────────────────────────────────────────────────────────────┐
│  iTransformer  →  StockPreprocessor  (stateful class)            │
│                   .fit_transform()  .transform()                 │
│                   .inverse_target() .save()  .load()             │
│                                                                  │
│  TFT           →  standalone functions                           │
│                   split_dataframe()   fit_scaler()               │
│                   scale_dataframe()   inverse_transform_close()  │
│                   save_scaler()       load_scaler()              │
└──────────────────────────────────────────────────────────────────┘

Key rules (same in both approaches)
-------------------------------------
  - Chronological split only — NO shuffling
  - Scaler fitted on TRAINING data only
  - Same scaler applied to validation data
"""

import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    # iTransformer
    FEATURE_COLS, TARGET_COL, ITX_SCALER_PATH,
    # TFT
    ALL_FEATURES, CLOSE_IDX,
    # shared
    TRAIN_RATIO, TFT_SCALER_PATH,
    # legacy aliases (so old imports don't break)
    SCALER_PATH, SCALER_SAVE_PATH,
)


# ============================================================
# iTransformer — Stateful Preprocessor Class
# ============================================================
class StockPreprocessor:
    """
    Fit-transform wrapper for the iTransformer pipeline.

    The scaler is owned by this object, so the agent only needs
    to keep one reference to call transform / inverse_target later.

    Usage
    -----
        prep = StockPreprocessor()
        train_arr, val_arr, split_idx = prep.fit_transform(df)
        scaled     = prep.transform(df_new)
        real_prices = prep.inverse_target(scaled_predictions)
        prep.save()

        # Later / inference only:
        prep = StockPreprocessor.load()
    """

    def __init__(
        self,
        feature_cols: list  = None,
        target_col:   str   = TARGET_COL,
        train_ratio:  float = TRAIN_RATIO,
    ):
        self.feature_cols = feature_cols or FEATURE_COLS
        self.target_col   = target_col
        self.train_ratio  = train_ratio
        self.scaler       = MinMaxScaler(feature_range=(0, 1))
        self._is_fitted   = False
        self.target_idx   = self.feature_cols.index(target_col)

    # ── fit + split ────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame):
        """
        Chronological split → fit scaler on train → scale both splits.

        Returns
        -------
        train_arr : np.ndarray  [n_train, n_features]
        val_arr   : np.ndarray  [n_val,   n_features]
        split_idx : int
        """
        values    = df[self.feature_cols].values.astype(np.float32)
        n_total   = len(values)
        split_idx = int(n_total * self.train_ratio)

        train_raw = values[:split_idx]
        val_raw   = values[split_idx:]

        self.scaler.fit(train_raw)       # fit on train only
        self._is_fitted = True

        train_arr = self.scaler.transform(train_raw).astype(np.float32)
        val_arr   = self.scaler.transform(val_raw).astype(np.float32)

        print(f"[StockPreprocessor] Train: {len(train_arr):,}  "
              f"Val: {len(val_arr):,}  "
              f"Split index: {split_idx}")
        return train_arr, val_arr, split_idx

    def fit(self, df: pd.DataFrame):
        values = df[self.feature_cols].values.astype(np.float32)
        self.scaler.fit(values)
        self._is_fitted = True

    # ── transform only ────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Scale a DataFrame using the already-fitted scaler."""
        self._check_fitted()
        values = df[self.feature_cols].values.astype(np.float32)
        return self.scaler.transform(values).astype(np.float32)

    # ── inverse transform (target column only) ────────────────────────────
    def inverse_target(self, scaled_values: np.ndarray) -> np.ndarray:
        """
        Inverse-scale predictions back to real price units.

        Parameters
        ----------
        scaled_values : np.ndarray  shape [n] or [n, pred_len]

        Returns
        -------
        np.ndarray  same shape, in original price units
        """
        self._check_fitted()
        flat  = np.array(scaled_values).flatten()
        n_f   = len(self.feature_cols)

        dummy = np.zeros((len(flat), n_f), dtype=np.float32)
        dummy[:, self.target_idx] = flat

        inverted = self.scaler.inverse_transform(dummy)[:, self.target_idx]
        return inverted.reshape(np.array(scaled_values).shape)

    # ── persist / load ────────────────────────────────────────────────────
    def save(self, path: str = None):
        self._check_fitted()
        path = path or ITX_SCALER_PATH
        joblib.dump(self.scaler, path)
        print(f"[StockPreprocessor] Scaler saved → {path}")

    @classmethod
    def load(
        cls,
        path:         str  = None,
        feature_cols: list = None,
        target_col:   str  = TARGET_COL,
    ) -> "StockPreprocessor":
        path = path or ITX_SCALER_PATH
        obj             = cls(feature_cols=feature_cols, target_col=target_col)
        obj.scaler      = joblib.load(path)
        obj._is_fitted  = True
        print(f"[StockPreprocessor] Scaler loaded ← {path}")
        return obj

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "[StockPreprocessor] Not fitted yet — call fit_transform() first."
            )


# ============================================================
# TFT — Standalone Preprocessing Functions
# ============================================================

def split_dataframe(
    df:          pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
):
    """
    Chronological (no-shuffle) train/val split that returns DataFrames.
    Used by the TFT pipeline which works with DataFrames throughout.

    Returns
    -------
    train_df  : pd.DataFrame
    val_df    : pd.DataFrame
    split_idx : int
    """
    n         = len(df)
    split_idx = int(n * train_ratio)
    train_df  = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df    = df.iloc[split_idx:].copy().reset_index(drop=True)
    print(f"[TFTPreprocess] Train: {len(train_df):,} rows  "
          f"Val: {len(val_df):,} rows")
    return train_df, val_df, split_idx


def fit_scaler(
    train_df: pd.DataFrame,
    features: list = ALL_FEATURES,
) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on training data only.
    Returns the fitted scaler (caller is responsible for storing it).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[features].values)
    return scaler


def scale_dataframe(
    df:       pd.DataFrame,
    scaler:   MinMaxScaler,
    features: list = ALL_FEATURES,
) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to a DataFrame.
    Non-feature columns (e.g. 'date') are preserved unchanged.

    Returns a new scaled DataFrame (original is not modified).
    """
    scaled           = df.copy()
    scaled[features] = scaler.transform(df[features].values)
    return scaled


def inverse_transform_close(
    predictions:  np.ndarray,
    scaler:       MinMaxScaler,
    close_idx:    int   = CLOSE_IDX,
    num_features: int   = None,
) -> np.ndarray:
    """
    Inverse-transform TFT predicted close prices to real price units.

    Parameters
    ----------
    predictions  : np.ndarray  shape (N, horizon) or (horizon,)
    scaler       : fitted MinMaxScaler from fit_scaler()
    close_idx    : column index of 'close' inside ALL_FEATURES  (default CLOSE_IDX=3)
    num_features : total number of feature columns              (default len(ALL_FEATURES))

    Returns
    -------
    np.ndarray  same shape as predictions, in original price units
    """
    num_features = num_features or len(ALL_FEATURES)
    flat  = predictions.reshape(-1)
    dummy = np.zeros((len(flat), num_features), dtype=np.float32)
    dummy[:, close_idx] = flat
    inv   = scaler.inverse_transform(dummy)[:, close_idx]
    return inv.reshape(predictions.shape)


def save_scaler(scaler: MinMaxScaler, path: str = None):
    """Save TFT scaler using pickle (matches TFT codebase convention)."""
    path = path or TFT_SCALER_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[TFTPreprocess] Scaler saved → {path}")


def load_scaler(path: str = None) -> MinMaxScaler:
    """Load TFT scaler from pickle file."""
    path = path or TFT_SCALER_PATH
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[TFTPreprocess] Scaler loaded ← {path}")
    return scaler
