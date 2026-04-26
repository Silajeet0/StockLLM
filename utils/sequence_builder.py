"""
utils/sequence_builder.py
==========================
Unified sliding-window dataset builder for both models.

┌─────────────────────────────────────────────────────────────────┐
│  iTransformer  →  StockSequenceDataset   (numpy array → x, y)  │
│  TFT           →  TFTDataset             (DataFrame  → x_enc,  │
│                                            x_dec, y)            │
│  Shared        →  build_itx_dataloaders  / build_tft_dataloaders│
└─────────────────────────────────────────────────────────────────┘

iTransformer window layout
--------------------------
    x  [seq_len, n_vars]  ← scaled_arr[i : i+seq_len]
    y  [pred_len]         ← scaled_arr[i+seq_len : i+seq_len+pred_len, target_idx]

TFT window layout
-----------------
    x_enc  [lookback, 21]  ← all_features[i : i+lookback]   (past, unknown future)
    x_dec  [horizon,   4]  ← time_features[i+lookback : i+lookback+horizon]
    y      [horizon]       ← close[i+lookback : i+lookback+horizon]
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    # shared
    SEQ_LEN, PRED_LEN, TRAIN_CFG,
    # iTransformer
    TARGET_IDX,
    # TFT
    ALL_FEATURES, FUTURE_FEATURES, CLOSE_IDX,
    LOOKBACK, HORIZON, BATCH_SIZE,
)


# ============================================================
# iTransformer Dataset
# ============================================================
class StockSequenceDataset(Dataset):
    """
    Sliding-window dataset for the iTransformer.
    Accepts a pre-scaled numpy array.

    Parameters
    ----------
    data       : np.ndarray  shape [T, n_features]  — already MinMax-scaled
    seq_len    : encoder look-back window (default: SEQ_LEN = 60)
    pred_len   : forecast horizon         (default: PRED_LEN = 5)
    target_idx : column index of the target variable (default: TARGET_IDX = 0)

    Returns per sample
    ------------------
    x : torch.Tensor  [seq_len, n_features]  — full multivariate input window
    y : torch.Tensor  [pred_len]             — future target (close) values
    """

    def __init__(
        self,
        data:       np.ndarray,
        seq_len:    int = SEQ_LEN,
        pred_len:   int = PRED_LEN,
        target_idx: int = TARGET_IDX,
    ):
        super().__init__()
        self.data       = data.astype(np.float32)
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.target_idx = target_idx
        self.n_samples  = len(data) - seq_len - pred_len + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"[StockSequenceDataset] Not enough rows: "
                f"{len(data)} < seq_len({seq_len}) + pred_len({pred_len})"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[
            idx + self.seq_len : idx + self.seq_len + self.pred_len,
            self.target_idx,
        ]
        
        # Create dummy time markers [Seq, 4] that thuml expects
        # (4 matches the 'h' frequency in your iTransformerConfig)
        x_mark = np.zeros((self.seq_len, 4), dtype=np.float32)
        y_mark = np.zeros((self.pred_len, 4), dtype=np.float32)

        return (
            torch.from_numpy(x), 
            torch.from_numpy(y), 
            torch.from_numpy(x_mark), 
            torch.from_numpy(y_mark)
        )


# ============================================================
# TFT Dataset
# ============================================================
class TFTDataset(Dataset):
    """
    Sliding-window dataset for the Temporal Fusion Transformer.
    Accepts a scaled pandas DataFrame that already contains all
    feature columns + time/calendar columns.

    Parameters
    ----------
    df       : pd.DataFrame  shape [T, n_cols]
               Must contain ALL_FEATURES (21) and FUTURE_FEATURES (4) columns.
    lookback : encoder window length (default: LOOKBACK = SEQ_LEN = 60)
    horizon  : forecast horizon      (default: HORIZON  = PRED_LEN = 5)

    Returns per sample
    ------------------
    x_enc : torch.Tensor  [lookback, 21]  — full feature window (encoder input)
    x_dec : torch.Tensor  [horizon,   4]  — known future time features (decoder)
    y     : torch.Tensor  [horizon]       — future close prices (scaled)
    """

    def __init__(
        self,
        df:       pd.DataFrame,
        lookback: int = LOOKBACK,
        horizon:  int = HORIZON,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon  = horizon

        # Extract numpy arrays once for speed
        self.enc_data  = df[ALL_FEATURES].values.astype(np.float32)     # (T, 21)
        self.dec_data  = df[FUTURE_FEATURES].values.astype(np.float32)  # (T, 4)
        self.close_col = self.enc_data[:, CLOSE_IDX]                    # (T,)

        self.n_samples = len(df) - lookback - horizon + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"[TFTDataset] DataFrame too short ({len(df)} rows) for "
                f"lookback={lookback} + horizon={horizon}."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        enc_end = idx + self.lookback
        dec_end = enc_end + self.horizon

        x_enc = self.enc_data[idx    : enc_end]   # (lookback, 21)
        x_dec = self.dec_data[enc_end : dec_end]  # (horizon, 4)
        y     = self.close_col[enc_end : dec_end] # (horizon,)

        return (
            torch.tensor(x_enc, dtype=torch.float32),
            torch.tensor(x_dec, dtype=torch.float32),
            torch.tensor(y,     dtype=torch.float32),
        )


# ============================================================
# DataLoader builders
# ============================================================
def build_itx_dataloaders(
    train_arr:   np.ndarray,
    val_arr:     np.ndarray,
    seq_len:     int  = SEQ_LEN,
    pred_len:    int  = PRED_LEN,
    target_idx:  int  = TARGET_IDX,
    batch_size:  int  = None,
    num_workers: int  = None,
    pin_memory:  bool = None,
):
    """
    Build iTransformer train/val DataLoaders from pre-scaled numpy arrays.

    Parameters
    ----------
    train_arr  : np.ndarray  [n_train, n_features]  — scaled training data
    val_arr    : np.ndarray  [n_val,   n_features]  — scaled validation data

    Returns
    -------
    train_loader, val_loader
    """
    bs = batch_size  or TRAIN_CFG["batch_size"]
    nw = num_workers if num_workers is not None else TRAIN_CFG["num_workers"]
    pm = pin_memory  if pin_memory  is not None else TRAIN_CFG["pin_memory"]

    train_ds = StockSequenceDataset(train_arr, seq_len, pred_len, target_idx)
    val_ds   = StockSequenceDataset(val_arr,   seq_len, pred_len, target_idx)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pm, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pm, drop_last=False,
    )

    print(f"[SeqBuilder/iTx] Train batches: {len(train_loader):,}  "
          f"Val batches: {len(val_loader):,}  "
          f"(batch_size={bs})")
    return train_loader, val_loader


def build_tft_dataloaders(
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    lookback:    int = LOOKBACK,
    horizon:     int = HORIZON,
    batch_size:  int = BATCH_SIZE,
    num_workers: int = 0,
):
    """
    Build TFT train/val DataLoaders from scaled DataFrames.

    Parameters
    ----------
    train_df : pd.DataFrame  — scaled training DataFrame (must have ALL_FEATURES cols)
    val_df   : pd.DataFrame  — scaled validation DataFrame

    Returns
    -------
    train_loader, val_loader, train_dataset, val_dataset
    """
    train_ds = TFTDataset(train_df, lookback, horizon)
    val_ds   = TFTDataset(val_df,   lookback, horizon)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[SeqBuilder/TFT] Train samples: {len(train_ds):,}  "
          f"Val samples: {len(val_ds):,}  "
          f"(batch_size={batch_size})")
    return train_loader, val_loader, train_ds, val_ds


# ============================================================
# Backward-compatibility alias
# ============================================================
# The iTransformer agent calls build_dataloaders() — keep that name working.
build_dataloaders = build_itx_dataloaders


# ============================================================
# Sanity check
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("iTransformer dataset")
    print("=" * 55)
    np.random.seed(0)
    dummy_arr = np.random.rand(3700, 18).astype(np.float32)

    ds = StockSequenceDataset(dummy_arr, seq_len=60, pred_len=5, target_idx=0)
    x, y = ds[0]
    print(f"  x : {tuple(x.shape)}   y : {tuple(y.shape)}")
    print(f"  Total samples : {len(ds):,}")

    tl, vl = build_itx_dataloaders(dummy_arr[:2960], dummy_arr[2960:])
    xb, yb = next(iter(tl))
    print(f"  Batch x : {tuple(xb.shape)}   Batch y : {tuple(yb.shape)}")

    print()
    print("=" * 55)
    print("TFT dataset")
    print("=" * 55)

    # Build a dummy DataFrame with all required columns
    from config.settings import ALL_FEATURES, FUTURE_FEATURES
    T       = 3700
    cols    = list(dict.fromkeys(ALL_FEATURES + FUTURE_FEATURES))  # deduplicated
    df_dummy = pd.DataFrame(np.random.rand(T, len(cols)), columns=cols)

    tft_ds = TFTDataset(df_dummy, lookback=60, horizon=5)
    x_enc, x_dec, y_tft = tft_ds[0]
    print(f"  x_enc : {tuple(x_enc.shape)}")
    print(f"  x_dec : {tuple(x_dec.shape)}")
    print(f"  y     : {tuple(y_tft.shape)}")
    print(f"  Total samples : {len(tft_ds):,}")

    train_df = df_dummy.iloc[:2960].reset_index(drop=True)
    val_df   = df_dummy.iloc[2960:].reset_index(drop=True)
    tl2, vl2, _, _ = build_tft_dataloaders(train_df, val_df)
    enc_b, dec_b, y_b = next(iter(tl2))
    print(f"  Batch x_enc : {tuple(enc_b.shape)}   "
          f"x_dec : {tuple(dec_b.shape)}   y : {tuple(y_b.shape)}")

    print()
    print("✅  Both datasets verified")
