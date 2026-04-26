"""
agents/itransformer_agent.py
============================
iTransformer agent using the official THUML implementation
(github.com/thuml/iTransformer).

Three bugs fixed vs the previous version
-----------------------------------------
Bug 1 — NameError on import
    `torch.serialization.add_safe_globals([iTransformerConfig])` was called at
    module level BEFORE the `iTransformerConfig` class was defined.
    Fix: moved the call to after the class definition.

Bug 2 — TypeError: unexpected keyword argument 'output_attention'
    `_top_features` was calling
        model(..., output_attention=True)
    but THUML's Model.forward() does NOT accept that kwarg.
    `output_attention` is a CONFIG attribute read inside __init__ as
    `self.output_attention`; the forward pass checks `self.output_attention`
    internally to decide whether to return attention weights.
    Fix: temporarily set `model.output_attention = True` on the live object,
    call forward(), then restore the original value.

Bug 3 — ValueError: not enough values to unpack
    `StockSequenceDataset.__getitem__` returned (x, y) — 2 items — but every
    training/eval loop in this agent unpacked 4:
        for x, y, x_mark, y_mark in loader
    Fix: `StockSequenceDataset` now returns 4 items. See sequence_builder.py.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    FEATURE_COLS, TRAIN_CFG,
    ITX_MODEL_CKPT, ITX_SCALER_PATH, ITX_METRICS_PATH,
    SEQ_LEN, PRED_LEN, TARGET_IDX, FEAT_CSV,
)
from models.iTransformer import Model as iTransformer
from utils.data_loader import download_data
from utils.feature_engineering import build_features
from utils.preprocessing import StockPreprocessor
from utils.sequence_builder import StockSequenceDataset, build_itx_dataloaders as build_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# Config dataclass for THUML iTransformer
# ──────────────────────────────────────────────────────────────────────────────
class iTransformerConfig:
    """
    Mirrors the argparse Namespace that the THUML codebase normally builds.
    Every attribute below maps to a field the Model class reads in __init__.

    Architecture choices
    --------------------
    task_name = 'long_term_forecast'
        The THUML repo has several task heads. 'long_term_forecast' is the one
        the iTransformer paper benchmarks; it uses an inverted-attention
        encoder + a direct linear projection head.

    enc_in / dec_in / c_out = 18
        Number of variates (features). We use all 18 FEATURE_COLS.
        dec_in is required by the signature but unused by iTransformer
        (there is no autoregressive decoder); c_out controls the output
        projection shape.

    d_model = 128
        Hidden size. Paper uses 512 on ETTh1; 128 is appropriate for
        our 18-variate dataset and keeps GPU memory low.

    e_layers = 3
        Number of stacked iTransformer blocks.

    d_ff = 256
        Feed-forward expansion width inside each block.

    embed = 'timeF', freq = 'h'
        Controls the time-stamp embedding branch. We pass zero tensors for
        x_mark so the specific choice here does not affect the output, but
        the attributes must be present or the PatchEmbedding will crash.

    output_attention = False
        When True, Model.forward() returns (output, attention_weights).
        We flip this flag to True only inside _top_features() and restore
        it immediately after, so normal training/inference stays efficient.

    use_norm = True
        Instance normalisation applied at the start of forward(); important
        for non-stationary stock data.
    """
    def __init__(self, n_vars: int = len(FEATURE_COLS)):
        self.task_name       = "long_term_forecast"
        self.seq_len         = SEQ_LEN      # 60 — encoder lookback
        self.pred_len        = PRED_LEN     # 5  — forecast horizon
        self.label_len       = 0            # unused by iTransformer

        self.enc_in          = n_vars       # 18 variates in
        self.dec_in          = n_vars       # required by signature, not used
        self.c_out           = n_vars       # output shape: [B, pred_len, n_vars]

        self.d_model         = 128
        self.n_heads         = 8            # d_model / n_heads = 16 dims/head
        self.e_layers        = 3
        self.d_ff            = 256
        self.factor          = 1

        self.dropout         = 0.1
        self.embed           = "timeF"
        self.freq            = "h"

        self.activation      = "gelu"
        self.output_attention= False        # BUG 2 FIX: controlled via attribute
        self.use_norm        = True
        self.class_strategy  = "projection"


# ──────────────────────────────────────────────────────────────────────────────
# BUG 1 FIX: add_safe_globals must be called AFTER the class is defined.
# This allows torch.load(..., weights_only=True) to safely deserialise
# checkpoints that contain an iTransformerConfig object.
# ──────────────────────────────────────────────────────────────────────────────
torch.serialization.add_safe_globals([iTransformerConfig])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _confidence_from_mae(val_mae: float, price_range: float) -> float:
    """1 - 2*(MAE/price_range), clipped to [0.05, 0.99]."""
    if price_range < 1e-6:
        return 0.5
    return float(np.clip(1.0 - 2.0 * (val_mae / price_range), 0.05, 0.99))


def _top_features(model, x, top_k=5):
    """
    Extract top-k most influential features using attention weights.
    Uses iTransformer.get_var_attention_weights() which is purpose-built for this.
    """
    try:
        model.eval()
        with torch.no_grad():
            attn = model.get_var_attention_weights(x)  # [B, n_heads, N_vars, N_vars]
        # Average over batch and heads, sum incoming attention per variable
        imp = attn.mean(dim=(0, 1)).sum(dim=0).cpu().numpy()
        ranked = imp.argsort()[::-1][:top_k]
        return [FEATURE_COLS[i] for i in ranked if i < len(FEATURE_COLS)]
    except Exception as e:
        print(f"[iTransformerAgent] Attention unavailable ({e}) — fallback used.")
        return FEATURE_COLS[:top_k]

# ──────────────────────────────────────────────────────────────────────────────
# Training / evaluation loops
# NOTE: Both loops unpack 4 items: (x, y, x_mark, y_mark).
#       StockSequenceDataset now returns all 4 — see sequence_builder.py.
# ──────────────────────────────────────────────────────────────────────────────
def _train_epoch(
    model,
    loader,
    opt,
    crit,
    device,
    clip,
):
    model.train()
    total_loss = 0.0

    for x, y, x_mark, y_mark in loader:
        x       = x.to(device)
        y       = y.to(device)
        x_mark  = x_mark.to(device)
        y_mark  = y_mark.to(device)

        opt.zero_grad()

        outputs = model(x)  # iTransformer.forward only takes x
        preds = outputs[:, -model.pred_len:]  # shape [B, pred_len]

        loss = crit(preds, y)
        loss.backward()

        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)

        opt.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def _eval_epoch(
    model,
    loader,
    crit,
    device,
):
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0

    with torch.no_grad():
        for x, y, x_mark, y_mark in loader:
            x      = x.to(device)
            y      = y.to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)

            outputs = model(x)  # iTransformer.forward only takes x
            preds = outputs[:, -model.pred_len:]  # shape [B, pred_len]

            total_loss += crit(preds, y).item() * x.size(0)
            total_mae  += F.l1_loss(preds, y).item() * x.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_mae / n

# ──────────────────────────────────────────────────────────────────────────────
# Main agent
# ──────────────────────────────────────────────────────────────────────────────

def _make_model(cfg) -> "iTransformer":
    """Unpack iTransformerConfig into iTransformer keyword args."""
    if isinstance(cfg, dict):
        return iTransformer(**cfg)
    return iTransformer(
        seq_len    = getattr(cfg, "seq_len",    60),
        pred_len   = getattr(cfg, "pred_len",   5),
        n_vars     = getattr(cfg, "enc_in",     len(FEATURE_COLS)),
        d_model    = getattr(cfg, "d_model",    128),
        n_heads    = getattr(cfg, "n_heads",    8),
        n_layers   = getattr(cfg, "e_layers",   3),
        d_ff       = getattr(cfg, "d_ff",       256),
        dropout    = getattr(cfg, "dropout",    0.1),
        target_idx = getattr(cfg, "target_idx", TARGET_IDX),
    )

class iTransformerAgent:
    """
    Self-contained agent wrapping the THUML iTransformer.

    Entry-points
    ------------
    agent.run(force_download=False)   # train + predict
    agent.infer(force_download=False) # load saved checkpoint + predict
    """

    def __init__(self, model_cfg=None, train_cfg=None):
        self.model_cfg   = model_cfg or iTransformerConfig()
        self.train_cfg   = train_cfg or TRAIN_CFG.copy()
        self.device      = _get_device()
        print(f"[iTransformerAgent] Using device: {self.device}")
        self.model       = None
        self.prep        = None
        self.feat_df     = None
        self.train_arr   = None
        self.val_arr     = None
        self._price_range = 1.0
        self.target_idx = None
        self.feature_cols = None

    @property
    def seq_len(self):
        return self.model_cfg.seq_len

    @property
    def pred_len(self):
        return self.model_cfg.pred_len

    def _init_model(self, lr=None, d_model=None):
        import copy

        model_cfg = copy.deepcopy(self.model_cfg)
        train_cfg = copy.deepcopy(self.train_cfg)

        if d_model:
            model_cfg.d_model = d_model

        if lr:
            train_cfg["lr"] = lr

        model = _make_model(model_cfg).to(self.device)
        model.enc_in = model.n_vars   # compatibility alias

        return model, train_cfg
    
    def _train_model(self, model, data_scaled, train_cfg = None, fine_tune=False, return_val_loss=False):
        from torch.utils.data import DataLoader
        from utils.sequence_builder import StockSequenceDataset
        cfg = train_cfg if train_cfg else self.train_cfg
        # build dataset from raw array
        dataset = StockSequenceDataset(
            data_scaled,
            self.model_cfg.seq_len,
            self.model_cfg.pred_len,
            self.target_idx
        )

        loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=False
        )

        crit = nn.MSELoss()

        lr = cfg["lr"] * (0.1 if fine_tune else 1.0)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=cfg["weight_decay"]
        )

        model.train()

        total_loss = 0.0
        epochs = 1 if fine_tune else cfg["epochs"]

        for _ in range(epochs):
            loss = _train_epoch(
                model,
                loader,
                opt,
                crit,
                self.device,
                cfg["grad_clip"]
            )
            total_loss += loss

        avg_loss = total_loss / epochs

        if return_val_loss:
            return avg_loss

        return model
    
    def predict_step(self, model, data_slice):
        device = self.device
        seq_len = self.seq_len
        pred_len = self.pred_len

        x = torch.from_numpy(data_slice[-seq_len:]).unsqueeze(0).to(device).float()

        x_mark = torch.zeros(1, seq_len, 4, device=device)
        y_mark = torch.zeros(1, pred_len, 4, device=device)
        dec_inp = torch.zeros(1, pred_len, x.shape[-1], device=device)

        model.eval()
        with torch.no_grad():
            out = model(x)  # shape [B, pred_len]
            preds = out[:, -pred_len:].cpu().numpy()[0]
        
        

        return preds
    
    def set_hyperparams(self, cfg):
        for k, v in cfg.items():
            if hasattr(self.model_cfg, k):
                setattr(self.model_cfg, k, v)
            elif k in self.train_cfg:
                self.train_cfg[k] = v

    # ── Data loading ──────────────────────────────────────────────────────────
    def load_data(self, force_download: bool = False) -> "iTransformerAgent":
        import pandas as pd

        if os.path.exists(FEAT_CSV) and not force_download:
            print(f"[iTransformerAgent] Loading cached features from {FEAT_CSV}")
            self.feat_df = pd.read_csv(FEAT_CSV, parse_dates=["date"], index_col="date")
        else:
            self.feat_df = build_features(download_data(force=force_download))

        # Ensure columns exist
        missing = [c for c in FEATURE_COLS if c not in self.feat_df.columns]
        if missing:
            raise ValueError(
                f"[iTransformerAgent] FEAT_CSV missing columns: {missing}. "
                "Run with --rebuild-features."
            )
        
        self.feature_cols = FEATURE_COLS
        self.target_col = "close"
        self.target_idx = self.feature_cols.index(self.target_col)

        print(f"[iTransformerAgent] Target column: {self.target_col}")
        print(f"[iTransformerAgent] Target index: {self.target_idx}")

        self._price_range = float(
            self.feat_df["close"].max() - self.feat_df["close"].min()
        )

        print(f"[iTransformerAgent] Feature DataFrame: {self.feat_df.shape}")
        return self

    # ── Preprocessing ─────────────────────────────────────────────────────────
    def preprocess(self) -> "iTransformerAgent":
        assert self.feat_df is not None, "Call load_data() first."
        avail = [c for c in FEATURE_COLS if c in self.feat_df.columns]
        self.model_cfg.enc_in = len(avail)
        self.model_cfg.dec_in = len(avail)
        self.model_cfg.c_out  = len(avail)
        self.prep = StockPreprocessor(feature_cols=avail)
        self.train_arr, self.val_arr, _ = self.prep.fit_transform(self.feat_df)
        self.prep.save(ITX_SCALER_PATH)
        return self

    def build_loaders(self):
        assert self.train_arr is not None, "Call preprocess() first."
        # StockSequenceDataset now returns (x, y, x_mark, y_mark) — 4 items
        return build_dataloaders(
            self.train_arr, self.val_arr,
            batch_size=self.train_cfg["batch_size"],
        )

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self) -> "iTransformerAgent":
        tl, vl = self.build_loaders()
        self.model = _make_model(self.model_cfg).to(self.device)
        self.model.enc_in = self.model.n_vars   # compatibility alias

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[iTransformerAgent] Model parameters: {n_params:,}")

        crit  = nn.MSELoss()
        opt   = torch.optim.AdamW(
            self.model.parameters(),
            lr           = self.train_cfg["lr"],
            weight_decay = self.train_cfg["weight_decay"],
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max   = self.train_cfg["epochs"],
            eta_min = self.train_cfg["lr"] * 0.01,
        )

        best = float("inf")
        pat  = 0

        print("\n" + "=" * 65)
        print(f"{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}  "
              f"{'Val MAE':>12}  {'LR':>10}")
        print("=" * 65)

        for ep in range(1, self.train_cfg["epochs"] + 1):
            t0     = time.time()
            trloss = _train_epoch(self.model, tl, opt, crit, self.device,
                                  self.train_cfg["grad_clip"])
            vloss, vmae = _eval_epoch(self.model, vl, crit, self.device)
            sched.step()
            lr = opt.param_groups[0]["lr"]
            elapsed = time.time() - t0

            print(f"{ep:>6}  {trloss:>12.6f}  {vloss:>12.6f}  "
                  f"{vmae:>12.6f}  {lr:>10.2e}  [{elapsed:.1f}s]")

            if vloss < best:
                best = vloss
                pat  = 0
                torch.save({
                    "epoch":      ep,
                    "model_cfg":  self.model_cfg,
                    "state_dict": self.model.state_dict(),
                    "val_loss":   vloss,
                    "val_mae":    vmae,
                }, ITX_MODEL_CKPT)
                print(f"           ✅  Best model saved (val_loss={best:.6f})")
            else:
                pat += 1
                if pat >= self.train_cfg["patience"]:
                    print(f"\n[iTransformerAgent] Early stopping at epoch {ep}")
                    break

        print("=" * 65)
        ckpt = torch.load(ITX_MODEL_CKPT, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        print(f"[iTransformerAgent] Loaded best checkpoint "
              f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
        return self

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate(self) -> dict:
        assert self.model is not None and self.val_arr is not None

        vl = DataLoader(
            StockSequenceDataset(self.val_arr, SEQ_LEN, PRED_LEN, TARGET_IDX),
            batch_size=self.train_cfg["batch_size"],
            shuffle=False,
        )

        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x, y, x_mark, y_mark in vl:
                x      = x.to(self.device)
                y      = y.to(self.device)
                x_mark = x_mark.to(self.device)
                y_mark = y_mark.to(self.device)

                outputs = self.model(x)  # iTransformer.forward only takes x
                preds = outputs[:, -PRED_LEN:]  # shape [B, pred_len]

                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        ps = np.vstack(all_preds)
        ts = np.vstack(all_targets)

        pr = self.prep.inverse_target(ps)
        tr = self.prep.inverse_target(ts)

        mae  = float(np.mean(np.abs(pr - tr)))
        rmse = float(np.sqrt(np.mean((pr - tr) ** 2)))

        print(f"\n[iTransformerAgent] Validation  MAE=₹{mae:.4f}  RMSE=₹{rmse:.4f}")
        return {"val_mae": mae, "val_rmse": rmse}
    
    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self) -> dict:
        assert self.model is not None and self.feat_df is not None

        metrics = self.evaluate()

        fs = self.prep.transform(self.feat_df)
        x  = torch.from_numpy(fs[-SEQ_LEN:]).unsqueeze(0).to(self.device).float()

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)  # iTransformer.forward only takes x
            ps = outputs[:, -PRED_LEN:].cpu().numpy()  # [1, pred_len]

        pp = self.prep.inverse_target(ps[0]).tolist()
        la = float(self.feat_df["close"].iloc[-1])

        result = {
            "model": "iTransformer",
            "predicted_prices": [round(p, 4) for p in pp],
            "trend": "Bullish" if pp[-1] > la else "Bearish",
            "confidence": round(
                _confidence_from_mae(metrics["val_mae"], self._price_range), 4
            ),
            "important_features": _top_features(self.model, x),  # uses attention safely
            "val_mae": round(metrics["val_mae"], 4),
            "val_rmse": round(metrics["val_rmse"], 4),
            "last_actual_price": round(la, 4),
        }

        with open(ITX_METRICS_PATH, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[iTransformerAgent] Metrics saved → {ITX_METRICS_PATH}")
        return result
    # ── Full pipeline ──────────────────────────────────────────────────────────
    def run(self, force_download: bool = False) -> dict:
        self.load_data(force_download)
        self.preprocess()
        self.train()
        result = self.predict()
        print("\n" + "=" * 55 + "\n🎯  iTransformer Prediction Output\n" + "=" * 55)
        for k, v in result.items():
            print(f"  {k:<25}: {v}")
        print("=" * 55)
        return result

    # ── Load saved model ──────────────────────────────────────────────────────
    def load_model(self) -> "iTransformerAgent":
        assert os.path.exists(ITX_MODEL_CKPT),  f"No checkpoint at {ITX_MODEL_CKPT}"
        assert os.path.exists(ITX_SCALER_PATH), f"No scaler at {ITX_SCALER_PATH}"

        ckpt = torch.load(ITX_MODEL_CKPT, map_location=self.device)

        self.model_cfg = ckpt["model_cfg"]
        self.model = _make_model(self.model_cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        # Patch compatibility attributes used by evaluate() / _top_features()
        self.model.enc_in   = self.model.n_vars
        self.model.pred_len = self.model.pred_len  # already set in __init__

        # ✅ Load preprocessor (source of truth)
        self.prep = StockPreprocessor.load(
            ITX_SCALER_PATH,
            feature_cols=getattr(self.model_cfg, "feature_cols", FEATURE_COLS),
        )

        # ✅ ALIGN EVERYTHING FROM PREPROCESSOR
        self.feature_cols = self.prep.feature_cols
        self.target_col   = self.prep.target_col
        self.target_idx   = self.prep.target_idx

        print(f"[iTransformerAgent] Loaded model ← {ITX_MODEL_CKPT}")
        print(f"[iTransformerAgent] Target column: {self.target_col}")
        print(f"[iTransformerAgent] Target index: {self.target_idx}")

        return self

    def infer(self, force_download: bool = False) -> dict:
        """Load saved checkpoint and run inference — no training."""
        self.load_data(force_download)
        self.preprocess()   # sets self.val_arr — needed by evaluate() inside predict()
        self.load_model()   # overwrites self.prep with the saved scaler
        return self.predict()

if __name__ == "__main__":
    iTransformerAgent().run()
