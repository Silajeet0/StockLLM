# ============================================================
# agents/tft_agent.py — Fixed (real-price confidence + smart data loading)
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tft_model import TemporalFusionTransformer
from utils.data_loader import download_data
from utils.feature_engineering import engineer_features
from utils.preprocessing import (
    split_dataframe, fit_scaler, scale_dataframe,
    save_scaler, load_scaler, inverse_transform_close,
)
from utils.sequence_builder import build_tft_dataloaders, TFTDataset
from config.settings import (
    HIDDEN_SIZE, LSTM_LAYERS, NUM_HEADS, DROPOUT, HORIZON,
    LR, MAX_EPOCHS, PATIENCE, BATCH_SIZE,
    PAST_FEATURES, FUTURE_FEATURES, ALL_FEATURES,
    CLOSE_IDX, LOOKBACK,
    TFT_MODEL_CKPT, TFT_SCALER_PATH, TFT_METRICS_PATH, FEAT_CSV,
    MODEL_SAVE_PATH,
)


# ============================================================
# Quantile Loss
# ============================================================
def quantile_loss(pred, target, quantiles=(0.1, 0.5, 0.9)):
    """
    Pinball / quantile loss.
    pred   : (B, T, 3)  — three quantile predictions
    target : (B, T)     — ground truth
    """
    loss = 0.0
    for i, q in enumerate(quantiles):
        e = target - pred[:, :, i]
        loss += torch.max((q - 1) * e, q * e).mean()
    return loss


# ============================================================
# Lightning Module
# ============================================================
class TFTLightningModule(L.LightningModule):

    def __init__(
        self,
        num_enc_vars: int   = len(ALL_FEATURES),
        num_dec_vars: int   = len(FUTURE_FEATURES),
        hidden_size:  int   = HIDDEN_SIZE,
        lstm_layers:  int   = LSTM_LAYERS,
        num_heads:    int   = NUM_HEADS,
        dropout:      float = DROPOUT,
        horizon:      int   = HORIZON,
        lr:           float = LR,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = TemporalFusionTransformer(
            num_enc_vars=num_enc_vars,
            num_dec_vars=num_dec_vars,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_heads=num_heads,
            dropout=dropout,
            horizon=horizon,
        )
        self.quantiles = (0.1, 0.5, 0.9)

    def forward(self, x_enc, x_dec):
        return self.model(x_enc, x_dec)

    def _shared_step(self, batch, stage: str):
        x_enc, x_dec, y = batch
        y_hat = self(x_enc, x_dec)
        loss  = quantile_loss(y_hat, y, self.quantiles)
        # Log quantile loss (scaled space) — used only for early stopping
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", patience=5, factor=0.5
            ),
            "monitor": "val_loss",
        }
        return [opt], [sched]


# ============================================================
# TFT Agent
# ============================================================
class TFTAgent:
    """
    Self-contained TFT Agent.

    Quick start
    -----------
        agent  = TFTAgent()
        result = agent.run()                    # full pipeline
        result = agent.run(retrain=False)       # load saved checkpoint

    Output schema (compatible with Orchestrator)
    --------------------------------------------
    {
        "model":             "TFT",
        "predicted_prices":  [p1..p5],
        "trend":             "Bullish" | "Bearish",
        "confidence":        float,             # real-price MAE based, matches iTransformer
        "important_features": [str, ...],
        "last_actual_price": float,
        "val_mae":           float,             # real ₹ MAE — comparable to iTransformer
        "val_rmse":          float,             # real ₹ RMSE
    }
    """

    def __init__(self, model_save_path: str = TFT_MODEL_CKPT):
        self.model_save_path  = model_save_path
        self.scaler           = None
        self.module           = None
        self.feat_df          = None
        self.full_scaled_df   = None
        self._val_scaled      = None   # stored for post-training evaluation
        self._price_range     = 1.0    # used in confidence formula
        # Real-price metrics — set by _evaluate_real_prices()
        self.val_mae_real     = None
        self.val_rmse_real    = None
        self.confidence       = 0.5    # default until evaluated

    # ── Step 1: Load + engineer features ─────────────────────────────────
    def load_data(self, force_download: bool = False) -> "TFTAgent":
        """
        Smart feature loading:
          1. If cached CSV exists and has ALL required TFT columns → use it.
          2. If CSV exists but is missing time features (e.g. iTransformer wrote it)
             → re-engineer from cached raw data (no re-download unless forced).
          3. If force_download=True → download fresh raw data then engineer.
        """
        if os.path.exists(FEAT_CSV) and not force_download:
            cached = pd.read_csv(FEAT_CSV, parse_dates=["date"])
            missing_tft_cols = [c for c in ALL_FEATURES if c not in cached.columns]
            if not missing_tft_cols:
                # CSV already has all 21 TFT features — use directly
                self.feat_df = cached
                self._price_range = float(cached["close"].max() - cached["close"].min())
                print(f"[TFTAgent] Loaded cached features: {self.feat_df.shape}")
                return self
            else:
                # CSV is missing time features (iTransformer's 18-col version)
                # Re-engineer from raw (raw CSV is already cached — no re-download)
                print(f"[TFTAgent] Cached CSV missing time features — re-engineering "
                      f"(raw data will NOT be re-downloaded).")
                raw = download_data(force=False)   # ← always use cached raw
                self.feat_df = engineer_features(raw)
                self._price_range = float(self.feat_df["close"].max() - self.feat_df["close"].min())
                return self

        # force_download=True or no CSV at all
        raw = download_data(force=force_download)
        self.feat_df = engineer_features(raw)
        self._price_range = float(self.feat_df["close"].max() - self.feat_df["close"].min())

        missing = [c for c in ALL_FEATURES if c not in self.feat_df.columns]
        if missing:
            raise ValueError(f"[TFTAgent] Feature CSV still missing: {missing}")

        print(f"[TFTAgent] Feature DataFrame: {self.feat_df.shape}")
        return self

    # ── Step 2: Preprocess ────────────────────────────────────────────────
    def preprocess(self) -> tuple:
        """Split → fit scaler on train only → scale all splits."""
        assert self.feat_df is not None, "Call load_data() first."

        train_df, val_df, _ = split_dataframe(self.feat_df)

        self.scaler = fit_scaler(train_df, features=ALL_FEATURES)
        save_scaler(self.scaler, TFT_SCALER_PATH)

        train_scaled = scale_dataframe(train_df, self.scaler, features=ALL_FEATURES)
        val_scaled   = scale_dataframe(val_df,   self.scaler, features=ALL_FEATURES)

        # Store val_scaled so _evaluate_real_prices() can use it after training
        self._val_scaled = val_scaled

        # Full-dataset scaled version for prediction
        self.full_scaled_df = scale_dataframe(
            self.feat_df, self.scaler, features=ALL_FEATURES
        )
        return train_scaled, val_scaled

    # ── Step 3: Post-training real-price evaluation ───────────────────────
    def _evaluate_real_prices(self) -> tuple:
        """
        Run the best-checkpoint model over the validation set and compute
        MAE + RMSE in REAL RUPEE units (not scaled space).

        This gives a confidence metric directly comparable to iTransformer's.
        Called automatically after train() and load_model().
        """
        assert self.module     is not None, "Model not loaded."
        assert self._val_scaled is not None, "val_scaled not set — call preprocess() first."

        val_ds     = TFTDataset(self._val_scaled, LOOKBACK, HORIZON)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model  = self.module.model
        device = next(self.module.parameters()).device
        model.eval()

        all_pred, all_true = [], []
        with torch.no_grad():
            for x_enc, x_dec, y in val_loader:
                x_enc, x_dec = x_enc.to(device), x_dec.to(device)
                # Use P50 (median) quantile — index 1
                pred = model(x_enc, x_dec)[:, :, 1]
                all_pred.append(pred.cpu().numpy())
                all_true.append(y.numpy())

        pred_scaled = np.vstack(all_pred)   # [N, horizon]
        true_scaled = np.vstack(all_true)

        pred_real = inverse_transform_close(pred_scaled, self.scaler, CLOSE_IDX)
        true_real = inverse_transform_close(true_scaled, self.scaler, CLOSE_IDX)

        mae  = float(np.mean(np.abs(pred_real - true_real)))
        rmse = float(np.sqrt(np.mean((pred_real - true_real) ** 2)))

        print(f"[TFTAgent] Real-price validation  MAE=₹{mae:.2f}  RMSE=₹{rmse:.2f}")

        self.val_mae_real  = mae
        self.val_rmse_real = rmse

        # Same formula as iTransformer — now directly comparable
        self.confidence = float(np.clip(1.0 - 2.0 * (mae / self._price_range), 0.05, 0.99))
        print(f"[TFTAgent] Confidence: {self.confidence:.1%}  "
              f"(MAE/price_range = {mae:.2f}/{self._price_range:.2f} = {mae/self._price_range:.3f})")

        return mae, rmse

    # ── Step 4: Train ─────────────────────────────────────────────────────
    def train(
        self,
        train_loader,
        val_loader,
        max_epochs: int = MAX_EPOCHS,
        patience:   int = PATIENCE,
        log_dir:    str = "logs/tft",
    ) -> "TFTAgent":

        self.module = TFTLightningModule()
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=patience, mode="min", verbose=True,
            ),
            ModelCheckpoint(
                dirpath=os.path.dirname(self.model_save_path),
                filename="tft_best",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                verbose=True,
            ),
        ]

        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            logger=CSVLogger(log_dir, name="tft"),
            enable_progress_bar=True,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
        )

        trainer.fit(self.module, train_loader, val_loader)

        ckpt_path = callbacks[1].best_model_path
        print(f"\n[TFTAgent] Best checkpoint: {ckpt_path}")

        self.module = TFTLightningModule.load_from_checkpoint(ckpt_path)
        self.module.eval()

        # ── Compute real-price metrics (the key fix) ──────────────────────
        self._evaluate_real_prices()
        return self

    # ── Step 5: Load saved checkpoint ─────────────────────────────────────
    def load_model(self, ckpt_path: str = None) -> "TFTAgent":
        ckpt_path = ckpt_path or self.model_save_path
        ckpt_dir  = os.path.dirname(ckpt_path)
        ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")])
        if not ckpt_files:
            raise FileNotFoundError(
                f"[TFTAgent] No .ckpt file found in {ckpt_dir}. Run training first."
            )
        best_ckpt = os.path.join(ckpt_dir, ckpt_files[-1])
        self.module = TFTLightningModule.load_from_checkpoint(best_ckpt)
        self.module.eval()
        print(f"[TFTAgent] Loaded checkpoint ← {best_ckpt}")

        # Recompute real-price confidence on the saved val data
        if self._val_scaled is not None:
            self._evaluate_real_prices()
        else:
            print("[TFTAgent] ⚠️  val_scaled not available — confidence will use default 0.5")
        return self

    # ── Step 6: Predict ───────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, scaled_df: pd.DataFrame) -> dict:
        """
        Predict next HORIZON trading days of close prices.
        Confidence is computed from real-price MAE (same method as iTransformer).
        """
        if self.module is None:
            raise RuntimeError("[TFTAgent] Model not loaded. Call train() or load_model().")
        if self.scaler is None:
            raise RuntimeError("[TFTAgent] Scaler not set. Call preprocess() first.")

        self.module.eval()
        model  = self.module.model
        device = next(self.module.parameters()).device

        # ── Encoder: last LOOKBACK rows ───────────────────────────────────
        enc_arr = scaled_df[ALL_FEATURES].values[-LOOKBACK:].astype(np.float32)

        # ── Decoder: known future calendar features ───────────────────────
        last_date    = pd.to_datetime(scaled_df["date"].iloc[-1])
        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=1), periods=HORIZON, freq="B"
        )
        dec_raw = np.column_stack([
            future_dates.dayofweek.values,
            future_dates.day.values,
            future_dates.month.values,
            future_dates.isocalendar().week.values.astype(int),
        ]).astype(np.float32)

        # Scale decoder features through the trained scaler
        n_past  = len(PAST_FEATURES)
        dummy   = np.zeros((HORIZON, len(ALL_FEATURES)), dtype=np.float32)
        dummy[:, n_past:] = dec_raw
        dec_arr = self.scaler.transform(dummy)[:, n_past:].astype(np.float32)

        x_enc = torch.tensor(enc_arr).unsqueeze(0).to(device)
        x_dec = torch.tensor(dec_arr).unsqueeze(0).to(device)

        # ── Forward pass — use P50 (index 1) ─────────────────────────────
        preds_scaled     = model(x_enc, x_dec)[:, :, 1].squeeze(0).cpu().numpy()
        predicted_prices = inverse_transform_close(preds_scaled, self.scaler, CLOSE_IDX)

        # ── Last actual close (real ₹) ────────────────────────────────────
        last_close_scaled = float(scaled_df[ALL_FEATURES].values[-1, CLOSE_IDX])
        last_actual       = float(
            inverse_transform_close(np.array([last_close_scaled]), self.scaler, CLOSE_IDX)[0]
        )

        trend = "Bullish" if predicted_prices[-1] > last_actual else "Bearish"

        # ── Feature importance ────────────────────────────────────────────
        _ = model(x_enc, x_dec)   # populate stored attention weights
        importance = model.get_feature_importance(
            enc_feature_names=ALL_FEATURES,
            dec_feature_names=FUTURE_FEATURES,
            top_k=5,
        )

        result = {
            "model":              "TFT",
            "predicted_prices":   [round(float(p), 4) for p in predicted_prices],
            "trend":              trend,
            "confidence":         round(self.confidence, 4),   # real-price based
            "important_features": importance["top_encoder_features"],
            "last_actual_price":  round(last_actual, 4),
            "val_mae":            round(self.val_mae_real,  4) if self.val_mae_real  is not None else None,
            "val_rmse":           round(self.val_rmse_real, 4) if self.val_rmse_real is not None else None,
        }

        with open(TFT_METRICS_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[TFTAgent] Metrics saved → {TFT_METRICS_PATH}")

        print("\n" + "=" * 55)
        print("🎯  TFT Prediction Output")
        print("=" * 55)
        for k, v in result.items():
            print(f"  {k:<25}: {v}")
        print("=" * 55)
        return result

    # ── Full self-contained pipeline ──────────────────────────────────────
    def run(self, force_download: bool = False, retrain: bool = True) -> dict:
        """
        Complete pipeline: load data → preprocess → train → predict.

        retrain=False: loads saved checkpoint + saved scaler.
        If either artefact is missing, falls back to training automatically.
        """
        self.load_data(force_download=force_download)

        ckpt_dir   = os.path.dirname(TFT_MODEL_CKPT)
        has_ckpt   = (
            os.path.isdir(ckpt_dir) and
            any(f.endswith(".ckpt") for f in os.listdir(ckpt_dir))
        )
        has_scaler = os.path.exists(TFT_SCALER_PATH)

        if retrain or not has_ckpt or not has_scaler:
            if not retrain:
                missing = ([" checkpoint"] if not has_ckpt else []) + \
                          (["scaler"]      if not has_scaler else [])
                print(f"[TFTAgent] --no-retrain requested but {missing} not found. Training.")
            train_scaled, val_scaled = self.preprocess()
            train_loader, val_loader, _, _ = build_tft_dataloaders(train_scaled, val_scaled)
            self.train(train_loader, val_loader)
        else:
            print("[TFTAgent] --no-retrain: loading saved scaler and checkpoint.")
            self.scaler         = load_scaler(TFT_SCALER_PATH)
            # Preprocess to set _val_scaled (needed for real-price confidence)
            _, val_scaled       = self.preprocess()
            self.load_model()

        return self.predict(self.full_scaled_df)

    def save(self, path: str = None):
        path = path or self.model_save_path
        if self.module is not None:
            torch.save(self.module.state_dict(), path)
            print(f"[TFTAgent] Saved → {path}")

    def load(self, ckpt_path: str):
        self.module = TFTLightningModule.load_from_checkpoint(ckpt_path)
        self.module.eval()
        print(f"[TFTAgent] Loaded ← {ckpt_path}")


# ============================================================
if __name__ == "__main__":
    agent  = TFTAgent()
    result = agent.run()
