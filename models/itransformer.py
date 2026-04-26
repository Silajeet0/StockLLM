"""
iTransformer Model for Multi-Horizon Stock Price Forecasting
============================================================
Architecture: Inverted Transformer — attention across variables (features),
not across time steps. Each feature's full time-series is treated as ONE token.

Paper: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
       Liu et al., 2024 (ICLR)

Flow:
  [Batch, Seq_Len, N_Features]
        ↓  Transpose
  [Batch, N_Features, Seq_Len]
        ↓  Linear Embedding (per-variable)
  [Batch, N_Features, d_model]
        ↓  N × TransformerEncoderLayer  (attention across features)
  [Batch, N_Features, d_model]
        ↓  Projection Head
  [Batch, N_Features, Pred_Len]
        ↓  Select target variable (close = index 0)
  [Batch, Pred_Len]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding (applied to the feature dimension, i.e. "variable axis")
# ---------------------------------------------------------------------------
class VariablePositionalEncoding(nn.Module):
    """Learned positional encoding for the variable (feature) dimension."""

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_vars, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_vars, d_model]
        return self.dropout(x + self.pos_embed)


# ---------------------------------------------------------------------------
# Feed-Forward Network inside each Transformer block
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Single iTransformer Encoder Block
# ---------------------------------------------------------------------------
class iTransformerBlock(nn.Module):
    """
    One encoder block of the iTransformer.
    Self-attention runs across the VARIABLE dimension (N_vars tokens),
    where each token contains the full temporal embedding of one feature.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,   # [B, N_vars, d_model]
        )
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_vars, d_model]
        # --- Self-Attention with Pre-LN ---
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.drop(attn_out)

        # --- Feed-Forward with Pre-LN ---
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)

        return x


# ---------------------------------------------------------------------------
# Main iTransformer
# ---------------------------------------------------------------------------
class iTransformer(nn.Module):
    """
    iTransformer for multivariate time-series forecasting.

    Parameters
    ----------
    seq_len    : look-back window length (e.g. 60 days)
    pred_len   : forecast horizon       (e.g. 5 days)
    n_vars     : number of input features (e.g. 19)
    d_model    : embedding dimension
    n_heads    : number of attention heads
    n_layers   : number of stacked Transformer blocks
    d_ff       : feed-forward hidden size
    dropout    : dropout probability
    target_idx : column index of the target variable in the input feature matrix
                 (default 0 → first column should be 'close' after reordering)
    """

    def __init__(
        self,
        seq_len: int = 60,
        pred_len: int = 5,
        n_vars: int = 19,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        target_idx: int = 0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.target_idx = target_idx

        # --- 1. Per-variable embedding: map each variable's time series
        #        [seq_len] → [d_model]
        self.input_projection = nn.Linear(seq_len, d_model)

        # --- 2. Positional encoding over variable tokens
        self.var_pos_enc = VariablePositionalEncoding(n_vars, d_model, dropout)

        # --- 3. Stack of iTransformer blocks
        self.encoder_layers = nn.ModuleList(
            [iTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # --- 4. Projection head: d_model → pred_len (per variable)
        self.output_projection = nn.Linear(d_model, pred_len)

        self._init_weights()

    # -----------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape [B, seq_len, n_vars]
            Normalised multivariate input window.

        Returns
        -------
        torch.Tensor  shape [B, pred_len]
            Predicted values for the TARGET variable over the next pred_len steps.
        """
        B, T, N = x.shape
        assert T == self.seq_len,  f"Expected seq_len={self.seq_len}, got {T}"
        assert N == self.n_vars,   f"Expected n_vars={self.n_vars}, got {N}"

        # --- Transpose: [B, T, N] → [B, N, T]
        x = x.permute(0, 2, 1)          # [B, N_vars, seq_len]

        # --- Per-variable linear embedding
        x = self.input_projection(x)    # [B, N_vars, d_model]

        # --- Add variable positional encoding
        x = self.var_pos_enc(x)         # [B, N_vars, d_model]

        # --- Transformer encoder (attention across variables)
        for layer in self.encoder_layers:
            x = layer(x)                # [B, N_vars, d_model]
        x = self.encoder_norm(x)

        # --- Output projection: [B, N_vars, pred_len]
        x = self.output_projection(x)   # [B, N_vars, pred_len]

        # --- Select target variable only → [B, pred_len]
        out = x[:, self.target_idx, :]  # [B, pred_len]

        return out

    # -----------------------------------------------------------------------
    def get_var_attention_weights(self, x: torch.Tensor):
        """
        Extract attention weights from the FIRST encoder layer for
        feature-importance analysis.

        Returns
        -------
        attn_weights : torch.Tensor  shape [B, n_heads, N_vars, N_vars]
        """
        B, T, N = x.shape
        x = x.permute(0, 2, 1)
        x = self.input_projection(x)
        x = self.var_pos_enc(x)

        # Run only the first layer and grab weights
        layer = self.encoder_layers[0]
        x_norm = layer.norm1(x)
        _, attn_weights = layer.attn(x_norm, x_norm, x_norm, average_attn_weights=False)
        return attn_weights   # [B, n_heads, N_vars, N_vars]


# ---------------------------------------------------------------------------
# Lightning wrapper for clean training loops
# ---------------------------------------------------------------------------
try:
    import pytorch_lightning as pl
    from torch.optim.lr_scheduler import CosineAnnealingLR

    class iTransformerLightning(pl.LightningModule):
        """
        PyTorch Lightning wrapper around iTransformer.
        Handles training, validation, optimiser, and LR scheduling.
        """

        def __init__(self, model_cfg: dict, train_cfg: dict):
            super().__init__()
            self.save_hyperparameters()
            self.model = iTransformer(**model_cfg)
            self.train_cfg = train_cfg
            self.criterion = nn.MSELoss()

        def forward(self, x):
            return self.model(x)

        def _shared_step(self, batch, stage: str):
            x, y = batch          # x: [B, seq, n_vars]  y: [B, pred_len]
            pred = self(x)
            loss = self.criterion(pred, y)
            mae  = F.l1_loss(pred, y)
            self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
            self.log(f"{stage}_mae",  mae,  prog_bar=True, on_epoch=True, on_step=False)
            return loss

        def training_step(self, batch, batch_idx):
            return self._shared_step(batch, "train")

        def validation_step(self, batch, batch_idx):
            return self._shared_step(batch, "val")

        def configure_optimizers(self):
            lr  = self.train_cfg.get("lr", 1e-4)
            wd  = self.train_cfg.get("weight_decay", 1e-4)
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
            scheduler = CosineAnnealingLR(
                opt,
                T_max=self.train_cfg.get("epochs", 50),
                eta_min=lr * 0.01,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

except ImportError:
    pass  # Lightning optional; plain PyTorch training also supported


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("iTransformer — Architecture Sanity Check")
    print("=" * 60)

    cfg = dict(seq_len=60, pred_len=5, n_vars=19, d_model=128,
               n_heads=8, n_layers=3, d_ff=256, dropout=0.1, target_idx=0)

    model = iTransformer(**cfg)
    dummy_input = torch.randn(8, 60, 19)   # [batch=8, seq=60, vars=19]
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Input  shape : {list(dummy_input.shape)}")
    print(f"Output shape : {list(output.shape)}")
    print(f"Trainable params: {total_params:,}")

    # Attention weights
    attn = model.get_var_attention_weights(dummy_input)
    print(f"Attn weights : {list(attn.shape)}")
    print("=" * 60)
    print("✅  Model OK")
