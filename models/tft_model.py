import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Gated Linear Unit
# ============================================================

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.gate = nn.Linear(input_size, output_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc(x)) * torch.sigmoid(self.gate(x))


# ============================================================
# 2. Gated Residual Network
# ============================================================

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.ctx = nn.Linear(context_size, hidden_size, bias=False) if context_size else None

        self.glu = GatedLinearUnit(hidden_size, output_size, dropout)
        self.norm = nn.LayerNorm(output_size)

        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x, context=None):
        residual = self.skip(x)

        h = F.elu(self.fc1(x))
        if context is not None and self.ctx is not None:
            h = h + self.ctx(context)

        h = self.fc2(h)
        return self.norm(self.glu(h) + residual)


# ============================================================
# 3. Variable Selection Network
# ============================================================

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_vars, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_vars = num_vars

        self.var_projs = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(num_vars)
        ])

        self.weight_grn = GatedResidualNetwork(
            num_vars * hidden_size,
            hidden_size,
            num_vars,
            dropout
        )

    def forward(self, x: torch.Tensor):
        B, T, V = x.shape

        # 🔥 FORCE DEVICE CONSISTENCY
        device = next(self.parameters()).device
        x = x.to(device)

        embeds = torch.stack(
            [self.var_projs[i](x[:, :, i:i+1]) for i in range(V)],
            dim=2
        )

        flat    = embeds.view(B, T, V * self.hidden_size)
        weights = torch.softmax(self.weight_grn(flat), dim=-1)

        combined = (embeds * weights.unsqueeze(-1)).sum(dim=2)
        return combined, weights


# ============================================================
# 4. Interpretable Attention
# ============================================================

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, self.d_head)

        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B = query.size(0)

        Q = self.q(query).view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k(key).view(B, -1, self.num_heads, self.d_head).transpose(1, 2)

        V = self.v(value).unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        return self.o(out), attn.mean(dim=1)


# ============================================================
# 5. Temporal Fusion Transformer (FIXED)
# ============================================================

class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        num_enc_vars,
        num_dec_vars,
        hidden_size=64,
        lstm_layers=2,
        num_heads=4,
        dropout=0.1,
        horizon=5,
        quantiles=(0.1, 0.5, 0.9)
    ):
        super().__init__()

        self.horizon = horizon
        self.quantiles = quantiles
        self.Q = len(quantiles)

        H = hidden_size

        # VSN
        self.enc_vsn = VariableSelectionNetwork(num_enc_vars, H, dropout)
        self.dec_vsn = VariableSelectionNetwork(num_dec_vars, H, dropout)

        # LSTM
        self.encoder = nn.LSTM(H, H, lstm_layers, batch_first=True)
        self.decoder = nn.LSTM(H, H, lstm_layers, batch_first=True)

        # Gating
        self.enc_gate = GatedLinearUnit(H, H, dropout)
        self.dec_gate = GatedLinearUnit(H, H, dropout)
        self.norm = nn.LayerNorm(H)

        # Attention
        self.attn = InterpretableMultiHeadAttention(H, num_heads, dropout)

        # FFN
        self.ff = GatedResidualNetwork(H, H * 4, H, dropout)

        # 🔥 QUANTILE OUTPUT HEAD (FIX)
        self.output_proj = nn.Linear(H, self.Q)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ========================================================
    # FORWARD
    # ========================================================
    def forward(self, x_enc, x_dec):

        enc_emb, self.enc_var_weights = self.enc_vsn(x_enc)
        dec_emb, self.dec_var_weights = self.dec_vsn(x_dec)

        enc_out, (h, c) = self.encoder(enc_emb)
        enc_out = self.norm(self.enc_gate(enc_out) + enc_emb)

        dec_out, _ = self.decoder(dec_emb, (h, c))
        dec_out = self.norm(self.dec_gate(dec_out) + dec_emb)

        enriched = self.ff(dec_out)

        context = torch.cat([enc_out, enriched], dim=1)

        attn_out, self.attn_weights = self.attn(
            enriched, context, context
        )

        out = self.norm(attn_out + enriched)

        # (B, 5, 3)
        quantiles = self.output_proj(out)

        return quantiles


    # ========================================================
    # FEATURE IMPORTANCE (RESTORED)
    # ========================================================
    @torch.no_grad()
    def get_feature_importance(self, enc_feature_names, dec_feature_names, top_k=5):

        enc_imp = self.enc_var_weights.mean(dim=(0, 1)).cpu().numpy()
        dec_imp = self.dec_var_weights.mean(dim=(0, 1)).cpu().numpy()

        enc_ranked = sorted(zip(enc_feature_names, enc_imp), key=lambda x: x[1], reverse=True)[:top_k]
        dec_ranked = sorted(zip(dec_feature_names, dec_imp), key=lambda x: x[1], reverse=True)[:top_k]

        return {
            "top_encoder_features": [f for f, _ in enc_ranked],
            "top_decoder_features": [f for f, _ in dec_ranked],
        }