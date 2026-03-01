# Baseline model for evaluation.
# PyTorch implementation of the EigenmodeTransformer proposed in:
# "Fast grid-free strength mapping of multiple sound sources from microphone array data
#  using a Transformer architecture" by Kujawski et al.

import torch.nn as nn

class ViTEncoderLayer(nn.Module):

    def __init__(self, token_dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim, eps=1e-6)
        self.attn  = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(token_dim, eps=1e-6)
        self.mlp   = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        normed = self.norm2(x)
        x = x + self.mlp(normed)
        return x


class EigmodeTransformer(nn.Module):
    def __init__(
        self,
        nchannels    = 64,
        num_layers   = 12,
        num_heads    = 12,
        dropout_rate = 0.1,
        #lam          = 5.0,
    ):
        super().__init__()
        #self.lam = lam
        token_dim = nchannels * 2

        # --- stack 12 ViT encoder layers ---
        self.encoder = nn.Sequential(*[
            ViTEncoderLayer(token_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])

        self.norm     = nn.LayerNorm(token_dim, eps=1e-6)
        self.dropout  = nn.Dropout(dropout_rate)
        self.mlp_head = nn.Sequential(
            nn.Linear(nchannels * token_dim, 512), nn.GELU(), nn.Dropout(dropout_rate),
        )

        self.strength_head = nn.Linear(512, 1)
        self.loc_head      = nn.Linear(512, 2)  # Assuming 2D location output

    def forward(self, eigmodes):
        # --- 12 layers of ViT encoder ---
        tokens = self.encoder(eigmodes)
        # --- norm ---
        tokens = self.norm(tokens)
        # --- flatten ---
        tokens = tokens.flatten(1) # [B, nchannels*token_dim]
        # --- dropout ---
        tokens = self.dropout(tokens)
        # --- MLP head ---
        x = self.mlp_head(tokens)

        return self.strength_head(x), self.loc_head(x)