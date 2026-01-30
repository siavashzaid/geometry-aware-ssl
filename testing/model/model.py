import torch
import torch.nn as nn

from modules import MPNNLayer, MPNNTokenizer, SelfAttentionEncoder, PredictionHead

class MPNNTransformerModel(nn.Module):
    """
    Full pipeline:
      data (f.e. h5Dataset) -> MPNNTokenizer -> SelfAttentionEncoder -> PredictionHead

    Expected inputs (PyGeometric style):
      x:         [N, node_in_dim]
      edge_index:[2, E]
      edge_attr: [E, edge_in_dim]

    Output:
      locations: [I, 2] (unbatched) or [B, I, 2] if you pass batched tokens later
    """
    def __init__(
        self,
        # --- tokenizer params --- #
        node_in_dim: int,
        edge_in_dim: int,
        mpnn_hidden_dim: int = 128,
        token_dim: int = 128,
        mpnn_num_layers: int = 2,
        mpnn_dropout: float = 0.0,
        # --- self-attention encoder params --- #
        attn_num_heads: int = 8, # token dim must be divisible by attn_num_heads
        attn_num_layers: int = 12,
        attn_dropout: float = 0.1,
        # --- prediction head params --- #
        head_mlp_hidden_dim: int = 512,
        num_output_sources: int = 1,
        head_dropout: float = 0.1,
    ):
        super().__init__()

        # --- tokenizer (graph -> mic tokens) ---
        self.tokenizer = MPNNTokenizer(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=mpnn_hidden_dim,
            out_dim=token_dim,
            num_layers=mpnn_num_layers,
            dropout=mpnn_dropout,
        )

        # --- self-attention encoder (tokens -> pooled embedding) ---
        self.encoder = SelfAttentionEncoder(
            embed_dim=token_dim,
            num_heads=attn_num_heads,
            num_layers=attn_num_layers,
            dropout=attn_dropout,
        )

        # --- prediction head (pooled embedding -> outputs) ---
        self.head = PredictionHead(
            embed_dim=token_dim,
            mlp_hidden_dim=head_mlp_hidden_dim,
            num_output_sources=num_output_sources,
            dropout=head_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        
        # 1) graph -> mic tokens: [N, D]
        tokens = self.tokenizer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # 2) tokens -> pooled: [D] (or [B, D] if tokens were batched)
        encoded = self.encoder(tokens)

        # 3) pooled -> locations: [I, 2] (or [B, I, 2])
        locations = self.head(encoded)
        return locations

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        return self.forward(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def forward_from_data(self, data) -> torch.Tensor:
        """
        Convenience for PyG Data/Batch objects that expose .x, .edge_index, .edge_attr
        """
        return self.forward(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
