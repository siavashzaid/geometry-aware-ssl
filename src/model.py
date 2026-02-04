import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

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
        mpnn_num_layers: int = 1,
        mpnn_dropout: float = 0.0,
        # --- self-attention encoder params --- #
        attn_num_heads: int = 8, # token dim must be divisible by attn_num_heads
        attn_num_layers: int = 2,
        attn_dropout: float = 0.0,
        # --- prediction head params --- #
        head_mlp_hidden_dim: int = 512,
        num_output_sources: int = 1,
        head_dropout: float = 0.0,
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
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        
        # 1) graph -> mic tokens: [N, D]
        tokens = self.tokenizer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # 2) tokens -> contextualized token: [D] (or [B, D] if tokens were batched)
        if batch is None:
            pooled = self.encoder(tokens)
        # batch nodes for transformer
        else:    
            batched_tokens, src_key_padding_mask = self.batch_nodes_for_transformer(tokens, batch_vector=batch)
            pooled = self.encoder(batched_tokens, src_key_padding_mask=src_key_padding_mask) 

        # 3) pooled -> locations: [I, 2] (or [B, I, 2])
        locations = self.head(pooled)
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
        batch = getattr(data, 'batch', None)
        return self.forward(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=batch)

    @staticmethod 
    def batch_nodes_for_transformer(node_embeddings, batch_vector):
        """
        Convert PyG flat batched node embeddings to dense 3D tensor for Transformer.
        - PyG batches graphs by concatenating all nodes into a flat [N, D] tensor
        - Transformers require dense [B, L, D] tensors where L (seq_len) is uniform
    
        Expected inputs (PyTorch Geometric style):
        node_embeddings: [N, hidden_dim]  — flat tensor with all nodes from all graphs
        batch_vector:    [N]              — graph assignment index (e.g. [0,0,1,1,1,2,...])
        
        Output:
        padded:                  [B, max_nodes, hidden_dim] — dense tensor, zero-padded
        src_key_padding_mask:    [B, max_nodes]             — True where padded (ignore in attention)
        
        """
        # Get device so that embeddings and masks are on the same device (GPU/CPU)
        device = node_embeddings.device

        # Split the flat node tensor into a list of per-graph tensors
        node_lists = []
        for i in range(batch_vector.max().item() + 1):
            mask = (batch_vector == i)
            node_lists.append(node_embeddings[mask])  # (num_nodes_graph_i, hidden_dim)
        
        # Pad all graphs to max_nodes in dimension 1 (sequence length)
        padded = pad_sequence(node_lists, batch_first=True, padding_value=0.0).to(device)  # (batch_size, max_num_nodes, hidden_dim)
        
        # Create attention mask: True where PADDED (TransformerEncoder ignores True positions)
        lengths = torch.tensor([n.size(0) for n in node_lists], device=device)  # (batch_size,)
        max_len = padded.size(1)
        src_key_padding_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        src_key_padding_mask = ~src_key_padding_mask # [batch_size, max_num_nodes]
        
        return padded, src_key_padding_mask