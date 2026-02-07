import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MPNNLayer(MessagePassing):
    """
    One message-passing block with edge features, mean aggregation,
    residual connection, and LayerNorm to reduce oversmoothing in fully-connected graphs.

    Input/Output node dim stays constant: hidden_dim -> hidden_dim
    """
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0):
        
        super().__init__(aggr="mean") # fully connected graph, so messages dont blow up with "add" aggregation

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        #self.norm = nn.LayerNorm(hidden_dim) # test later on if layernorm helps

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # propagate: message passing + aggregation + update
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        
        #return self.norm(x + out) #test later on if layernorm helps
        return x + out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x_i: target node features [E, H]
        # x_j: source node features [E, H]
        # edge_attr:             [E, E_dim]
        msg_in = torch.cat([x_i, x_j, edge_attr], dim=-1)  # [E, 2H + E_dim]
        return self.msg_mlp(msg_in)                        # [E, H]

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # aggr_out: [N, H], x: [N, H]
        upd_in = torch.cat([x, aggr_out], dim=-1)          # [N, 2H]
        return self.upd_mlp(upd_in)                        # [N, H]

class MPNNTokenizer(nn.Module):
    """
    MPNN-based tokenizer to convert graphs into token embeddings for attention mechanism:
      node_in -> hidden_dim -> (L x MPNNLayer) -> out_dim
    """
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        # --- project node features to hidden dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # --- multiple MPNN layers
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim=hidden_dim, edge_dim=edge_in_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # --- final projection to the token dim you want for cross-attention
        self.node_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x:         [N, node_in_dim]
        edge_index:[2, E]
        edge_attr: [E, edge_in_dim]
        returns:   [N, out_dim] microphone embeddings (tokens)
        """
        h = self.node_encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.node_head(h)

class SelfAttentionEncoder(nn.Module):
    """
    Transformer encoder following ViT-Base architecture:
    - 12 layers of multi-head self-attention (8 heads, D=128)
    - Takes microphone embeddings and outputs pooled vector for MLP head
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8, # embed_dim must be divisible by num_heads 
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  
        
        # --- Multihead self-attention layer ---
        multihead_self_attention_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # typical 4x expansion in transformer FFN
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # pre-norm for better training stability
        )
        
        # -- Transformer encoder with multiple multihead self-attention layers ---
        self.transformer_encoder = nn.TransformerEncoder(
            multihead_self_attention_layer,
            num_layers=num_layers,
        )
        
    def forward(self, tokens: torch.Tensor, src_key_padding_mask: torch.tensor = None) -> torch.Tensor:
        """
        tokens: [N, embed_dim] or [B, N, embed_dim] microphone embeddings from MPNNTokenizer
        returns: [embed_dim] or [B, embed_dim] encoded features after global pooling
        """
        # Adds batch dimension if needed: [N, D] -> [1, N, D]
        squeeze_output = False
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
            squeeze_output = True
        
        
        # --- Add CLS token ---
        B, N, D = tokens.shape

        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        tokens = torch.cat((cls_token, tokens), dim=1)  # [B, N+1, D]

        # Adjust src_key_padding_mask for CLS token
        if src_key_padding_mask is not None:    
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        # --- Multihead self-attention layers ---
        encoded = self.transformer_encoder(tokens, src_key_padding_mask=src_key_padding_mask)  # [B, N, D]
        
        # --- Global pooling ---
        #Alternative 1: mean over all microphone tokens
        #pooled = encoded.mean(dim=1)  # [B, D]
        #Alternative 2: use CLS token as global representation
        pooled = encoded[:, 0, :]  # [B, D] use CLS token
        
        # Remove batch dimension if input was unbatched
        if squeeze_output:
            pooled = pooled.squeeze(0)  # [D]
        
        return pooled

class PredictionHead(nn.Module):
    """
    Two-layer MLP (512 neurons each) that outputs source locations and strengths.
    - Location head: outputs I source locations (x, y coordinates)
    - Strength head: outputs normalized strengths via Softmax
    """
    def __init__(
        self,
        embed_dim: int = 128,
        mlp_hidden_dim: int = 512,
        num_output_sources: int = 1,  # I = 1 source component
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_output_sources = num_output_sources
        
        # --- Two-layer MLP with 512 neurons each ---
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # --- Source location head (2D coordinates per source) ---
        self.location_head = nn.Linear(mlp_hidden_dim, num_output_sources * 2)
        
        # --- Source strength head (normalized via softmax) ---
        #self.strength_head = nn.Linear(mlp_hidden_dim, num_output_sources)
        
    def forward(self, encoded_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        encoded_features: [embed_dim] or [B, embed_dim] from TransformerEncoder
        
        Returns:
        --------
        locations: [num_output_sources, 2] or [B, num_output_sources, 2] 
                   predicted source locations (x, y)
        strengths: [num_output_sources] or [B, num_output_sources]
                   normalized source strengths (sum to 1)
        """
        # Handle both batched and unbatched input
        squeeze_output = False
        if encoded_features.dim() == 1:
            encoded_features = encoded_features.unsqueeze(0)
            squeeze_output = True
        
        # --- MLP processing ---
        features = self.mlp(encoded_features)  # [B, mlp_hidden_dim]
        
        # --- LOCATION HEAD OUTPUT ---
        # Raw output is [B, num_output_sources * 2]
        locations = self.location_head(features)  # [B, I * 2]
        # Reshape to [B, I, 2] where each source has (x, y) coordinates
        locations = locations.view(-1, self.num_output_sources, 2)  # [B, I, 2]
        
        # --- STRENGTH HEAD OUTPUT ---
        # Raw output is [B, num_output_sources]
        #strengths = self.strength_head(features)  # [B, I]
        # Apply softmax to normalize strengths (they sum to 1)
        #strengths = torch.softmax(strengths, dim=-1)  # [B, I]
        
        # Remove batch dimension if input was unbatched
        if squeeze_output:
            locations = locations.squeeze(0)  # [I, 2]
            #strengths = strengths.squeeze(0)  # [I]
        
        return locations# , strengths

