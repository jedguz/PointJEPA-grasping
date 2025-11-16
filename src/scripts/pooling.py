# scripts/pooling.py
"""
Script containing various pooling mechanisms for aggregating patch embeddings.
The best performance is yielded by pooling with attention mechanisms.

Input: x of shape (B, L, D)
Output: (B, D)

L is the number of patches (tokens), D is the feature dimension, and B is the batch size.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MeanPool(nn.Module):
    """
    Mean-pooling over token dimension.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

class MaxPool(nn.Module):
    """
    Max-pooling over token dimension.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1).values

class AttentionPool(nn.Module):
    """
    Attention-based pooling with learnable query vector.
    Uses multi-head attention to aggregate patch embeddings.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Initialize query with proper scaling
        nn.init.normal_(self.query, std=math.sqrt(1.0 / dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        
        # Self-attention with learnable query
        pooled, attn_weights = self.attn(q, x, x)  # pooled: (B, 1, D)
        return pooled.squeeze(1)  # (B, D)

class SimpleAttentionPool(nn.Module):
    """
    Simplified attention pooling with single linear layer.
    More lightweight alternative to multi-head attention.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x)  # (B, L, 1)
        weights = F.softmax(scores, dim=1)  # (B, L, 1)
        weights = self.dropout(weights)
        
        pooled = torch.sum(x * weights, dim=1)  # (B, D)
        return pooled

class GatedAttentionPool(nn.Module):
    """
    Gated attention pooling with learnable gate mechanism.
    Combines attention with gating for more expressive pooling.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Linear(dim, 1)
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x)  # (B, L, 1)
        weights = F.softmax(scores, dim=1)  # (B, L, 1)
        weights = self.dropout(weights)
        
        gates = torch.sigmoid(self.gate(x))  # (B, L, D)
        
        gated_x = x * gates  # (B, L, D)
        pooled = torch.sum(gated_x * weights, dim=1)  # (B, D)
        return pooled

class SelfAttentionPool(nn.Module):
    """
    Self-attention pooling that learns to attend to important patches.
    Uses mean of attended representations as final pooling.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.self_attn(x, x, x)  # (B, L, D)
        attended = self.norm(attended + x)  
        
        pooled = attended.mean(dim=1)  # (B, D)
        return pooled

class WeightedAttentionPool(nn.Module):
    """
    Weighted attention pooling with learnable importance weights.
    Combines multiple attention mechanisms for robust pooling.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.combination_weights = nn.Parameter(torch.ones(2))
        
        self.norm = nn.LayerNorm(dim)
        
        nn.init.normal_(self.query, std=math.sqrt(1.0 / dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        query_pooled, _ = self.query_attn(q, x, x)  # (B, 1, D)
        query_pooled = query_pooled.squeeze(1)  # (B, D)
        
        self_attended, _ = self.self_attn(x, x, x)  # (B, L, D)
        self_pooled = self_attended.mean(dim=1)  # (B, D)
        
        weights = F.softmax(self.combination_weights, dim=0)
        combined = weights[0] * query_pooled + weights[1] * self_pooled
        
        return self.norm(combined)


def get_pooling(name: str, dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create pooling layers.
    
    Args:
        name: Pooling type name
        dim: Feature dimension
        **kwargs: Additional arguments (num_heads, dropout, etc.)
    
    Returns:
        Pooling module
    """
    name = name.lower()
    
    if name in ('mean', 'avg'):
        return MeanPool()
    elif name == 'max':
        return MaxPool()
    elif name in ('att', 'attention', 'attn'):
        return AttentionPool(
            dim=dim,
            num_heads=kwargs.get('num_heads', 4),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name in ('simple_att', 'simple_attention'):
        return SimpleAttentionPool(
            dim=dim,
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name in ('gated_att', 'gated_attention'):
        return GatedAttentionPool(
            dim=dim,
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name in ('self_att', 'self_attention'):
        return SelfAttentionPool(
            dim=dim,
            num_heads=kwargs.get('num_heads', 4),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name in ('weighted_att', 'weighted_attention'):
        return WeightedAttentionPool(
            dim=dim,
            num_heads=kwargs.get('num_heads', 4),
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown pooling type '{name}'. Available types: "
                        f"mean, max, attention, simple_attention, gated_attention, "
                        f"self_attention, weighted_attention")

# Utility function for testing pooling modules
def test_pooling_module(pooling_module, batch_size=4, seq_len=64, dim=256):
    """Test a pooling module with sample data."""
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test forward pass
    pooling_module.eval()
    with torch.no_grad():
        output = pooling_module(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in pooling_module.parameters())}")
    
    return output
