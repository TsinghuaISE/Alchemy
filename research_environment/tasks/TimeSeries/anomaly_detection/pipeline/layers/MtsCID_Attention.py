"""
MtsCID Attention Mechanisms

This module contains attention mechanism components extracted from the MtsCID model.
These components are model-agnostic and can be reused by other Transformer-based models.
"""

import math
import torch
import torch.nn as nn

from .MtsCID_Utils import complex_einsum, complex_softmax, complex_dropout


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for Transformer models.
    
    Generates fixed positional encodings using sine and cosine functions of different
    frequencies, as described in "Attention is All You Need" (Vaswani et al., 2017).
    
    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length (default: 5000)
        
    Shape:
        - Input: (batch_size, seq_len, *)
        - Output: (1, seq_len, d_model)
        
    Example:
        >>> pos_emb = PositionalEmbedding(d_model=512)
        >>> x = torch.randn(32, 100, 512)
        >>> pe = pos_emb(x)  # (1, 100, 512)
        >>> x = x + pe
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = torch.zeros((max_len, d_model), dtype=torch.float)
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()
        
        # Apply sine to even indices
        self.pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        
        # Apply cosine to odd indices
        if d_model % 2 == 0:
            self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        else:
            # Handle odd d_model by truncating the last cosine value
            self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))[:, :-1]
        
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        """
        Get positional embeddings for the input sequence.
        
        Args:
            x: Input tensor (batch_size, seq_len, *)
            
        Returns:
            Positional embeddings (1, seq_len, d_model)
        """
        return self.pe[:, :x.size(1)]


class Attention(nn.Module):
    """
    Scaled dot-product attention with support for complex-valued tensors.
    
    Computes attention weights using scaled dot-product and applies them to values.
    Supports both real and complex-valued inputs through the complex_* utility functions.
    
    Args:
        window_size: Size of the attention window
        mask_flag: Whether to apply attention mask (default: False)
        scale: Custom scaling factor. If None, uses 1/sqrt(d_k) (default: None)
        dropout: Dropout probability (default: 0.0)
        
    Shape:
        - queries: (batch_size, seq_len, n_heads, d_k)
        - keys: (batch_size, seq_len, n_heads, d_k)
        - values: (batch_size, seq_len, n_heads, d_v)
        - Output: ((batch_size, seq_len, n_heads, d_v), (batch_size, seq_len, n_heads, seq_len))
        
    Example:
        >>> attn = Attention(window_size=100, dropout=0.1)
        >>> q = k = v = torch.randn(32, 100, 8, 64)
        >>> output, attn_weights = attn(q, k, v)
    """
    
    def __init__(self, window_size, mask_flag=False, scale=None, dropout=0.0):
        super().__init__()
        self.window_size = window_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            queries: Query tensor (batch_size, seq_len, n_heads, d_k)
            keys: Key tensor (batch_size, seq_len, n_heads, d_k)
            values: Value tensor (batch_size, seq_len, n_heads, d_v)
            attn_mask: Optional attention mask (not currently used)
            
        Returns:
            Tuple of (output, attention_weights)
            - output: (batch_size, seq_len, n_heads, d_v)
            - attention_weights: (batch_size, seq_len, n_heads, seq_len)
        """
        N, L, Head, C = queries.shape
        scale = self.scale if self.scale is not None else 1.0 / math.sqrt(C)
        
        # Compute attention scores
        attn_scores = complex_einsum('nlhd,nshd->nhls', queries, keys)
        
        # Apply softmax and dropout
        attn_weights = complex_dropout(
            self.dropout,
            complex_softmax(scale * attn_scores, dim=-1)
        )
        
        # Apply attention to values
        updated_values = complex_einsum('nhls,nshd->nlhd', attn_weights, values)
        
        # Return output and averaged attention weights
        return updated_values.contiguous(), attn_weights.permute(0, 2, 1, 3).mean(dim=-2)


class AttentionLayer(nn.Module):
    """
    Complete attention layer with query, key, value projections.
    
    This layer wraps the Attention module and adds linear projections for queries,
    keys, and values. It also includes positional embeddings (though not used in forward).
    
    Args:
        w_size: Window size for attention
        d_model: Model dimension
        n_heads: Number of attention heads
        d_keys: Dimension of keys (default: d_model // n_heads)
        d_values: Dimension of values (default: d_model // n_heads)
        mask_flag: Whether to use attention mask (default: False)
        scale: Custom attention scale (default: None)
        dropout: Dropout probability (default: 0.0)
        
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: ((batch_size, seq_len, d_model), (batch_size, seq_len, n_heads, seq_len))
        
    Example:
        >>> attn_layer = AttentionLayer(w_size=100, d_model=512, n_heads=8)
        >>> x = torch.randn(32, 100, 512)
        >>> output, attn_weights = attn_layer(x)
    """
    
    def __init__(
        self,
        w_size,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        mask_flag=False,
        scale=None,
        dropout=0.0
    ):
        super().__init__()
        # Adjust n_heads if d_model is not divisible
        n_heads = n_heads if (d_model % n_heads) == 0 else 1
        z = d_model % n_heads if (d_model // n_heads) == 0 else (d_model // n_heads)
        
        self.d_keys = d_keys if d_keys is not None else z
        self.d_values = d_values if d_values is not None else z
        self.n_heads = n_heads
        self.d_model = d_model

        # Positional embedding (available but not used in forward)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.d_values)
        
        # Output projection (identity in this implementation)
        self.out_proj = lambda x: x
        
        # Attention module
        self.attn = Attention(
            window_size=w_size,
            mask_flag=mask_flag,
            scale=scale,
            dropout=dropout
        )

    def forward(self, input_data):
        """
        Apply attention layer.
        
        Args:
            input_data: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Tuple of (output, attention_weights)
            - output: (batch_size, seq_len, d_model)
            - attention_weights: (batch_size, seq_len, n_heads, seq_len)
        """
        N, L, _ = input_data.shape
        
        # Reshape for multi-head attention
        # Note: In this implementation, Q, K, V are just reshaped from input
        # without using the linear projections W_Q, W_K, W_V
        Q = input_data.contiguous().view(N, L, self.n_heads, -1)
        K = input_data.contiguous().view(N, L, self.n_heads, -1)
        V = input_data.contiguous().view(N, L, self.n_heads, -1)
        
        # Apply attention
        updated_V, attn = self.attn(Q, K, V)
        
        # Reshape and project output
        out = self.out_proj(updated_V.view(N, L, -1))
        
        return out, attn

