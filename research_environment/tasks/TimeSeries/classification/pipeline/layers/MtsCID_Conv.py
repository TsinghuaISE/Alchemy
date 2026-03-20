"""
MtsCID Convolutional Blocks

This module contains convolutional block components extracted from the MtsCID model.
These components implement Inception-style multi-scale convolutions and can be reused
by other models.
"""

import torch
import torch.nn as nn
from einops import rearrange

from .MtsCID_Attention import AttentionLayer


class Inception_Block(nn.Module):
    """
    Inception-style convolutional block with multiple kernel sizes.
    
    This block applies 1D convolutions with different kernel sizes in parallel
    and averages their outputs. This allows the model to capture patterns at
    multiple scales simultaneously.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_list: List of kernel sizes to use (default: [1, 3, 5])
        groups: Number of groups for grouped convolution (default: 1)
        init_weight: Whether to initialize weights with Kaiming initialization (default: True)
        
    Shape:
        - Input: (batch_size, in_channels, seq_len)
        - Output: (batch_size, out_channels, seq_len)
        
    Example:
        >>> block = Inception_Block(in_channels=64, out_channels=128, kernel_list=[1, 3, 5])
        >>> x = torch.randn(32, 64, 100)
        >>> out = block(x)  # (32, 128, 100)
    """
    
    def __init__(self, in_channels, out_channels, kernel_list=None, groups=1, init_weight=True):
        super().__init__()
        kernel_list = kernel_list or [1, 3, 5]
        kernels = []
        
        for k in kernel_list:
            kernels.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    padding='same',
                    padding_mode='circular',
                    bias=False,
                    groups=groups
                )
            )
        
        self.convs = nn.ModuleList(kernels)
        
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolutional weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Apply multi-scale convolutions and average the results.
        
        Args:
            x: Input tensor (batch_size, in_channels, seq_len)
            
        Returns:
            Output tensor (batch_size, out_channels, seq_len)
        """
        res_list = [conv(x) for conv in self.convs]
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Attention_Block(nn.Module):
    """
    Inception-style attention block with multi-scale patches.
    
    This block divides the input into patches of different sizes, applies attention
    to each patch scale, and combines the results. This allows the model to capture
    temporal dependencies at multiple granularities.
    
    Args:
        w_size: Window size (sequence length)
        in_dim: Input dimension (not used in current implementation)
        d_model: Model dimension (not used in current implementation)
        patch_list: List of patch sizes to use (default: [10, 20])
        init_weight: Whether to initialize weights with Kaiming initialization (default: True)
        
    Shape:
        - Input: (batch_size, seq_len, features)
        - Output: (batch_size, seq_len, features)
        
    Example:
        >>> block = Inception_Attention_Block(w_size=100, in_dim=64, d_model=64, patch_list=[10, 20])
        >>> x = torch.randn(32, 100, 64)
        >>> out = block(x)  # (32, 100, 64)
        
    Note:
        The sequence length (w_size) must be divisible by all patch sizes in patch_list.
    """
    
    def __init__(self, w_size, in_dim, d_model, patch_list=None, init_weight=True):
        super().__init__()
        self.w_size = w_size
        self.in_dim = in_dim
        self.d_model = d_model
        self.patch_list = patch_list or [10, 20]
        
        patch_attention_layers = []
        linear_layers = []
        
        for patch_size in self.patch_list:
            patch_number = w_size // patch_size
            # Create attention layer for this patch scale
            patch_attention_layers.append(
                AttentionLayer(w_size=patch_number, d_model=patch_size, n_heads=1)
            )
            # Create linear layer to project attention outputs
            linear_layers.append(nn.Linear(patch_number, patch_size))
        
        self.patch_attention_layers = nn.ModuleList(patch_attention_layers)
        self.linear_layers = nn.ModuleList(linear_layers)
        
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize linear weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Apply multi-scale patch attention and combine results.
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            
        Returns:
            Output tensor (batch_size, seq_len, features)
        """
        B, _, _ = x.size()
        res_list = []
        
        for i, p_size in enumerate(self.patch_list):
            # Rearrange into patches: (batch * features, num_patches, patch_size)
            z = rearrange(x, 'b (w p) c  -> (b c) w p', p=p_size).contiguous()
            
            # Apply attention (returns output and attention weights)
            _, z = self.patch_attention_layers[i](z)
            
            # Project attention outputs
            z = self.linear_layers[i](z)
            
            # Rearrange back to original shape
            z = rearrange(z, '(b c) w p -> b (w p) c', b=B).contiguous()
            res_list.append(z)
        
        # Average results from all patch scales
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


