"""
MtsCID Normalization Layers

This module contains normalization layers extracted from the MtsCID model.
These layers are model-agnostic and can be reused by other time series models.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) for time series.
    
    RevIN normalizes the input time series by removing the mean and scaling by the
    standard deviation, then applies an optional affine transformation. The normalization
    can be reversed to recover the original scale, which is useful for time series
    forecasting and anomaly detection.
    
    Reference:
        Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting
        against Distribution Shift", ICLR 2022.
    
    Args:
        num_features: Number of features (channels) in the input
        eps: Small constant for numerical stability (default: 1e-5)
        affine: If True, applies learnable affine transformation (default: True)
        device: Device ID for CUDA. If -1 or device not available, uses CPU (default: -1)
        
    Shape:
        - Input: (batch_size, seq_len, num_features)
        - Output: (batch_size, seq_len, num_features)
        
    Example:
        >>> revin = RevIN(num_features=7, affine=True)
        >>> x = torch.randn(32, 96, 7)
        >>> # Normalize
        >>> x_norm = revin(x, mode='norm')
        >>> # ... model processing ...
        >>> # Denormalize
        >>> x_denorm = revin(x_norm, mode='denorm')
    """
    
    def __init__(self, num_features: int, eps=1e-5, affine=True, device=-1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.device = torch.device(
            f'cuda:{device}' if (torch.cuda.is_available() and device > 0) else 'cpu'
        )
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        Apply RevIN normalization or denormalization.
        
        Args:
            x: Input tensor (batch_size, seq_len, num_features)
            mode: 'norm' for normalization, 'denorm' for denormalization
            
        Returns:
            Normalized or denormalized tensor
            
        Raises:
            NotImplementedError: If mode is not 'norm' or 'denorm'
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode '{mode}' not supported. Use 'norm' or 'denorm'.")
        return x

    def _init_params(self):
        """Initialize affine transformation parameters."""
        self.affine_weight = torch.ones(self.num_features, device=self.device)
        self.affine_bias = torch.zeros(self.num_features, device=self.device)

    def _get_statistics(self, x):
        """
        Compute and store mean and standard deviation of the input.
        
        Statistics are computed over all dimensions except batch and features.
        """
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        """
        Normalize the input using stored statistics.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        Reverse the normalization using stored statistics.
        
        Args:
            x: Normalized tensor
            
        Returns:
            Denormalized tensor
        """
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


