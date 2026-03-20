"""
MtsCID Utility Functions

This module contains general-purpose utility functions for MtsCID model,
including complex number operations and loss computation functions.
These functions are model-agnostic and can be reused by other models.
"""

import torch
import torch.nn as nn


# ===================== Complex Number Operations =====================

def complex_operator(net_layer, x):
    """
    Apply a neural network layer to complex-valued tensors.
    
    Args:
        net_layer: Neural network layer (or ModuleList of layers)
        x: Input tensor (real or complex)
        
    Returns:
        Output tensor after applying the layer
    """
    if not torch.is_complex(x):
        return net_layer[0](x) if isinstance(net_layer, nn.ModuleList) else net_layer(x)
    
    # Check if it's a ModuleList first
    if isinstance(net_layer, nn.ModuleList):
        if isinstance(net_layer[0], nn.LSTM):
            return torch.complex(net_layer[0](x.real)[0], net_layer[1](x.imag)[0])
        else:
            return torch.complex(net_layer[0](x.real), net_layer[1](x.imag))
    else:
        # Single layer
        if isinstance(net_layer, nn.LSTM):
            return torch.complex(net_layer(x.real)[0], net_layer(x.imag)[0])
        else:
            return torch.complex(net_layer(x.real), net_layer(x.imag))


def complex_einsum(order, x, y):
    """
    Perform Einstein summation on complex-valued tensors.
    
    Args:
        order: Einstein summation notation string
        x: First input tensor (real or complex)
        y: Second input tensor (real or complex)
        
    Returns:
        Result of Einstein summation
    """
    x_flag = True
    y_flag = True
    
    if not torch.is_complex(x):
        x_flag = False
        x = torch.complex(x, torch.zeros_like(x).to(x.device))
    
    if not torch.is_complex(y):
        y_flag = False
        y = torch.complex(y, torch.zeros_like(y).to(y.device))
    
    if x_flag or y_flag:
        return torch.complex(
            torch.einsum(order, x.real, y.real) - torch.einsum(order, x.imag, y.imag),
            torch.einsum(order, x.real, y.imag) + torch.einsum(order, x.imag, y.real),
        )
    
    return torch.einsum(order, x.real, y.real)


def complex_softmax(x, dim=-1):
    """
    Apply softmax to complex-valued tensors.
    
    Args:
        x: Input tensor (real or complex)
        dim: Dimension along which to apply softmax
        
    Returns:
        Softmax output
    """
    if not torch.is_complex(x):
        return torch.softmax(x, dim=dim)
    
    return torch.complex(
        torch.softmax(x.real, dim=dim),
        torch.softmax(x.imag, dim=dim)
    )


def complex_dropout(dropout_func, x):
    """
    Apply dropout to complex-valued tensors.
    
    Args:
        dropout_func: Dropout function/layer
        x: Input tensor (real or complex)
        
    Returns:
        Output after dropout
    """
    if not torch.is_complex(x):
        return dropout_func(x)
    
    # For complex tensors, return as-is (dropout applied separately to real/imag if needed)
    return torch.complex(x.real, x.imag)


# ===================== Loss Computation =====================

def harmonic_loss_compute(t_loss, f_loss, operator="mean"):
    """
    Combine temporal (t_loss) and frequency (f_loss) domain losses.
    
    This function implements various strategies to aggregate losses from
    temporal and frequency domains for anomaly detection.
    
    Args:
        t_loss: Temporal domain loss tensor
        f_loss: Frequency domain loss tensor
        operator: Aggregation method, one of:
            - 'normal_mean': Simple mean of temporal loss weighted by frequency loss
            - 'mean': Max of temporal loss weighted by softmax of mean frequency loss
            - 'max': Max of temporal loss weighted by softmax of max frequency loss
            - 'harmonic_mean': Harmonic mean of bidirectional weighted losses
            - 'harmonic_max': Harmonic max of bidirectional weighted losses
            
    Returns:
        Combined loss tensor
    """
    assert operator in ["normal_mean", "mean", "max", "harmonic_mean", "harmonic_max"], \
        f"Invalid operator: {operator}. Must be one of: normal_mean, mean, max, harmonic_mean, harmonic_max"

    # Compute aggregated weights
    t_wa = t_loss.mean(dim=-2, keepdim=True)  # Temporal weighted average
    f_wa = f_loss.mean(dim=-2, keepdim=True)  # Frequency weighted average
    t_wm = t_loss.max(dim=-2, keepdim=True)[0]  # Temporal weighted max
    f_wm = f_loss.max(dim=-2, keepdim=True)[0]  # Frequency weighted max

    if operator == "mean":
        # Max of temporal loss weighted by softmax of mean frequency loss
        loss = (t_loss * torch.softmax(f_wa, dim=-1)).max(dim=-1)[0]
    elif operator == "max":
        # Max of temporal loss weighted by softmax of max frequency loss
        loss = (t_loss * torch.softmax(f_wm, dim=-1)).max(dim=-1)[0]
    elif operator == "harmonic_mean":
        # Harmonic mean: average of bidirectional weighted means
        nt_loss = (t_loss * torch.softmax(f_wa, dim=-1)).mean(dim=-1)
        nf_loss = (f_loss * torch.softmax(t_wa, dim=-1)).mean(dim=-1)
        loss = (nt_loss + nf_loss) / 2
    elif operator == "harmonic_max":
        # Harmonic max: average of bidirectional weighted maxes
        nt_loss = (t_loss * torch.softmax(f_wm, dim=-1)).max(dim=-1)[0]
        nf_loss = (f_loss * torch.softmax(t_wm, dim=-1)).max(dim=-1)[0]
        loss = (nt_loss + nf_loss) / 2
    else:  # normal_mean
        # Simple element-wise product of mean temporal loss and frequency loss
        loss = t_loss.mean(dim=-1) * f_loss

    return loss

