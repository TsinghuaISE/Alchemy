"""
MtsCID Loss Functions

This module contains loss functions extracted from the MtsCID model.
These loss functions are model-agnostic and can be reused by other models.
"""

import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    """
    Entropy loss for regularization.
    
    Computes the entropy of the input distribution and returns the negative entropy
    as a loss. This encourages the model to produce more confident (less uniform)
    distributions.
    
    Args:
        eps: Small constant for numerical stability (default: 1e-12)
        
    Shape:
        - Input: (*, N) where * means any number of dimensions and N is the number of classes
        - Output: scalar
        
    Example:
        >>> loss_fn = EntropyLoss()
        >>> # Attention weights from softmax
        >>> attn = torch.softmax(torch.randn(4, 10, 10), dim=-1)
        >>> loss = loss_fn(attn)
    """
    
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        Compute entropy loss.
        
        Args:
            x: Input tensor (typically probability distributions)
            
        Returns:
            Scalar entropy loss
        """
        loss = -1 * x * torch.log(x + self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss


class GatheringLoss(nn.Module):
    """
    Gathering loss for memory-based anomaly detection.
    
    This loss computes the distance between queries and their nearest memory items
    in the frequency domain. It's used to encourage the model to learn compact
    representations that can be well-represented by a set of memory prototypes.
    
    Args:
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed
        memto_framework: If True, uses the MEMTO framework where memory items are
            shared across the batch. If False, each sample has its own memory items.
            
    Shape:
        - queries: (batch_size, seq_len, features)
        - items: (num_items, features) if memto_framework=True, else (batch_size, num_items, features)
        - Output: (batch_size, seq_len) if reduction='none', else scalar
        
    Example:
        >>> loss_fn = GatheringLoss(reduction='none', memto_framework=True)
        >>> queries = torch.randn(4, 100, 64)  # batch, seq_len, features
        >>> memory_items = torch.randn(100, 64)  # num_items, features
        >>> loss = loss_fn(queries, memory_items)  # (4, 100)
    """
    
    def __init__(self, reduction='none', memto_framework=True):
        super().__init__()
        self.reduction = reduction
        self.memto_framework = memto_framework

    def forward(self, queries, items):
        """
        Compute gathering loss.
        
        Args:
            queries: Query representations (batch_size, seq_len, features)
            items: Memory items (num_items, features) or (batch_size, num_items, features)
            
        Returns:
            Gathering loss tensor
        """
        batch_size = queries.size(0)
        loss_mse = torch.nn.MSELoss(reduction=self.reduction)

        # Transform to frequency domain and normalize
        f = torch.fft.rfft(queries, dim=-2).permute(0, 2, 1)
        i_query_angle = torch.angle(f)
        unit_magnitude_queries = torch.fft.irfft(torch.exp(-1j * i_query_angle)).permute(0, 2, 1)

        if self.memto_framework:
            # Shared memory items across batch
            score = torch.einsum('bij,kj->bik', unit_magnitude_queries, items)
            _, indices = torch.topk(score, 1, dim=-1)
            step_basis = torch.gather(
                items.unsqueeze(0).repeat(batch_size, 1, 1),
                1,
                indices.expand(-1, -1, items.size(-1))
            )
            gathering_loss = loss_mse(queries, step_basis)
        else:
            # Per-sample memory items
            score = torch.einsum('bij,bkj->bik', unit_magnitude_queries, items)
            _, indices = torch.topk(score, 1, dim=-1)
            C = torch.gather(items, 1, indices.expand(-1, -1, items.size(-1)))
            gathering_loss = loss_mse(queries, C)

        if self.reduction != 'none':
            return gathering_loss

        # Sum over features and reshape
        gathering_loss = torch.sum(gathering_loss, dim=-1)
        gathering_loss = gathering_loss.contiguous().view(batch_size, -1)
        return gathering_loss


