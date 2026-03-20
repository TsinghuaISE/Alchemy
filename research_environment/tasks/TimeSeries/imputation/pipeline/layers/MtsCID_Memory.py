"""
MtsCID Memory Modules

This module contains memory-related functions extracted from the MtsCID model.
These functions generate memory matrices for memory-augmented anomaly detection.
"""

import torch


def generate_rolling_matrix(input_matrix):
    """
    Generate a rolling matrix by shifting the input matrix along the time dimension.
    
    This function creates multiple versions of the input matrix, each shifted by
    a different amount. This is used to create temporal variations of memory items.
    
    Args:
        input_matrix: Input matrix of shape (F, L) where F is features and L is length
        
    Returns:
        Rolling matrix of shape (L, F, L) containing L shifted versions
        
    Example:
        >>> mat = torch.randn(10, 100)
        >>> rolled = generate_rolling_matrix(mat)
        >>> rolled.shape
        torch.Size([100, 10, 100])
    """
    F, L = input_matrix.size()
    output_matrix = torch.empty(L, F, L)
    
    for step in range(L):
        rolled_matrix = input_matrix.roll(shifts=step, dims=1)
        output_matrix[step] = rolled_matrix
    
    return output_matrix


def create_memory_matrix(N, L, mem_type='sinusoid', option='option1'):
    """
    Create memory matrix for memory-augmented anomaly detection.
    
    This function generates a pair of matrices (real and imaginary parts) that serve
    as memory prototypes for the model. Different initialization strategies are supported.
    
    Args:
        N: Number of memory items (typically equal to number of features)
        L: Length of each memory item (typically equal to sequence length)
        mem_type: Type of initialization:
            - 'sinusoid' or 'cosine_only': Sinusoidal patterns with different frequencies
            - 'uniform' or 'uniform_only': Uniform random initialization
            - 'orthogonal_uniform' or 'orthogonal_uniform_only': Orthogonal uniform random
            - 'normal' or 'normal_only': Normal random initialization
            - 'orthogonal_normal' or 'orthogonal_normal_only': Orthogonal normal random
            - other: Zero initialization
            Types ending with '_only' return zero imaginary part
        option: Memory organization option:
            - 'option1': Standard (N, L) matrices
            - 'option4': Apply rolling transformation
            
    Returns:
        Tuple of (init_matrix_r, init_matrix_i):
        - init_matrix_r: Real part of memory matrix
        - init_matrix_i: Imaginary part of memory matrix (or zeros for '_only' types)
        
    Shape:
        - Output: Two tensors of shape (N, L) or (L, N, L) if option='option4'
        
    Example:
        >>> # Create sinusoidal memory with 25 items of length 100
        >>> mem_r, mem_i = create_memory_matrix(25, 100, mem_type='sinusoid')
        >>> mem_r.shape, mem_i.shape
        (torch.Size([25, 100]), torch.Size([25, 100]))
        
        >>> # Create memory with only real part
        >>> mem_r, mem_i = create_memory_matrix(25, 100, mem_type='sinusoid_only')
        >>> torch.all(mem_i == 0)
        True
    """
    with torch.no_grad():
        if mem_type in ['sinusoid', 'cosine_only']:
            # Sinusoidal initialization with different frequencies
            row_indices = torch.arange(N).reshape(-1, 1)
            col_indices = torch.arange(L)
            grid = row_indices * col_indices
            init_matrix_r = torch.cos((1 / L) * 2 * torch.tensor([torch.pi]) * grid)
            init_matrix_i = torch.sin((1 / L) * 2 * torch.tensor([torch.pi]) * grid)
            
        elif mem_type in ['uniform', 'uniform_only']:
            # Uniform random initialization
            init_matrix_r = torch.rand((N, L), dtype=torch.float)
            init_matrix_i = torch.rand((N, L), dtype=torch.float)
            
        elif mem_type in ['orthogonal_uniform', 'orthogonal_uniform_only']:
            # Orthogonal uniform random initialization
            init_matrix_r = torch.nn.init.orthogonal_(torch.rand((N, L), dtype=torch.float))
            init_matrix_i = torch.nn.init.orthogonal_(torch.rand((N, L), dtype=torch.float))
            
        elif mem_type in ['normal', 'normal_only']:
            # Normal random initialization
            init_matrix_r = torch.randn((N, L), dtype=torch.float)
            init_matrix_i = torch.randn((N, L), dtype=torch.float)
            
        elif mem_type in ['orthogonal_normal', 'orthogonal_normal_only']:
            # Orthogonal normal random initialization
            init_matrix_r = torch.nn.init.orthogonal_(torch.randn((N, L), dtype=torch.float))
            init_matrix_i = torch.nn.init.orthogonal_(torch.randn((N, L), dtype=torch.float))
            
        else:
            # Default: zero initialization
            init_matrix_r = torch.cos(torch.zeros((N, L)))
            init_matrix_i = torch.sin(torch.zeros((N, L)))

        # Apply rolling transformation if requested
        if option == 'option4':
            init_matrix_r = generate_rolling_matrix(init_matrix_r)
            init_matrix_i = generate_rolling_matrix(init_matrix_i)

        # Return both real and imaginary parts, or zero imaginary for '_only' types
        if 'only' not in mem_type:
            return init_matrix_r, init_matrix_i
        return init_matrix_r, torch.zeros_like(init_matrix_r)


