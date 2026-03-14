import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import math
import numpy as np

# ===== Inlined Components =====
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding_wo_temporal(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_wo_temporal, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        return self.dropout(x)

# ===== Exponential Smoothing Attention =====
class ExponentialSmoothingAttention(nn.Module):
    """
    Exponential Smoothing Attention mechanism with O(L log L) complexity.
    Implements Algorithm 1 using FFT-based cross-correlation.
    """
    def __init__(self, d_model, n_heads):
        super(ExponentialSmoothingAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Learnable smoothing parameters per head
        self.alpha = nn.Parameter(torch.ones(n_heads) * 0.5)
        # Initial state per head
        self.v0 = nn.Parameter(torch.randn(n_heads, self.d_head))
        
    def forward(self, V):
        """
        V: [B, L, d_model]
        Returns: [B, L, d_model]
        """
        B, L, _ = V.shape
        
        # Reshape to multi-head: [B, L, n_heads, d_head]
        V = V.view(B, L, self.n_heads, self.d_head)
        
        # Clamp alpha to (0, 1)
        alpha = torch.sigmoid(self.alpha)  # [n_heads]
        
        # Efficient FFT-based implementation using vectorized operations
        # Construct attention weights: alpha * (1-alpha)^j for j=0..L-1
        j_range = torch.arange(L, device=V.device, dtype=torch.float32)  # [L]
        # [n_heads, L]
        weights = alpha.unsqueeze(1) * torch.pow(1 - alpha.unsqueeze(1), j_range.unsqueeze(0))
        weights = weights.flip(1)  # Reverse for convolution
        
        # Prepend v0 to V: [B, L+1, n_heads, d_head]
        v0_expanded = self.v0.unsqueeze(0).unsqueeze(0).expand(B, 1, self.n_heads, self.d_head)
        V_extended = torch.cat([v0_expanded, V], dim=1)
        
        # Pad to next power of 2 for efficient FFT
        fft_size = 2 ** math.ceil(math.log2(L + 1 + L - 1))
        
        # Transpose for batched FFT: [B, n_heads, d_head, L+1]
        V_extended = V_extended.permute(0, 2, 3, 1)
        
        # FFT of weights: [n_heads, fft_size]
        weights_fft = torch.fft.rfft(weights, n=fft_size, dim=1)
        
        # FFT of values: [B, n_heads, d_head, fft_size//2+1]
        V_fft = torch.fft.rfft(V_extended, n=fft_size, dim=3)
        
        # Cross-correlation in frequency domain: [B, n_heads, d_head, fft_size//2+1]
        conv_fft = V_fft * weights_fft.unsqueeze(0).unsqueeze(2)
        
        # IFFT: [B, n_heads, d_head, fft_size]
        conv_result = torch.fft.irfft(conv_fft, n=fft_size, dim=3)
        
        # Extract valid region: [B, n_heads, d_head, L]
        result = conv_result[:, :, :, L-1:2*L-1]
        
        # Reshape back: [B, L, n_heads, d_head] -> [B, L, d_model]
        output = result.permute(0, 3, 1, 2).reshape(B, L, self.d_model)
        
        return output

class MultiHeadExponentialSmoothingAttention(nn.Module):
    """
    Multi-Head Exponential Smoothing Attention for growth extraction.
    """
    def __init__(self, d_model, n_heads):
        super(MultiHeadExponentialSmoothingAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.input_projection = nn.Linear(d_model, d_model)
        self.esa = ExponentialSmoothingAttention(d_model, n_heads)
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, Z):
        """
        Z: [B, L, d_model]
        Returns: Growth representation [B, L, d_model]
        """
        B, L, _ = Z.shape
        
        # Project input
        Z_proj = self.input_projection(Z)
        
        # Compute successive differences using vectorized operations
        # Prepend zeros as initial state for difference
        Z_shifted = torch.cat([torch.zeros(B, 1, self.d_model, device=Z.device), Z_proj[:, :-1, :]], dim=1)
        Z_diff = Z_proj - Z_shifted
        
        # Apply ESA to differences
        B_out = self.esa(Z_diff)
        
        # Output projection
        B_out = self.output_projection(B_out)
        
        return B_out

# ===== Frequency Attention =====
class FrequencyAttention(nn.Module):
    """
    Frequency Attention mechanism using DFT for seasonal pattern extraction.
    """
    def __init__(self, d_model, top_k=5):
        super(FrequencyAttention, self).__init__()
        self.d_model = d_model
        self.top_k = top_k
        
    def forward(self, Z, target_len=None):
        """
        Z: [B, L, d_model]
        target_len: If provided, extrapolate to this length
        Returns: Seasonal representation [B, target_len or L, d_model]
        """
        B, L, D = Z.shape
        
        if target_len is None:
            target_len = L
        
        # Apply DFT along temporal dimension
        Z_fft = torch.fft.rfft(Z, dim=1)  # [B, F, d_model] where F = L//2 + 1
        
        # Extract amplitude and phase
        amplitude = torch.abs(Z_fft)  # [B, F, d_model]
        phase = torch.angle(Z_fft)  # [B, F, d_model]
        
        # For each feature dimension, select top-K frequencies
        F = Z_fft.shape[1]
        
        # Exclude DC component (index 0) and only consider positive frequencies
        if F > 2:
            amplitude_valid = amplitude[:, 1:, :]  # [B, F-1, d_model]
            
            # Select top-K frequencies per dimension
            top_k = min(self.top_k, F - 1)
            topk_values, topk_indices = torch.topk(amplitude_valid, top_k, dim=1)  # [B, top_k, d_model]
            
            # Adjust indices (add 1 to account for skipped DC)
            topk_indices = topk_indices + 1
        else:
            # If sequence too short, use all available frequencies
            top_k = F - 1 if F > 1 else 0
            if top_k > 0:
                topk_indices = torch.arange(1, F, device=Z.device).unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
                topk_values = amplitude[:, 1:, :]
            else:
                # Return zeros if no valid frequencies
                return torch.zeros(B, target_len, D, device=Z.device)
        
        # Reconstruct seasonal pattern using vectorized operations
        # Create time vector: [target_len]
        time_steps = torch.arange(target_len, device=Z.device, dtype=torch.float32)
        
        # Initialize output
        S = torch.zeros(B, target_len, D, device=Z.device)
        
        # Vectorized reconstruction for all top-K frequencies
        # [B, top_k, D] -> [B, D, top_k]
        topk_indices_t = topk_indices.permute(0, 2, 1)
        
        # Gather amplitudes and phases for selected frequencies
        # [B, top_k, D] -> [B, D, top_k]
        selected_amps = torch.gather(amplitude, 1, topk_indices).permute(0, 2, 1)
        selected_phases = torch.gather(phase, 1, topk_indices).permute(0, 2, 1)
        
        # Compute frequency values: 2*pi*freq_idx/L
        # [B, D, top_k]
        freq_vals = 2 * math.pi * topk_indices_t.float() / L
        
        # Compute cosine components: [B, D, top_k, target_len]
        # time_steps: [target_len] -> [1, 1, 1, target_len]
        # freq_vals: [B, D, top_k] -> [B, D, top_k, 1]
        # selected_phases: [B, D, top_k] -> [B, D, top_k, 1]
        cos_args = freq_vals.unsqueeze(-1) * time_steps.view(1, 1, 1, -1) + selected_phases.unsqueeze(-1)
        cos_components = torch.cos(cos_args)  # [B, D, top_k, target_len]
        
        # Weight by amplitudes and sum: [B, D, top_k, target_len] * [B, D, top_k, 1]
        weighted_components = cos_components * selected_amps.unsqueeze(-1)
        
        # Sum over top_k dimension: [B, D, target_len]
        S = weighted_components.sum(dim=2).permute(0, 2, 1)  # [B, target_len, D]
        
        return S

# ===== Encoder Layer =====
class ETSformerEncoderLayer(nn.Module):
    """
    ETSformer encoder layer with progressive decomposition.
    """
    def __init__(self, d_model, n_heads, d_ff, top_k, dropout, activation='gelu'):
        super(ETSformerEncoderLayer, self).__init__()
        
        self.fa = FrequencyAttention(d_model, top_k)
        self.mh_esa = MultiHeadExponentialSmoothingAttention(d_model, n_heads)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Sigmoid(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Z):
        """
        Z: Residual [B, L, d_model]
        Returns: (Z_new, B, S)
        """
        # Extract seasonal
        S = self.fa(Z)
        
        # Remove seasonal from residual
        Z = Z - S
        
        # Extract growth
        B = self.mh_esa(Z)
        
        # Remove growth from residual
        Z = self.norm1(Z - B)
        
        # Feed-forward
        Z_ff = self.ff(Z)
        Z = self.norm2(Z + self.dropout(Z_ff))
        
        return Z, B, S

# ===== Encoder =====
class ETSformerEncoder(nn.Module):
    """
    ETSformer encoder with stacked layers.
    """
    def __init__(self, layers):
        super(ETSformerEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, Z):
        """
        Z: [B, L, d_model]
        Returns: (Z_final, B_list, S_list)
        """
        B_list = []
        S_list = []
        
        for layer in self.layers:
            Z, B, S = layer(Z)
            B_list.append(B)
            S_list.append(S)
        
        return Z, B_list, S_list

# ===== Main Model =====
class Model(nn.Module):
    """
    ETSformer for time series classification.
    Adapted from forecasting to classification by using encoder-only architecture.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        
        # Hyperparameters with defaults
        self.top_k = getattr(configs, 'top_k', 5)
        self.num_class = getattr(configs, 'num_class', 2)
        
        # Input embedding (no temporal features)
        self.enc_embedding = DataEmbedding_wo_temporal(
            configs.enc_in, 
            configs.d_model, 
            configs.dropout
        )
        
        # Encoder
        encoder_layers = [
            ETSformerEncoderLayer(
                configs.d_model,
                configs.n_heads,
                configs.d_ff,
                self.top_k,
                configs.dropout,
                configs.activation
            ) for _ in range(configs.e_layers)
        ]
        self.encoder = ETSformerEncoder(encoder_layers)
        
        # Classification head
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, self.num_class)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, num_class]