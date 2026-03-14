import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ===== Inlined Components =====

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

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
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, C]
        # do patching
        n_vars = x.shape[2]
        x = self.padding_patch_layer(x.permute(0, 2, 1))  # [B, C, L] -> [B, C, L+padding]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, num_patch, patch_len]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # [B*C, num_patch, patch_len]
        
        # Input encoding
        x = self.value_embedding(x)  # [B*C, num_patch, d_model]
        
        # Positional embedding
        x = x + self.position_embedding(x)  # [B*C, num_patch, d_model]
        
        return self.dropout(x), n_vars

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [B*C, num_patch, d_model]
        x = self.flatten(x)  # [B*C, num_patch * d_model]
        x = self.linear(x)  # [B*C, target_window]
        x = self.dropout(x)
        x = x.reshape(-1, self.n_vars, x.shape[-1])  # [B, C, target_window]
        x = x.permute(0, 2, 1)  # [B, target_window, C]
        return x

class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation='gelu', n_layers=1):
        super(TSTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TSTEncoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # x: [B*C, num_patch, d_model]
        for layer in self.layers:
            x = layer(x)
        return x

class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation='gelu'):
        super(TSTEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B*C, num_patch, d_model]
        
        # Multi-head self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

# ===== End Inlined Components =====

class Model(nn.Module):
    """
    PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
    Paper link: https://arxiv.org/abs/2211.14730
    
    This implementation adapts PatchTST for the imputation task by:
    1. Processing input sequences in patches
    2. Using Transformer encoder to capture dependencies
    3. Reconstructing the full sequence from patch representations
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len  # For imputation, output length equals input length
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.activation = getattr(configs, 'activation', 'gelu')
        
        # Patching parameters
        self.patch_len = patch_len
        self.stride = stride
        
        # Calculate padding to ensure seq_len can be evenly divided
        if self.seq_len % self.stride != 0:
            self.padding = self.stride - (self.seq_len % self.stride)
        else:
            self.padding = 0
        
        # Calculate number of patches
        padded_len = self.seq_len + self.padding
        self.num_patches = (padded_len - self.patch_len) // self.stride + 1
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.padding, self.dropout
        )
        
        # Transformer encoder
        self.encoder = TSTEncoder(
            self.d_model, self.n_heads, self.d_ff, self.dropout, 
            self.activation, self.e_layers
        )
        
        # Projection head to reconstruct the sequence
        self.head = FlattenHead(
            self.enc_in, 
            self.num_patches * self.d_model, 
            self.seq_len,
            head_dropout=self.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass for the model
        
        Args:
            x_enc: [B, L, C] - Input sequence
            x_mark_enc: [B, L, mark_dim] - Time features (not used in this implementation)
            x_dec: Not used for imputation
            x_mark_dec: Not used for imputation
            mask: [B, L, C] - Mask for missing values
            
        Returns:
            dec_out: [B, L, C] - Reconstructed sequence
        """
        dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return dec_out  # [B, L, D]