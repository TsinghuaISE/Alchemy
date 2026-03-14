import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import ceil

class Model(nn.Module):
    """
    TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # Model parameters
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.top_k = getattr(configs, 'top_k', 5)  # Number of frequencies to select
        self.num_kernels = getattr(configs, 'num_kernels', 6)  # Number of inception kernels
        self.e_layers = configs.e_layers  # Number of TimesBlocks
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # Embedding layer
        self.enc_embedding = DataEmbedding(configs.enc_in, self.d_model, 
                                          configs.embed, configs.freq, configs.dropout)
        
        # TimesBlocks
        self.model = nn.ModuleList([
            TimesBlock(self.seq_len, self.pred_len, self.top_k, self.d_model, 
                      self.d_ff, self.num_kernels)
            for _ in range(self.e_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Projection layer
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev
        
        # Embedding: [B, T, C] -> [B, T, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # TimesBlocks with residual connections
        for i in range(self.e_layers):
            enc_out = self.model[i](enc_out)
        
        # Layer normalization
        enc_out = self.layer_norm(enc_out)
        
        # Projection: [B, T, d_model] -> [B, T, c_out]
        dec_out = self.projection(enc_out)
        
        # De-normalization
        dec_out = dec_out * stdev + means
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

class TimesBlock(nn.Module):
    """
    TimesBlock: Transform 1D time series into 2D space based on multiple periodicities
    and capture temporal 2D-variations using inception blocks
    """
    
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.d_model = d_model
        
        # Parameter-efficient inception block
        self.conv = InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels)
        
    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        B, T, N = x.shape
        
        # 1. Period discovery using FFT (Equation 1-2)
        period_list, period_weight = self.period_discovery(x)
        
        # 2. Transform to 2D and apply inception block for each period
        res = []
        for i in range(self.top_k):
            period = period_list[i]
            
            # Padding to make sequence compatible with period
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N], 
                                     device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            
            # Reshape to 2D: [B, T, N] -> [B, P, F, N] where P=period, F=length/period
            out = out.reshape(B, length // period, period, N)
            # [B, P, F, N] -> [B, N, P, F]
            out = out.permute(0, 3, 1, 2).contiguous()
            
            # Apply 2D inception block
            out = self.conv(out)
            
            # Reshape back to 1D: [B, N, P, F] -> [B, P, F, N] -> [B, T, N]
            out = out.permute(0, 2, 3, 1).contiguous()
            out = out.reshape(B, -1, N)
            
            # Truncate to original length
            out = out[:, :self.seq_len, :]
            res.append(out)
        
        # 3. Adaptive aggregation (Equation 6)
        res = torch.stack(res, dim=-1)  # [B, T, N, top_k]
        
        # Normalize period weights
        period_weight = F.softmax(period_weight, dim=1)  # [B, top_k]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, top_k]
        
        # Weighted sum
        res = torch.sum(res * period_weight, dim=-1)  # [B, T, N]
        
        # Residual connection
        res = res + x
        
        return res
    
    def period_discovery(self, x):
        """
        Discover dominant periods using FFT (Equation 1-2)
        x: [B, T, N]
        Returns:
            period_list: List of top-k period lengths
            period_weight: [B, top_k] amplitude values for each period
        """
        B, T, N = x.shape
        
        # FFT along time dimension
        xf = torch.fft.rfft(x, dim=1)  # [B, T//2+1, N]
        
        # Calculate amplitude and average across channels
        amplitude = torch.abs(xf).mean(dim=-1)  # [B, T//2+1]
        
        # Only consider frequencies from 1 to T//2 (avoid DC component and conjugates)
        amplitude[:, 0] = 0  # Remove DC component
        
        # Select top-k frequencies
        _, top_indices = torch.topk(amplitude, self.top_k, dim=1)  # [B, top_k]
        top_indices = top_indices.detach().cpu().numpy()
        
        # Get period lengths and weights for each sample in batch
        period_list = []
        period_weight_list = []
        
        for b in range(B):
            periods = []
            weights = []
            for i in range(self.top_k):
                freq_idx = top_indices[b, i]
                if freq_idx == 0:
                    freq_idx = 1  # Avoid division by zero
                period = ceil(T / freq_idx)
                periods.append(period)
                weights.append(amplitude[b, freq_idx].item())
            period_list.append(periods)
            period_weight_list.append(weights)
        
        # Use the first sample's periods for the whole batch (approximation)
        # In practice, could use different periods per sample
        period_list = period_list[0]
        
        # Convert weights to tensor
        period_weight = torch.tensor(period_weight_list, device=x.device)  # [B, top_k]
        
        return period_list, period_weight

class InceptionBlockV1(nn.Module):
    """
    Inception block for processing 2D temporal variations
    Adapted from computer vision inception architecture
    """
    
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionBlockV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        
        # Define multiple kernel sizes for multi-scale feature extraction
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2*i+1, 
                                    padding=i))
        self.kernels = nn.ModuleList(kernels)
        
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        x: [B, N, P, F] where P=period, F=frequency
        """
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        
        # Concatenate and aggregate
        res = torch.stack(res_list, dim=-1).mean(dim=-1)
        
        return res

class DataEmbedding(nn.Module):
    """
    Data embedding: combines value embedding with temporal encoding
    """
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, 
                                                    freq=freq) if embed_type != 'timeF' else \
                                 TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, 
                                                     freq=freq)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, x_mark):
        """
        x: [B, T, C]
        x_mark: [B, T, mark_dim] or None
        """
        # Value embedding
        x = self.value_embedding(x)
        
        # Add positional embedding
        x = x + self.position_embedding(x)
        
        # Add temporal embedding if available
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)
        
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    """
    Token embedding using 1D convolution
    """
    
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular',
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x):
        # x: [B, T, C]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    """
    Fixed positional embedding
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * 
                   -(np.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [B, T, d_model]
        return self.pe[:, :x.size(1)]

class TemporalEmbedding(nn.Module):
    """
    Temporal embedding for time features
    """
    
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        
        Embed = nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        # x: [B, T, time_features]
        x = x.long()
        
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    """
    Time feature embedding using linear projection
    """
    
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)
    
    def forward(self, x):
        # x: [B, T, d_inp]
        return self.embed(x)