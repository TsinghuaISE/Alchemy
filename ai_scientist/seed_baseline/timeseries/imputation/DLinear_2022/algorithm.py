import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Inlined Components =====
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
# ===== End Inlined Components =====

class Model(nn.Module):
    """
    DLinear: Decomposition Linear model for time series forecasting.
    Paper: Are Transformers Effective for Time Series Forecasting? (AAAI 2023)
    
    This model implements three variants:
    1. Vanilla Linear: Direct linear mapping from input to output
    2. DLinear: Decomposition + separate linear layers for trend and seasonal
    3. NLinear: Normalization (subtract last value) + linear + denormalization
    
    The model uses Direct Multi-step (DMS) forecasting strategy.
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Get model configuration
        self.individual = getattr(configs, 'individual', False)
        self.enc_in = configs.enc_in
        self.c_out = getattr(configs, 'c_out', configs.enc_in)
        
        # Determine which variant to use based on configs
        # Use decomposition if moving_avg is specified and > 1
        self.use_decomp = hasattr(configs, 'moving_avg') and configs.moving_avg > 1
        # Use normalization if use_norm is specified
        self.use_norm = getattr(configs, 'use_norm', False)
        
        if self.use_decomp:
            # DLinear variant: decomposition + separate linear layers
            self.decomp = series_decomp(configs.moving_avg)
            
            if self.individual:
                # Individual linear layers for each channel
                self.Linear_Seasonal = nn.ModuleList([
                    nn.Linear(self.seq_len, self.pred_len) 
                    for _ in range(self.enc_in)
                ])
                self.Linear_Trend = nn.ModuleList([
                    nn.Linear(self.seq_len, self.pred_len) 
                    for _ in range(self.enc_in)
                ])
            else:
                # Shared linear layers across channels
                self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
                self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        else:
            # Vanilla Linear or NLinear variant
            if self.individual:
                self.Linear = nn.ModuleList([
                    nn.Linear(self.seq_len, self.pred_len) 
                    for _ in range(self.enc_in)
                ])
            else:
                self.Linear = nn.Linear(self.seq_len, self.pred_len)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass for short-term forecasting
        Args:
            x_enc: [B, seq_len, enc_in] - encoder input
            x_mark_enc: [B, seq_len, mark_dim] - encoder time features (not used)
            x_dec: [B, label_len+pred_len, dec_in] - decoder input (not used)
            x_mark_dec: [B, label_len+pred_len, mark_dim] - decoder time features (not used)
            mask: attention mask (not used)
        Returns:
            dec_out: [B, pred_len, c_out]
        """
        # DLinear only uses encoder input, ignoring decoder input and time features
        # This aligns with the paper's premise that simple linear models don't need
        # complex positional encodings or temporal embeddings
        
        # Normalization for stability (optional, can be controlled by config)
        # For other tasks, use the forecast directly
        dec_out = self.forecast(x_enc)
        
        # Return only the prediction part (last pred_len steps)
        return dec_out[:, -self.pred_len:, :]