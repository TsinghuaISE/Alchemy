import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class IEBlock(nn.Module):
    """
    Information Exchange Block with bottleneck design.
    Performs temporal projection, channel projection, and output projection.
    
    Args:
        input_dim (int): Temporal dimension H (input height)
        num_channels (int): Channel dimension W (input width)
        hidden_dim (int): Bottleneck dimension F' (F' << input_dim, output_dim)
        output_dim (int): Output feature dimension F
        dropout (float): Dropout rate
    """
    def __init__(self, input_dim, num_channels, hidden_dim, output_dim, dropout=0.1):
        super(IEBlock, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Temporal projection: R^H -> R^F' (applied column-wise with weight sharing)
        self.temporal_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Channel projection: R^W -> R^W (applied row-wise with weight sharing)
        self.channel_projection = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output projection: R^F' -> R^F (applied column-wise with weight sharing)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, H, W] where H is temporal dim, W is channel dim
        Returns:
            Output tensor of shape [B, F, W]
        """
        B, H, W = x.shape
        
        # Temporal projection: apply MLP to each column (along temporal dimension)
        # x: [B, H, W] -> [B, W, H] -> apply MLP -> [B, W, F'] -> [B, F', W]
        x_t = x.permute(0, 2, 1)  # [B, W, H]
        x_t = self.temporal_projection(x_t)  # [B, W, F']
        x_t = x_t.permute(0, 2, 1)  # [B, F', W]
        
        # Channel projection: apply MLP to each row (along channel dimension)
        # x_t: [B, F', W] -> [B, F', W]
        x_c = self.channel_projection(x_t)  # [B, F', W]
        
        # Output projection: apply MLP to each column
        # x_c: [B, F', W] -> [B, W, F'] -> apply MLP -> [B, W, F] -> [B, F, W]
        x_o = x_c.permute(0, 2, 1)  # [B, W, F']
        x_o = self.output_projection(x_o)  # [B, W, F]
        x_o = x_o.permute(0, 2, 1)  # [B, F, W]
        
        # Apply layer normalization
        x_o = x_o.permute(0, 2, 1)  # [B, W, F]
        x_o = self.layer_norm(x_o)
        x_o = x_o.permute(0, 2, 1)  # [B, F, W]
        
        return x_o

class ContinuousSampling(nn.Module):
    """
    Continuous sampling: transform sequence of length T to C x T/C matrix.
    Each column contains C consecutive tokens.
    """
    def __init__(self, chunk_size):
        super(ContinuousSampling, self).__init__()
        self.chunk_size = chunk_size
        
    def forward(self, x):
        """
        Args:
            x: [B, T, N] where T is sequence length, N is number of variables
        Returns:
            [B, C, T/C, N] where C is chunk_size
        """
        B, T, N = x.shape
        num_chunks = T // self.chunk_size
        
        # Reshape to [B, num_chunks, C, N] then permute to [B, C, num_chunks, N]
        x = x[:, :num_chunks * self.chunk_size, :]  # Ensure divisible
        x = x.reshape(B, num_chunks, self.chunk_size, N)
        x = x.permute(0, 2, 1, 3)  # [B, C, num_chunks, N]
        
        return x

class IntervalSampling(nn.Module):
    """
    Interval sampling: transform sequence of length T to C x T/C matrix.
    Each column contains C tokens with fixed interval.
    """
    def __init__(self, chunk_size):
        super(IntervalSampling, self).__init__()
        self.chunk_size = chunk_size
        
    def forward(self, x):
        """
        Args:
            x: [B, T, N] where T is sequence length, N is number of variables
        Returns:
            [B, C, T/C, N] where C is chunk_size
        """
        B, T, N = x.shape
        num_chunks = T // self.chunk_size
        
        # Sample with interval
        x = x[:, :num_chunks * self.chunk_size, :]  # Ensure divisible
        x = x.reshape(B, num_chunks, self.chunk_size, N)
        x = x.permute(0, 2, 1, 3)  # [B, C, num_chunks, N]
        
        return x

class Model(nn.Module):
    """
    LightTS: Light-weight Time Series forecasting model with pure MLP architecture.
    
    Paper: Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # Hyperparameters
        self.chunk_size = min(configs.seq_len, 16)  # C in paper, ensure <= seq_len
        
        # Ensure seq_len is divisible by chunk_size for proper sampling
        if self.seq_len % self.chunk_size != 0:
            # Pad seq_len to be divisible by chunk_size
            self.padded_seq_len = self.seq_len + (self.chunk_size - self.seq_len % self.chunk_size)
        else:
            self.padded_seq_len = self.seq_len
            
        self.num_chunks = self.padded_seq_len // self.chunk_size
        
        # Feature dimensions
        self.hidden_dim = 32  # F' in paper (bottleneck dimension)
        self.feature_dim = 64  # F in paper (output feature dimension)
        
        # Dropout rate
        self.dropout = configs.dropout
        
        # Part I: Sampling and feature extraction for each time series
        self.continuous_sampling = ContinuousSampling(self.chunk_size)
        self.interval_sampling = IntervalSampling(self.chunk_size)
        
        # IEBlock-A for continuous sampling
        self.ieblock_a = IEBlock(
            input_dim=self.chunk_size,
            num_channels=self.num_chunks,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            dropout=self.dropout
        )
        
        # IEBlock-B for interval sampling
        self.ieblock_b = IEBlock(
            input_dim=self.chunk_size,
            num_channels=self.num_chunks,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            dropout=self.dropout
        )
        
        # Down-projection from [F, T/C] to [F]
        self.down_projection = nn.Linear(self.num_chunks, 1)
        
        # Part II: Combine features and learn inter-variable correlations
        # Input: [2F, N], Output: [pred_len, N]
        self.ieblock_c = IEBlock(
            input_dim=2 * self.feature_dim,
            num_channels=self.enc_in,
            hidden_dim=self.hidden_dim,
            output_dim=self.pred_len,
            dropout=self.dropout
        )
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Args:
            x_enc: [B, seq_len, enc_in] - input sequence
            x_mark_enc: [B, seq_len, mark_dim] - temporal features (can be None)
            x_dec: [B, label_len + pred_len, dec_in] - decoder input (not used in LightTS)
            x_mark_dec: [B, label_len + pred_len, mark_dim] - decoder temporal features (not used)
            mask: attention mask (not used in LightTS)
        Returns:
            [B, pred_len, enc_in] - predictions
        """
        B, T, N = x_enc.shape
        
        # Normalization (Reversible Instance Normalization)
        means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, N]
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev
        
        # Pad sequence if necessary
        if T < self.padded_seq_len:
            padding = torch.zeros((B, self.padded_seq_len - T, N), device=x_enc.device)
            x_enc = torch.cat([x_enc, padding], dim=1)
        
        # Part I: Process each variable independently
        features_list = []
        
        for i in range(N):
            # Extract single variable: [B, T]
            x_var = x_enc[:, :, i]  # [B, padded_seq_len]
            x_var = x_var.unsqueeze(-1)  # [B, padded_seq_len, 1]
            
            # Continuous sampling: [B, C, T/C, 1]
            x_con = self.continuous_sampling(x_var)
            x_con = x_con.squeeze(-1)  # [B, C, T/C]
            
            # Apply IEBlock-A: [B, C, T/C] -> [B, F, T/C]
            feat_con = self.ieblock_a(x_con)
            
            # Down-project: [B, F, T/C] -> [B, F]
            feat_con = self.down_projection(feat_con).squeeze(-1)  # [B, F]
            
            # Interval sampling: [B, C, T/C, 1]
            x_int = self.interval_sampling(x_var)
            x_int = x_int.squeeze(-1)  # [B, C, T/C]
            
            # Apply IEBlock-B: [B, C, T/C] -> [B, F, T/C]
            feat_int = self.ieblock_b(x_int)
            
            # Down-project: [B, F, T/C] -> [B, F]
            feat_int = self.down_projection(feat_int).squeeze(-1)  # [B, F]
            
            # Concatenate continuous and interval features: [B, 2F]
            feat_var = torch.cat([feat_con, feat_int], dim=1)
            features_list.append(feat_var)
        
        # Stack features from all variables: [B, N, 2F]
        features = torch.stack(features_list, dim=1)
        
        # Part II: Learn inter-variable correlations
        # Permute to [B, 2F, N] for IEBlock-C
        features = features.permute(0, 2, 1)  # [B, 2F, N]
        
        # Apply IEBlock-C: [B, 2F, N] -> [B, pred_len, N]
        output = self.ieblock_c(features)
        
        # Permute back: [B, pred_len, N] -> [B, N, pred_len] -> [B, pred_len, N]
        output = output.permute(0, 2, 1)  # [B, N, pred_len]
        output = output.permute(0, 2, 1)  # [B, pred_len, N]
        
        # De-normalization
        output = output * stdev + means
        
        return output