import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# ===== Inlined Components =====

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(torch.log(torch.tensor(10000.0)) / d_model)).exp()

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

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(torch.log(torch.tensor(10000.0)) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                     freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class DeStationaryAttention(nn.Module):
    """De-stationary Attention mechanism that approximates attention from non-stationary series"""
    
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, attention_dropout=0.1):
        super(DeStationaryAttention, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.d_keys = d_keys
        self.n_heads = n_heads
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, tau=None, delta=None):
        """
        Args:
            queries: [B, L, D]
            keys: [B, S, D]
            values: [B, S, D]
            tau: [B] scaling factor (de-stationary factor)
            delta: [B, S] shifting vector (de-stationary factor)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # Project Q, K, V
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        # Transpose to [B, H, L, D]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / sqrt(self.d_keys)
        
        # Apply de-stationary factors if provided
        if tau is not None:
            # tau: [B] -> [B, 1, 1, 1]
            tau = tau.view(B, 1, 1, 1)
            scores = tau * scores
        
        if delta is not None:
            # delta: [B, S] -> [B, 1, 1, S]
            delta = delta.view(B, 1, 1, S)
            scores = scores + delta
        
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, values)  # [B, H, L, D]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        
        return self.out_projection(out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = DeStationaryAttention(d_model, n_heads, dropout=dropout)
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, tau=None, delta=None):
        # Self-attention with de-stationary factors
        new_x = self.attention(x, x, x, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # Feed-forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y)

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        
    def forward(self, x, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, tau=tau, delta=delta)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x

class DeStationaryFactorPredictor(nn.Module):
    """MLP to predict de-stationary factors tau and delta from series statistics"""
    
    def __init__(self, enc_in, seq_len, d_model=64):
        super(DeStationaryFactorPredictor, self).__init__()
        
        # Predictor for tau (scaling factor)
        self.tau_predictor = nn.Sequential(
            nn.Linear(enc_in + 1, d_model),  # +1 for sigma_x
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Predictor for delta (shifting vector)
        self.delta_predictor = nn.Sequential(
            nn.Linear(enc_in + enc_in, d_model),  # mean and raw series
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len)
        )
        
    def forward(self, x, mu_x, sigma_x):
        """
        Args:
            x: [B, L, C] raw series
            mu_x: [B, C] mean of x
            sigma_x: [B, C] std of x
        Returns:
            tau: [B] scaling factor
            delta: [B, L] shifting vector
        """
        B, L, C = x.shape
        
        # Predict tau from sigma_x and aggregated x
        x_mean = x.mean(dim=1)  # [B, C]
        tau_input = torch.cat([sigma_x, x_mean], dim=-1)  # [B, 2C]
        log_tau = self.tau_predictor(tau_input).squeeze(-1)  # [B]
        tau = torch.exp(log_tau)  # Ensure positive
        
        # Predict delta from mu_x and aggregated x
        delta_input = torch.cat([mu_x, x_mean], dim=-1)  # [B, 2C]
        delta = self.delta_predictor(delta_input)  # [B, L]
        
        return tau, delta

# ===== End Inlined Components =====

class Model(nn.Module):
    """
    Non-stationary Transformer for Time Series Imputation
    Paper: Non-stationary Transformers (2022)
    
    Key components:
    1. Series Stationarization: Normalize input series to make them stationary
    2. De-stationary Attention: Recover non-stationary information in attention mechanism
    3. De-normalization: Transform output back to original scale
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq,
            configs.dropout
        )
        
        # De-stationary factor predictor
        self.factor_predictor = DeStationaryFactorPredictor(
            configs.enc_in, 
            configs.seq_len,
            d_model=configs.d_model
        )
        
        # Encoder with De-stationary Attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model,
                    configs.n_heads,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        
        # Projection layer
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        
    def normalization(self, x):
        """
        Series Stationarization: Normalize input series
        Args:
            x: [B, L, C] input series
        Returns:
            x_normalized: [B, L, C] normalized series
            mu_x: [B, C] mean
            sigma_x: [B, C] std
        """
        # Calculate mean and std along temporal dimension
        mu_x = x.mean(dim=1, keepdim=False)  # [B, C]
        sigma_x = torch.sqrt(torch.var(x, dim=1, keepdim=False, unbiased=False) + 1e-5)  # [B, C]
        
        # Normalize
        x_normalized = (x - mu_x.unsqueeze(1)) / sigma_x.unsqueeze(1)
        
        return x_normalized, mu_x, sigma_x
    
    def denormalization(self, x, mu_x, sigma_x):
        """
        De-normalization: Transform output back to original scale
        Args:
            x: [B, L, C] normalized output
            mu_x: [B, C] mean
            sigma_x: [B, C] std
        Returns:
            x_denormalized: [B, L, C] denormalized output
        """
        x_denormalized = x * sigma_x.unsqueeze(1) + mu_x.unsqueeze(1)
        return x_denormalized
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Imputation forward pass
        Args:
            x_enc: [B, L, C] input series with missing values
            x_mark_enc: [B, L, mark_dim] temporal encoding
            mask: [B, L, C] mask (1=observed, 0=missing)
        Returns:
            dec_out: [B, L, C] imputed series
        """
        # Store raw series for de-stationary factor prediction
        x_raw = x_enc.clone()
        
        # Series Stationarization (Normalization)
        x_normalized, mu_x, sigma_x = self.normalization(x_enc)
        
        # Predict de-stationary factors from raw series statistics
        tau, delta = self.factor_predictor(x_raw, mu_x, sigma_x)
        
        # Embedding
        enc_out = self.enc_embedding(x_normalized, x_mark_enc)
        
        # Encoder with De-stationary Attention
        enc_out = self.encoder(enc_out, tau=tau, delta=delta)
        
        # Projection
        dec_out = self.projection(enc_out)
        
        # De-normalization
        dec_out = self.denormalization(dec_out, mu_x, sigma_x)
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass
        Args:
            x_enc: [B, L, C] input series
            x_mark_enc: [B, L, mark_dim] temporal encoding
            x_dec: not used in imputation
            x_mark_dec: not used in imputation
            mask: [B, L, C] mask (1=observed, 0=missing)
        Returns:
            dec_out: [B, L, C] imputed series (same length as input)
        """
        dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return dec_out  # [B, L, C]