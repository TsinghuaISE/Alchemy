import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import math

# ===== Embedding Components =====
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
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

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

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
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) \
            if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

# ===== Frequency Enhanced Blocks - Fourier Version =====
class FourierBlock(nn.Module):
    """
    Frequency Enhanced Block with Fourier Transform (FEB-f)
    """
    def __init__(self, d_model, modes=64, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        self.d_model = d_model
        self.modes = modes
        self.mode_select_method = mode_select_method
        
        # Learnable frequency kernel R ∈ C^(D×D×M)
        self.scale = 1 / (d_model * d_model)
        self.weights_real = nn.Parameter(self.scale * torch.randn(d_model, d_model, modes, dtype=torch.float32))
        self.weights_imag = nn.Parameter(self.scale * torch.randn(d_model, d_model, modes, dtype=torch.float32))

    def forward(self, q):
        # q: [B, L, D]
        B, L, D = q.shape
        
        # FFT along the time dimension
        q_ft = torch.fft.rfft(q, dim=1)  # [B, L//2+1, D]
        
        # Select modes
        modes = min(self.modes, q_ft.shape[1])
        
        # Initialize output
        out_ft = torch.zeros(B, L // 2 + 1, D, device=q.device, dtype=torch.cfloat)
        
        # Process selected modes
        for i in range(modes):
            # q_ft[:, i, :] shape: [B, D]
            # weights: [D, D]
            weight = torch.complex(self.weights_real[:, :, i], self.weights_imag[:, :, i])
            out_ft[:, i, :] = torch.einsum('bd,de->be', q_ft[:, i, :], weight)
        
        # Inverse FFT
        out = torch.fft.irfft(out_ft, n=L, dim=1)  # [B, L, D]
        
        return out

class FourierCrossAttention(nn.Module):
    """
    Frequency Enhanced Attention with Fourier Transform (FEA-f)
    """
    def __init__(self, d_model, n_heads, modes=64, activation='softmax'):
        super(FourierCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.modes = modes
        self.activation = activation
        
        assert d_model % n_heads == 0
        self.d_keys = d_model // n_heads
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        # queries: [B, L, D], keys: [B, S, D], values: [B, S, D]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # Project and reshape
        queries = self.query_projection(queries).view(B, L, H, -1)  # [B, L, H, D//H]
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        # Transpose to [B, H, L/S, D//H]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Apply FFT
        q_ft = torch.fft.rfft(queries, dim=2)  # [B, H, L//2+1, D//H]
        k_ft = torch.fft.rfft(keys, dim=2)     # [B, H, S//2+1, D//H]
        v_ft = torch.fft.rfft(values, dim=2)   # [B, H, S//2+1, D//H]
        
        # Select modes
        modes_q = min(self.modes, q_ft.shape[2])
        modes_k = min(self.modes, k_ft.shape[2])
        
        # Truncate to selected modes
        q_ft = q_ft[:, :, :modes_q, :]
        k_ft = k_ft[:, :, :modes_k, :]
        v_ft = v_ft[:, :, :modes_k, :]
        
        # Attention in frequency domain: σ(Q̃·K̃^T)·Ṽ
        # For simplicity, we use real-valued attention on magnitudes
        attn_weights = torch.einsum('bhld,bhsd->bhls', q_ft.real, k_ft.real) + \
                       torch.einsum('bhld,bhsd->bhls', q_ft.imag, k_ft.imag)
        
        if self.activation == 'softmax':
            attn_weights = torch.softmax(attn_weights, dim=-1)
        elif self.activation == 'tanh':
            attn_weights = torch.tanh(attn_weights)
        
        # Apply attention to values
        out_ft = torch.einsum('bhls,bhsd->bhld', attn_weights, v_ft)
        
        # Pad back to full frequency domain
        out_ft_full = torch.zeros(B, H, L // 2 + 1, self.d_keys, device=queries.device, dtype=torch.cfloat)
        out_ft_full[:, :, :modes_q, :] = out_ft
        
        # Inverse FFT
        out = torch.fft.irfft(out_ft_full, n=L, dim=2)  # [B, H, L, D//H]
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, -1)  # [B, L, D]
        out = self.out_projection(out)
        
        return out, None

# ===== Mixture of Experts Decomposition =====
class MOEDecomp(nn.Module):
    """
    Mixture Of Experts Decomposition block (MOEDecomp)
    X_trend = Softmax(L(x)) * F(x)
    """
    def __init__(self, d_model, kernel_sizes=[3, 5, 7]):
        super(MOEDecomp, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_experts = len(kernel_sizes)
        
        # Multiple average pooling filters with different kernel sizes
        self.avg_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=1, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Learnable mixing network
        self.mixing_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, self.num_experts)
        )

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        
        # Extract multiple trends using different filters
        trends = []
        for avg_pool in self.avg_pools:
            # Apply average pooling
            trend = avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, D]
            trends.append(trend)
        
        trends = torch.stack(trends, dim=-1)  # [B, L, D, num_experts]
        
        # Compute mixing weights
        weights = self.mixing_network(x)  # [B, L, num_experts]
        weights = F.softmax(weights, dim=-1).unsqueeze(2)  # [B, L, 1, num_experts]
        
        # Weighted combination
        x_trend = (trends * weights).sum(dim=-1)  # [B, L, D]
        x_seasonal = x - x_trend
        
        return x_seasonal, x_trend

# ===== Layer Normalization =====
class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias

# ===== Encoder Layer =====
class EncoderLayer(nn.Module):
    """
    FEDformer encoder layer with progressive decomposition architecture
    """
    def __init__(self, d_model, n_heads, d_ff=None, modes=64, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        # FEB block (Fourier Enhanced Block)
        self.feb = FourierBlock(d_model, modes=modes)
        
        # Feed-forward network
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        # Decomposition blocks
        self.decomp1 = MOEDecomp(d_model)
        self.decomp2 = MOEDecomp(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # FEB + residual + decomposition
        new_x = self.feb(x)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        
        # Feed-forward + residual + decomposition
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        
        return res, None

# ===== Encoder =====
class Encoder(nn.Module):
    """
    FEDformer encoder
    """
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

# ===== Decoder Layer =====
class DecoderLayer(nn.Module):
    """
    FEDformer decoder layer with progressive decomposition architecture
    """
    def __init__(self, d_model, n_heads, c_out, d_ff=None, modes=64, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        # FEB for self-attention
        self.self_feb = FourierBlock(d_model, modes=modes)
        
        # FEA for cross-attention
        self.cross_fea = FourierCrossAttention(d_model, n_heads, modes=modes, activation='softmax')
        
        # Feed-forward network
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        # Decomposition blocks
        self.decomp1 = MOEDecomp(d_model)
        self.decomp2 = MOEDecomp(d_model)
        self.decomp3 = MOEDecomp(d_model)
        
        # Trend projections
        self.projection1 = nn.Conv1d(in_channels=d_model, out_channels=c_out, 
                                     kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)
        self.projection2 = nn.Conv1d(in_channels=d_model, out_channels=c_out,
                                     kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)
        self.projection3 = nn.Conv1d(in_channels=d_model, out_channels=c_out,
                                     kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self FEB + residual + decomposition
        x = x + self.dropout(self.self_feb(x))
        x, trend1 = self.decomp1(x)
        
        # Cross FEA + residual + decomposition
        x = x + self.dropout(self.cross_fea(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        
        # Feed-forward + residual + decomposition
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)
        
        # Project trends
        trend1 = self.projection1(trend1.permute(0, 2, 1)).transpose(1, 2)
        trend2 = self.projection2(trend2.permute(0, 2, 1)).transpose(1, 2)
        trend3 = self.projection3(trend3.permute(0, 2, 1)).transpose(1, 2)
        
        residual_trend = trend1 + trend2 + trend3
        
        return x, residual_trend

# ===== Decoder =====
class Decoder(nn.Module):
    """
    FEDformer decoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
            
        return x, trend

# ===== Main Model =====
class Model(nn.Module):
    """
    FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting
    Paper link: https://arxiv.org/abs/2201.12740
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # Get modes parameter
        self.modes = getattr(configs, 'modes', 64)
        
        # Decomposition
        self.decomp = MOEDecomp(configs.d_model)
        
        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model,
                    configs.n_heads,
                    configs.d_ff,
                    modes=self.modes,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs.d_model,
                    configs.n_heads,
                    configs.c_out,
                    configs.d_ff,
                    modes=self.modes,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]