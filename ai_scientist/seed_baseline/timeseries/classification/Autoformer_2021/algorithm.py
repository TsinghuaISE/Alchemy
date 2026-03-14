import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import math

# ===== Series Decomposition Block =====
class SeriesDecomp(nn.Module):
    """
    Series decomposition block using moving average.
    Extracts trend-cyclical component via AvgPool and computes seasonal component.
    """
    def __init__(self, kernel_size=25):
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        Returns:
            seasonal: [B, L, D]
            trend: [B, L, D]
        """
        # Padding to keep length unchanged
        # pad_left = (kernel_size - 1) // 2
        # pad_right = kernel_size - 1 - pad_left
        pad_size = (self.kernel_size - 1) // 2
        # x: [B, L, D] -> [B, D, L] for AvgPool1d
        x_permuted = x.permute(0, 2, 1)  # [B, D, L]
        
        # Padding: replicate boundary values
        x_padded = F.pad(x_permuted, (pad_size, pad_size), mode='replicate')  # [B, D, L+pad]
        
        # Apply moving average
        trend = self.avg_pool(x_padded)  # [B, D, L]
        trend = trend.permute(0, 2, 1)  # [B, L, D]
        
        # Seasonal component
        seasonal = x - trend  # [B, L, D]
        
        return seasonal, trend

# ===== Auto-Correlation Mechanism =====
class AutoCorrelation(nn.Module):
    """
    Auto-Correlation mechanism with FFT-based autocorrelation and time delay aggregation.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        Time delay aggregation for training (batch-averaged delays).
        Args:
            values: [B, H, L, D]
            corr: [B, H, L]
        Returns:
            aggregated: [B, H, L, D]
        """
        B, H, L, D = values.shape
        
        # Calculate top_k based on factor and log(L)
        top_k = int(self.factor * math.log(L))
        top_k = max(1, min(top_k, L))
        
        # Average correlation across batch and heads to get shared delays
        mean_corr = corr.mean(dim=0).mean(dim=0)  # [L]
        
        # Select top-k delays
        weights, delays = torch.topk(mean_corr, top_k, dim=-1)  # [top_k]
        weights = torch.softmax(weights, dim=-1)  # [top_k]
        
        # Initialize aggregation
        tmp_corr = corr.unsqueeze(-1).expand(B, H, L, D)  # [B, H, L, D]
        tmp_values = values
        
        # Aggregate by rolling
        delays_agg = torch.zeros_like(values).float()  # [B, H, L, D]
        for i in range(top_k):
            pattern = torch.roll(tmp_values, shifts=-int(delays[i]), dims=2)  # [B, H, L, D]
            delays_agg += pattern * weights[i]
        
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        Time delay aggregation for inference (per-sample delays).
        Args:
            values: [B, H, L, D]
            corr: [B, H, L]
        Returns:
            aggregated: [B, H, L, D]
        """
        B, H, L, D = values.shape
        
        # Calculate top_k
        top_k = int(self.factor * math.log(L))
        top_k = max(1, min(top_k, L))
        
        # Select top-k delays per sample
        weights, delays = torch.topk(corr, top_k, dim=-1)  # [B, H, top_k]
        weights = torch.softmax(weights, dim=-1)  # [B, H, top_k]
        
        # Use torch.gather for vectorized aggregation
        # Double values for circular indexing
        tmp_values = values.repeat(1, 1, 2, 1)  # [B, H, 2L, D]
        
        # Create index tensor
        init_index = torch.arange(L, device=values.device).view(1, 1, 1, -1)  # [1, 1, 1, L]
        init_index = init_index.expand(B, H, top_k, L)  # [B, H, top_k, L]
        
        # Add delays to indices
        delays_expanded = delays.unsqueeze(-1).expand(B, H, top_k, L)  # [B, H, top_k, L]
        tmp_delay = init_index + delays_expanded  # [B, H, top_k, L]
        
        # Gather patterns
        tmp_delay = tmp_delay.view(B, H, -1)  # [B, H, top_k*L]
        tmp_delay = tmp_delay.unsqueeze(-1).expand(B, H, top_k * L, D)  # [B, H, top_k*L, D]
        
        patterns = torch.gather(tmp_values, dim=2, index=tmp_delay)  # [B, H, top_k*L, D]
        patterns = patterns.view(B, H, top_k, L, D)  # [B, H, top_k, L, D]
        
        # Weighted aggregation
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # [B, H, top_k, 1, 1]
        delays_agg = torch.sum(patterns * weights_expanded, dim=2)  # [B, H, L, D]
        
        return delays_agg

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Args:
            queries: [B, L, H, D]
            keys: [B, S, H, D]
            values: [B, S, H, D]
        Returns:
            out: [B, L, H, D]
            attn: None or attention weights
        """
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        
        # Transpose to [B, H, L, D] and [B, H, S, D]
        queries = queries.permute(0, 2, 1, 3)  # [B, H, L, D]
        keys = keys.permute(0, 2, 1, 3)  # [B, H, S, D]
        values = values.permute(0, 2, 1, 3)  # [B, H, S, D]
        
        # Compute autocorrelation using FFT (Wiener-Khinchin theorem)
        # For each head, compute R_QK(tau)
        # queries: [B, H, L, D], keys: [B, H, S, D]
        
        # Compute FFT for queries and keys
        q_fft = torch.fft.rfft(queries, dim=2)  # [B, H, L//2+1, D] (complex)
        k_fft = torch.fft.rfft(keys, dim=2)  # [B, H, S//2+1, D] (complex)
        
        # Compute cross-power spectral density: S_QK(f) = Q(f) * conj(K(f))
        # Sum over D dimension to get scalar spectral density
        spd = q_fft * torch.conj(k_fft)  # [B, H, min(L,S)//2+1, D]
        spd = spd.sum(dim=-1)  # [B, H, min(L,S)//2+1]
        
        # Inverse FFT to get autocorrelation
        corr = torch.fft.irfft(spd, n=S, dim=-1)  # [B, H, S]
        
        # For encoder-decoder attention, S might differ from L
        # Resize correlation to L if needed
        if S != L:
            # Interpolate or truncate
            if S > L:
                corr = corr[:, :, :L]
            else:
                # Pad with zeros
                corr = F.pad(corr, (0, L - S))
        
        # Time delay aggregation
        if self.training:
            V = self.time_delay_agg_training(values, corr)
        else:
            V = self.time_delay_agg_inference(values, corr)
        
        # Apply dropout
        V = self.dropout(V)
        
        # Transpose back to [B, L, H, D]
        V = V.permute(0, 2, 1, 3)  # [B, L, H, D]
        
        if self.output_attention:
            return V, corr
        else:
            return V, None

class AutoCorrelationLayer(nn.Module):
    """
    Multi-head Auto-Correlation layer with projections.
    """
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Args:
            queries: [B, L, d_model]
            keys: [B, S, d_model]
            values: [B, S, d_model]
        Returns:
            out: [B, L, d_model]
            attn: attention weights or None
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # Project and reshape
        queries = self.query_projection(queries).view(B, L, H, -1)  # [B, L, H, D]
        keys = self.key_projection(keys).view(B, S, H, -1)  # [B, S, H, D]
        values = self.value_projection(values).view(B, S, H, -1)  # [B, S, H, D]
        
        # Apply Auto-Correlation
        out, attn = self.inner_correlation(queries, keys, values, attn_mask, tau, delta)
        
        # Reshape and project
        out = out.reshape(B, L, -1)  # [B, L, H*D]
        out = self.out_projection(out)  # [B, L, d_model]
        
        return out, attn

# ===== Encoder Layer =====
class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with Auto-Correlation and series decomposition.
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.decomp1 = SeriesDecomp(kernel_size=moving_avg)
        self.decomp2 = SeriesDecomp(kernel_size=moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Args:
            x: [B, L, d_model]
        Returns:
            x: [B, L, d_model] (seasonal component)
            attn: attention weights or None
        """
        # Auto-Correlation with residual
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        
        # Series decomposition
        x, _ = self.decomp1(x)  # Only keep seasonal part
        
        # Feed-forward with residual
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # [B, d_ff, L]
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # [B, L, d_model]
        
        # Series decomposition
        x, _ = self.decomp2(x + y)  # Only keep seasonal part
        
        return x, attn

# ===== Encoder =====
class Encoder(nn.Module):
    """
    Autoformer encoder with stacked encoder layers.
    """
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Args:
            x: [B, L, d_model]
        Returns:
            x: [B, L, d_model]
            attns: list of attention weights
        """
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, attns

# ===== Decoder Layer =====
class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with Auto-Correlation and series decomposition.
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation='relu'):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.decomp1 = SeriesDecomp(kernel_size=moving_avg)
        self.decomp2 = SeriesDecomp(kernel_size=moving_avg)
        self.decomp3 = SeriesDecomp(kernel_size=moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu
        
        # Trend projection layers
        self.projection1 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, 
                                     stride=1, padding=1, padding_mode='circular', bias=False)
        self.projection2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3,
                                     stride=1, padding=1, padding_mode='circular', bias=False)
        self.projection3 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3,
                                     stride=1, padding=1, padding_mode='circular', bias=False)
        
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        Args:
            x: [B, L, d_model]
            cross: [B, S, d_model]
        Returns:
            x: [B, L, d_model] (seasonal component)
            trend: [B, L, c_out] (trend component)
        """
        # Self Auto-Correlation with residual
        new_x, _ = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)
        x = x + self.dropout(new_x)
        
        # Series decomposition
        x, trend1 = self.decomp1(x)
        
        # Cross Auto-Correlation with residual
        new_x, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        
        # Series decomposition
        x, trend2 = self.decomp2(x)
        
        # Feed-forward with residual
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # [B, d_ff, L]
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # [B, L, d_model]
        
        # Series decomposition
        x, trend3 = self.decomp3(x + y)
        
        # Project trends to c_out dimension
        # trend: [B, L, d_model] -> [B, d_model, L] -> [B, c_out, L] -> [B, L, c_out]
        trend1 = self.projection1(trend1.permute(0, 2, 1)).permute(0, 2, 1)
        trend2 = self.projection2(trend2.permute(0, 2, 1)).permute(0, 2, 1)
        trend3 = self.projection3(trend3.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Accumulate trends
        residual_trend = trend1 + trend2 + trend3
        
        return x, residual_trend

# ===== Decoder =====
class Decoder(nn.Module):
    """
    Autoformer decoder with stacked decoder layers and trend accumulation.
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        
    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None, tau=None, delta=None):
        """
        Args:
            x: [B, L, d_model]
            cross: [B, S, d_model]
            trend: [B, L, c_out] (initial trend)
        Returns:
            x: [B, L, d_model] (seasonal component)
            trend: [B, L, c_out] (accumulated trend)
        """
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
            trend = trend + residual_trend
        
        if self.norm is not None:
            x = self.norm(x)
        
        if self.projection is not None:
            x = self.projection(x)  # [B, L, c_out]
        
        return x, trend

# ===== Main Model =====
class Model(nn.Module):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
    Paper link: https://arxiv.org/abs/2106.13008
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, 'label_len', configs.seq_len // 2)
        self.pred_len = getattr(configs, 'pred_len', 0)
        
        # For anomaly detection, set pred_len to seq_len if not provided
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, 
                                          configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed,
                                          configs.freq, configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(mask_flag=False, factor=getattr(configs, 'factor', 1),
                                       attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=getattr(configs, 'moving_avg', 25),
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(mask_flag=True, factor=getattr(configs, 'factor', 1),
                                       attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(mask_flag=False, factor=getattr(configs, 'factor', 1),
                                       attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=getattr(configs, 'moving_avg', 25),
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        
        # Series decomposition for initialization
        self.decomp = SeriesDecomp(kernel_size=getattr(configs, 'moving_avg', 25))
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass for different tasks.
        """
        # For forecasting task
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, I, d_model]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # [B, I, d_model]
        
        # Decoder initialization
        # Decompose the latter half of encoder input
        seasonal_init, trend_init = self.decomp(x_enc[:, -self.label_len:, :])  # [B, label_len, D]
        
        # Placeholders for future
        seasonal_zeros = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]], 
                                     device=x_enc.device)
        trend_mean = mean.repeat(1, self.pred_len, 1)  # [B, pred_len, D]
        
        # Concatenate
        dec_inp_seasonal = torch.cat([seasonal_init, seasonal_zeros], dim=1)  # [B, label_len+pred_len, D]
        dec_inp_trend = torch.cat([trend_init, trend_mean], dim=1)  # [B, label_len+pred_len, D]
        
        # Decoder embedding
        dec_out = self.dec_embedding(dec_inp_seasonal, x_mark_dec)  # [B, label_len+pred_len, d_model]
        
        # Decoder forward
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=dec_inp_trend)  # [B, label_len+pred_len, c_out]
        
        # Final prediction
        dec_out = seasonal_part + trend_part  # [B, label_len+pred_len, c_out]
        
        # Return only the prediction part
        dec_out = dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]
        
        return dec_out