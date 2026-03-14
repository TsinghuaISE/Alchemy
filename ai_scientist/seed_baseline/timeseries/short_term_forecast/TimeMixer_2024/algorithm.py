import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# ===== Inlined Components =====
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
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

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

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

class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
# ===== End Inlined Components =====

class DFT_series_decomp(nn.Module):
    """
    DFT-based Series decomposition block
    """
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k, dim=1)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf, dim=1)
        x_trend = x - x_season
        return x_season, x_trend

class MultiScaleSeasonMixing(nn.Module):
    """
    Multi-scale Season Mixing module for capturing temporal patterns at different scales
    """
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # Mixing at different scales
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high.permute(0, 2, 1))
            out_low = out_low.permute(0, 2, 1) + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high)

        return out_season_list

class MultiScaleTrendMixing(nn.Module):
    """
    Multi-scale Trend Mixing module for capturing trend patterns
    """
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        # Mixing at different scales (from coarse to fine)
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low.permute(0, 2, 1))
            out_high = out_high.permute(0, 2, 1) + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low)

        out_trend_list.reverse()
        return out_trend_list

class PastDecomposableMixing(nn.Module):
    """
    Past Decomposable Mixing module - processes historical data
    """
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.down_sampling_window = configs.down_sampling_window
        self.down_sampling_layers = configs.down_sampling_layers

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = getattr(configs, 'channel_independence', False)

        if self.channel_independence:
            self.cross_layer = nn.Linear(configs.d_model, configs.d_model)
        else:
            self.cross_layer = nn.Linear(configs.enc_in, configs.enc_in)

        # Decomposition
        self.decomp_multi = nn.ModuleList(
            [DFT_series_decomp(configs.top_k) for _ in range(configs.down_sampling_layers + 1)]
        )

        # Mixing layers
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose at multiple scales
        season_list = []
        trend_list = []
        for i, x in enumerate(x_list):
            season, trend = self.decomp_multi[i](x)
            if self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            else:
                season = self.cross_layer(season.permute(0, 2, 1)).permute(0, 2, 1)
                trend = self.cross_layer(trend.permute(0, 2, 1)).permute(0, 2, 1)
            season_list.append(season)
            trend_list.append(trend)

        # Mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for i, (out_season, out_trend, length) in enumerate(
            zip(out_season_list, out_trend_list, length_list)
        ):
            out = out_season.permute(0, 2, 1) + out_trend.permute(0, 2, 1)
            if self.channel_independence:
                out = self.layer_norm(out)
            out_list.append(out[:, :length, :])

        return out_list

class Model(nn.Module):
    """
    TimeMixer model for time series imputation
    Paper: TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.label_len = getattr(configs, 'label_len', 0)
        
        # Model parameters
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.down_sampling_layers = configs.e_layers
        self.down_sampling_window = getattr(configs, 'down_sampling_window', 2)
        self.channel_independence = getattr(configs, 'channel_independence', False)
        self.top_k = getattr(configs, 'top_k', 5)
        
        # Update configs with derived parameters
        configs.down_sampling_layers = self.down_sampling_layers
        configs.down_sampling_window = self.down_sampling_window
        configs.channel_independence = self.channel_independence
        configs.top_k = self.top_k

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            getattr(configs, 'embed', 'timeF'),
            getattr(configs, 'freq', 'h'),
            configs.dropout
        )

        # Encoder - Past Decomposable Mixing
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(configs) for _ in range(configs.e_layers)]
        )

        # Projection layer
        self.projection = nn.Linear(configs.d_model, configs.c_out)

        # Normalization
        self.normalize_layers = Normalize(configs.enc_in, affine=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass - routes to task-specific method
        """

class TimeMixerConfig:
    """
    Configuration helper for TimeMixer model
    """
    def __init__(self, **kwargs):
        # Task settings
        self.task_name = kwargs.get('task_name', 'imputation')
        self.seq_len = kwargs.get('seq_len', 96)
        self.label_len = kwargs.get('label_len', 48)
        self.pred_len = kwargs.get('pred_len', 96)
        
        # Model architecture
        self.enc_in = kwargs.get('enc_in', 7)
        self.dec_in = kwargs.get('dec_in', 7)
        self.c_out = kwargs.get('c_out', 7)
        self.d_model = kwargs.get('d_model', 16)
        self.e_layers = kwargs.get('e_layers', 2)
        self.d_layers = kwargs.get('d_layers', 1)
        
        # TimeMixer specific
        self.down_sampling_layers = kwargs.get('down_sampling_layers', self.e_layers)
        self.down_sampling_window = kwargs.get('down_sampling_window', 2)
        self.channel_independence = kwargs.get('channel_independence', False)
        self.top_k = kwargs.get('top_k', 5)
        
        # Embedding
        self.embed = kwargs.get('embed', 'timeF')
        self.freq = kwargs.get('freq', 'h')
        self.dropout = kwargs.get('dropout', 0.1)
        
        # Training
        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.batch_size = kwargs.get('batch_size', 32)
        self.train_epochs = kwargs.get('train_epochs', 10)