import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
import torch.nn.functional as F

# ===== Inlined Components =====
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
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

class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean
# ===== End Inlined Components =====

class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """

    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24],
                 isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.device = device

        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                     kernel_size=(len(self.conv_kernel), 1))

        # feedforward network
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=self.device)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]  # truncate

        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        self.device = src.device
        # multi-scale
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)

        # merge
        mg = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1).to(self.device)), dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)

        return self.norm2(mg + y)

class SeasonalPrediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                 conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(SeasonalPrediction, self).__init__()

        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, n_heads=n_heads,
                                      decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                                      isometric_kernel=isometric_kernel, device=device)
                                  for i in range(d_layers)])

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=zt53IDUR1U
    """
    def __init__(self, configs, conv_kernel=[12, 16]):
        """
        conv_kernel: downsampling and upsampling convolution kernel_size
        """
        super(Model, self).__init__()

        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        for ii in conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((configs.seq_len + configs.pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((configs.seq_len + configs.pred_len + ii - 1) // ii)

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        # Multiple Series decomposition block from FEDformer
        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # embedding
        self.dec_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.conv_trans = SeasonalPrediction(embedding_size=configs.d_model, n_heads=configs.n_heads,
                                             dropout=configs.dropout,
                                             d_layers=configs.d_layers, decomp_kernel=decomp_kernel,
                                             c_out=configs.c_out, conv_kernel=conv_kernel,
                                             isometric_kernel=isometric_kernel, device=torch.device('cuda:0'))
        # refer to DLinear
        self.regression = nn.Linear(configs.seq_len, configs.pred_len)
        self.regression.weight = nn.Parameter(
            (1 / configs.pred_len) * torch.ones([configs.pred_len, configs.seq_len]),
            requires_grad=True)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # embedding
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.seq_len:, :], zeros], dim=1)
        dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
