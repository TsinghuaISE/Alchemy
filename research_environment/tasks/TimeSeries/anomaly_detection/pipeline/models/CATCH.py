"""
CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching
Paper: https://arxiv.org/pdf/2410.12261 (ICLR'25)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from torch.nn.functional import gumbel_softmax


# ===================== RevIN =====================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        elif mode == 'transform':
            x = self._normalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
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
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ===================== Channel Mask Generator =====================
class ChannelMaskGenerator(nn.Module):
    def __init__(self, input_size, n_vars):
        super(ChannelMaskGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size * 2, n_vars, bias=False),
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.generator[0].weight.zero_()
        self.n_vars = n_vars

    def forward(self, x):
        distribution_matrix = self.generator(x)
        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)
        inverse_eye = 1 - torch.eye(self.n_vars).to(x.device)
        diag = torch.eye(self.n_vars).to(x.device)
        resample_matrix = torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye) + diag
        return resample_matrix

    def _bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape
        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix
        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix + 1e-10)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix + 1e-10)
        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)
        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix


# ===================== Dynamical Contrastive Loss =====================
class DynamicalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, k=0.3):
        super(DynamicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.k = k

    def forward(self, scores, attn_mask, norm_matrix):
        b = scores.shape[0]
        n_vars = scores.shape[-1]
        cosine = (scores / (norm_matrix + 1e-10)).mean(1)
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask
        all_scores = torch.exp(cosine / self.temperature)
        clustering_loss = -torch.log(pos_scores.sum(dim=-1) / (all_scores.sum(dim=-1) + 1e-10) + 1e-10)
        eye = torch.eye(attn_mask.shape[-1]).unsqueeze(0).repeat(b, 1, 1).to(attn_mask.device)
        regular_loss = 1 / (n_vars * (n_vars - 1) + 1e-10) * torch.norm(
            eye.reshape(b, -1) - attn_mask.reshape((b, -1)), p=1, dim=-1
        )
        loss = clustering_loss.mean(1) + self.k * regular_loss
        return loss.mean()


# ===================== Frequency Loss =====================
class FrequencyLoss(nn.Module):
    def __init__(self, auxi_loss='MAE', auxi_type='complex', auxi_mode='fft', module_first=True):
        super(FrequencyLoss, self).__init__()
        self.auxi_loss = auxi_loss
        self.auxi_type = auxi_type
        self.module_first = module_first
        if auxi_mode == "fft":
            self.fft = torch.fft.fft
        elif auxi_mode == "rfft":
            self.fft = torch.fft.rfft
        else:
            raise NotImplementedError

    def forward(self, outputs, batch_y):
        if outputs.is_complex():
            frequency_outputs = outputs
        else:
            frequency_outputs = self.fft(outputs, dim=1)

        if self.auxi_type == 'complex':
            loss_auxi = frequency_outputs - self.fft(batch_y, dim=1)
        elif self.auxi_type == 'complex-phase':
            loss_auxi = (frequency_outputs - self.fft(batch_y, dim=1)).angle()
        elif self.auxi_type == 'mag':
            loss_auxi = frequency_outputs.abs() - self.fft(batch_y, dim=1).abs()
        elif self.auxi_type == 'phase':
            loss_auxi = frequency_outputs.angle() - self.fft(batch_y, dim=1).angle()
        else:
            loss_auxi = frequency_outputs - self.fft(batch_y, dim=1)

        if self.auxi_loss == "MAE":
            loss_auxi = loss_auxi.abs().mean() if self.module_first else loss_auxi.mean().abs()
        elif self.auxi_loss == "MSE":
            loss_auxi = (loss_auxi.abs() ** 2).mean() if self.module_first else (loss_auxi ** 2).mean().abs()
        else:
            raise NotImplementedError

        return loss_auxi


class FrequencyCriterion(nn.Module):
    """
    轻量级频域打分，用于测试阶段的分数融合。
    设计目标：只在 CATCH 测试分支调用，不影响其他模型。
    """

    def __init__(self, patch_size: int, patch_stride: int, win_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.win_size = win_size

    def forward(self, outputs: torch.Tensor, batch_y: torch.Tensor):
        """
        outputs: [B, T, C], batch_y: [B, T, C]
        返回频域分数字段，形状 [B, T]，通过简单广播回时间维度。
        """
        # unfold 到 patch
        out_patch = outputs.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)  # [B, n, C, P]
        y_patch = batch_y.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)    # [B, n, C, P]

        B, n, C, P = out_patch.shape
        out_patch = out_patch.contiguous().view(B * n, P, C)
        y_patch = y_patch.contiguous().view(B * n, P, C)

        # 频域差异
        out_fft = torch.fft.fft(out_patch, dim=1)
        y_fft = torch.fft.fft(y_patch, dim=1)
        diff = (out_fft - y_fft).abs().mean(dim=1)  # [B*n, C]

        diff = diff.view(B, n, C).mean(dim=1)  # [B, C]
        # 简单广播回时间维度（每个时间步相同的频域分数）
        freq_score = diff.mean(dim=-1, keepdim=True)  # [B, 1]
        freq_score = freq_score.repeat(1, outputs.size(1))  # [B, T]
        return freq_score


# ===================== Transformer Components =====================
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ChannelAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.dynamicalContrastiveLoss = DynamicalContrastiveLoss(k=regular_lambda, temperature=temperature)

    def forward(self, x, attn_mask=None):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        scale = 1 / self.d_k

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dynamical_contrastive_loss = None
        scores = einsum('b h i d, b h j d -> b h i j', q, k)

        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        norm_matrix = torch.einsum('bhid,bhjd->bhij', q_norm, k_norm)

        if attn_mask is not None:
            def _mask(scores_tensor, attn_mask_tensor):
                large_negative = -math.log(1e10)
                attention_mask = torch.where(attn_mask_tensor == 0, large_negative, 0.0)
                scores_tensor = scores_tensor * attn_mask_tensor.unsqueeze(1) + attention_mask.unsqueeze(1)
                return scores_tensor

            masked_scores = _mask(scores, attn_mask)
            dynamical_contrastive_loss = self.dynamicalContrastiveLoss(scores, attn_mask, norm_matrix)
        else:
            masked_scores = scores

        attn = self.attend(masked_scores * scale)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn, dynamical_contrastive_loss


class ChannelTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ChannelAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                              regular_lambda=regular_lambda, temperature=temperature)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, attn_mask=None):
        total_loss = 0
        last_attn = None
        for attn, ff in self.layers:
            x_n, attn_out, dcloss = attn(x, attn_mask=attn_mask)
            last_attn = attn_out
            if dcloss is not None:
                total_loss += dcloss
            x = x_n + x
            x = ff(x) + x
        device = x.device
        dcloss = total_loss / len(self.layers) if len(self.layers) > 0 else torch.zeros((), device=device)
        return x, last_attn, dcloss


class TransC(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model,
                 regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = ChannelTransformer(dim, depth, heads, dim_head, mlp_dim, dropout,
                                              regular_lambda=regular_lambda, temperature=temperature)
        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x, attn_mask=None):
        x = self.to_patch_embedding(x)
        x, attn, dcloss = self.transformer(x, attn_mask)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x, dcloss


# ===================== Flatten Head =====================
class FlattenHead(nn.Module):
    def __init__(self, individual, n_vars, nf, seq_len, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, seq_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, seq_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears1[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
        return x


# ===================== CATCH Backbone =====================
class CATCHBackbone(nn.Module):
    def __init__(self, configs):
        super(CATCHBackbone, self).__init__()

        self.seq_len = configs.seq_len
        self.c_in = configs.enc_in
        self.patch_size = getattr(configs, 'catch_patch_size', 16)
        self.patch_stride = getattr(configs, 'catch_patch_stride', 8)
        self.cf_dim = getattr(configs, 'catch_cf_dim', 64)
        self.e_layers = getattr(configs, 'e_layers', 3)
        self.n_heads = getattr(configs, 'n_heads', 2)
        self.d_ff = getattr(configs, 'd_ff', 256)
        self.d_model = getattr(configs, 'd_model', 128)
        self.head_dim = getattr(configs, 'catch_head_dim', 64)
        self.dropout = getattr(configs, 'dropout', 0.2)
        self.head_dropout = getattr(configs, 'catch_head_dropout', 0.1)
        self.individual = getattr(configs, 'catch_individual', 0)
        self.affine = getattr(configs, 'catch_affine', 0)
        self.subtract_last = getattr(configs, 'catch_subtract_last', 0)
        self.regular_lambda = getattr(configs, 'catch_regular_lambda', 0.5)
        self.temperature = getattr(configs, 'catch_temperature', 0.07)

        self.revin_layer = RevIN(self.c_in, affine=self.affine, subtract_last=self.subtract_last)

        patch_num = int((self.seq_len - self.patch_size) / self.patch_stride + 1)
        self.norm = nn.LayerNorm(self.patch_size)

        self.mask_generator = ChannelMaskGenerator(input_size=self.patch_size, n_vars=self.c_in)

        self.frequency_transformer = TransC(
            dim=self.cf_dim, depth=self.e_layers, heads=self.n_heads, mlp_dim=self.d_ff,
            dim_head=self.head_dim, dropout=self.dropout, patch_dim=self.patch_size * 2,
            horizon=self.seq_len * 2, d_model=self.d_model * 2,
            regular_lambda=self.regular_lambda, temperature=self.temperature
        )

        self.head_nf_f = self.d_model * 2 * patch_num
        self.head_f1 = FlattenHead(self.individual, self.c_in, self.head_nf_f, self.seq_len,
                                   head_dropout=self.head_dropout)
        self.head_f2 = FlattenHead(self.individual, self.c_in, self.head_nf_f, self.seq_len,
                                   head_dropout=self.head_dropout)

        self.ircom = nn.Linear(self.seq_len * 2, self.seq_len)
        self.get_r = nn.Linear(self.d_model * 2, self.d_model * 2)
        self.get_i = nn.Linear(self.d_model * 2, self.d_model * 2)

    def forward(self, z):
        z = self.revin_layer(z, 'norm')
        z = z.permute(0, 2, 1)
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        z1 = z1.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        z2 = z2.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)

        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]

        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))
        z_cat = torch.cat((z1, z2), -1)

        channel_mask = self.mask_generator(z_cat)
        z, dcloss = self.frequency_transformer(z_cat, channel_mask)
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        z1 = self.head_f1(z1)
        z2 = self.head_f2(z2)

        complex_z = torch.complex(z1, z2)
        z = torch.fft.ifft(complex_z)
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'denorm')

        return z, complex_z.permute(0, 2, 1), dcloss

    def get_normalized_input(self, x):
        return self.revin_layer(x, 'transform')


# ===================== Model =====================
class Model(nn.Module):
    """
    CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.configs = configs

        self.backbone = CATCHBackbone(configs)

        self.auxi_loss_fn = FrequencyLoss(
            auxi_loss=getattr(configs, 'catch_auxi_loss', 'MAE'),
            auxi_type=getattr(configs, 'catch_auxi_type', 'complex'),
            auxi_mode=getattr(configs, 'catch_auxi_mode', 'fft'),
            module_first=getattr(configs, 'catch_module_first', True)
        )

        self.dc_lambda = getattr(configs, 'catch_dc_lambda', 0.005)
        self.auxi_lambda = getattr(configs, 'catch_auxi_lambda', 0.005)
        self.score_lambda = getattr(configs, 'catch_score_lambda', 0.05)

        infer_patch_size = getattr(configs, 'catch_inference_patch_size', getattr(configs, 'catch_patch_size', 16))
        infer_patch_stride = getattr(configs, 'catch_inference_patch_stride', getattr(configs, 'catch_patch_stride', 1))
        self.freq_criterion = FrequencyCriterion(
            patch_size=infer_patch_size,
            patch_stride=infer_patch_stride,
            win_size=self.seq_len
        )

    def anomaly_detection(self, x_enc):
        dec_out, _, _ = self.backbone(x_enc)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        return None

    def forward_with_loss(self, x_enc):
        output, output_complex, dcloss = self.backbone(x_enc)
        norm_input = self.backbone.get_normalized_input(x_enc)
        auxi_loss = self.auxi_loss_fn(output_complex, norm_input)
        return output, dcloss, auxi_loss

    def get_mask_generator_params(self):
        return self.backbone.mask_generator.parameters()

    def get_main_params(self):
        return [param for name, param in self.named_parameters() if 'mask_generator' not in name]

