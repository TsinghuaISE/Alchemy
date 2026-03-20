import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============== 内联的 Embedding 类 ==============
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


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


# ============== 内联的 Inception_Block_1D（带groups参数）==============
class Inception_Block_1D(nn.Module):
    def __init__(self, in_channels, out_channels, group, num_kernels=6, init_weight=True):
        super(Inception_Block_1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i, groups=group)
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


# ============== SGN 核心模块 ==============
class PeriodMerging(nn.Module):
    def __init__(self, dim, group):
        super().__init__()
        self.group = group
        self.d_model = dim
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(2 * dim, dim)

    def forward(self, x):
        B, C, num, T = x.shape
        x = x.reshape(B, self.group, self.d_model, num, T)
        x = x.reshape(B * self.group, self.d_model, num, T)

        last_window = None
        if num <= 4:
            merged_windows = []
            for i in range(num - 1):
                merged_window = torch.cat([x[:, :, i, :], x[:, :, i + 1, :]], dim=1)
                merged_windows.append(merged_window)
            x_merged = torch.stack(merged_windows, dim=2)
            x_merged = x_merged.reshape(B * self.group, 2 * self.d_model, -1)
        else:
            if num % 2 == 1:
                last_window = x[:, :, -1, :]
                x = x[:, :, :-1, :]
                num -= 1
            x_even = x[:, :, 0::2, :]
            x_odd = x[:, :, 1::2, :]
            x_merged = torch.cat([x_even, x_odd], dim=1)
            x_merged = x_merged.view(B * self.group, 2 * self.d_model, -1)

        x_merged = x_merged.permute(0, 2, 1)
        x_reduced = self.reduction(x_merged)
        x_reduced = x_reduced.permute(0, 2, 1)
        x_reduced = x_reduced.reshape(B * self.group, self.d_model, -1, T)

        if last_window is not None:
            last_window = last_window.unsqueeze(2)
            x_reduced = torch.cat([x_reduced, last_window], dim=2)

        x_reduced = x_reduced.reshape(B * self.group, self.d_model, -1)
        x_reduced = x_reduced.permute(0, 2, 1)
        x_normalized = self.norm(x_reduced)
        x_normalized = x_normalized.permute(0, 2, 1)
        x_normalized = x_normalized.reshape(B * self.group, self.d_model, -1, T)
        x_normalized = x_normalized.reshape(B, self.group * self.d_model, -1, T)

        return x_normalized


class DilatedConvBlock(nn.Module):
    def __init__(self, kernel_size, group, d_model, ratio, num_kernel=5, dropout=0.1):
        super().__init__()
        self.group = group
        self.d_model = d_model

        self.conv = nn.Sequential(
            Inception_Block_1D(group * d_model, group * d_model * ratio, group=group * d_model, num_kernels=num_kernel),
            nn.GELU(),
            Inception_Block_1D(group * d_model * ratio, group * d_model, group=group * d_model, num_kernels=num_kernel)
        )
        self.norm = nn.BatchNorm1d(d_model)

        self.ffn1pw1 = nn.Conv1d(group * d_model, group * d_model * ratio, kernel_size=1, groups=group)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(group * d_model * ratio, group * d_model, kernel_size=1, groups=group)
        self.ffn1drop1 = nn.Dropout(dropout)
        self.ffn1drop2 = nn.Dropout(dropout)

        self.ffn2pw1 = nn.Conv1d(group * d_model, group * d_model * ratio, kernel_size=1, groups=d_model)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(group * d_model * ratio, group * d_model, kernel_size=1, groups=d_model)
        self.ffn2drop1 = nn.Dropout(dropout)
        self.ffn2drop2 = nn.Dropout(dropout)

    def forward(self, x):
        B, C, num, T = x.shape
        input = x

        x = x.permute(0, 2, 1, 3).reshape(B * num, C, T)
        x = self.conv(x)
        x = x.reshape(B * num, self.group, self.d_model, T)
        x = x.reshape(B * num * self.group, self.d_model, T)
        x = self.norm(x)
        x = x.reshape(B * num, self.group, self.d_model, T)
        x = x.reshape(B * num, self.group * self.d_model, T)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B * num, self.group, self.d_model, T)

        x = x.permute(0, 2, 1, 3).reshape(B * num, self.d_model * self.group, T)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B * num, self.d_model, self.group, T)
        x = x.permute(0, 2, 1, 3)

        x = x.reshape(B * num, self.group, self.d_model, T)
        x = x.reshape(B * num, C, T)
        x = x.reshape(B, num, C, T).permute(0, 2, 1, 3)

        return x + input


class DilatedConvEncoder(nn.Module):
    def __init__(self, block_num, kernel_size, d_model, ratio_ffn, num_kernels, group, dropout):
        super().__init__()
        self.net = nn.Sequential(*[
            DilatedConvBlock(
                kernel_size=kernel_size,
                group=group,
                d_model=d_model,
                ratio=ratio_ffn,
                num_kernel=num_kernels,
                dropout=dropout,
            ) for _ in range(block_num)
        ])

    def forward(self, x):
        return self.net(x)


class TemporalSwinTransformerBlock(nn.Module):
    def __init__(self, shift_size, attn_layer=None):
        super().__init__()
        self.tcn = attn_layer
        self.shift_size = shift_size

    def forward(self, x):
        B, C, num, T = x.shape
        x = x.reshape(B, C, num * T)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-T // 2,), dims=(2,))
        else:
            shifted_x = x

        x_windows = shifted_x.reshape(B, C, num, T)
        x_windows = self.tcn(x_windows)
        shifted_x = x_windows.reshape(B, C, num * T)

        if self.shift_size > 0:
            x_new = torch.roll(shifted_x, shifts=(T // 2,), dims=(2,))
            x_new[:, :, :T // 2] = x[:, :, :T // 2]
        else:
            x_new = shifted_x

        x_new = x_new.reshape(B, C, num, T)
        return x_new


class TemporalBasicLayer(nn.Module):
    def __init__(self, dim, depth, group, attn_layer=None, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            TemporalSwinTransformerBlock(shift_size=i % 2, attn_layer=attn_layer)
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(dim=dim, group=group)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class GumbelGroupEmbedding(nn.Module):
    def __init__(self, C, d_model, max_groups=8, groups_matrix=None,
                 temperature_init=1.0, temperature_min=0.1, anneal_rate=0.0003, sim_loss_weight=0.1):
        super().__init__()
        self.C = C
        self.G = max_groups
        self.d_model = d_model
        self.sim_loss_weight = sim_loss_weight

        # 关键修改：从参数传入分组矩阵，而非从文件加载
        if groups_matrix is not None:
            groups_matrix_tensor = torch.tensor(groups_matrix, dtype=torch.float32)
        else:
            # 如果没有传入，使用随机初始化
            groups_matrix_tensor = torch.randn(C, max_groups) * 0.1
        self.assign_logits = nn.Parameter(groups_matrix_tensor)

        self.group_embeds = DataEmbedding(1, self.d_model)

        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.anneal_rate = anneal_rate

        self.register_buffer("temperature", torch.tensor(temperature_init, dtype=torch.float32))
        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))

    def gumbel_softmax(self, logits, hard=False):
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            y = (logits + gumbel_noise) / self.temperature
        else:
            y = logits / self.temperature

        y = F.softmax(y, dim=-1)
        if hard:
            y_hard = F.one_hot(y.argmax(dim=-1), num_classes=self.G).float()
            return (y_hard - y).detach() + y
        return y

    def update_temperature(self):
        self.training_step += 1
        new_temp = self.temperature_init * np.exp(-self.anneal_rate * self.training_step.cpu().item())
        self.temperature = torch.tensor(max(self.temperature_min, new_temp), dtype=torch.float32)

    def compute_similarity_matrix(self, x):
        B, T, C = x.shape
        x = x.permute(2, 0, 1).reshape(C, B * T)
        x = F.normalize(x, dim=1)
        sim = torch.matmul(x, x.t())
        return sim

    def similarity_regularization(self, assign_probs, sim_matrix):
        sim_matrix = (sim_matrix + 1) / 2
        diff = assign_probs.unsqueeze(1) - assign_probs.unsqueeze(0)
        loss = (sim_matrix.unsqueeze(-1) * diff.pow(2)).sum() / (self.C * self.C)
        return loss

    def forward(self, x, return_info=False, hard_assign=False):
        if not self.training:
            self.temperature = torch.tensor(self.temperature_min, dtype=torch.float32)
            hard_assign = True
        else:
            self.update_temperature()

        B, T, C = x.shape
        assign_probs = self.gumbel_softmax(self.assign_logits, hard=hard_assign)

        x_grouped = torch.einsum('btc,cg->btg', x, assign_probs)
        x_grouped = x_grouped.permute(0, 2, 1).unsqueeze(-1)
        x_grouped = x_grouped.reshape(B * self.G, T, 1)
        x_grouped = self.group_embeds(x_grouped, None)
        x_grouped = x_grouped.reshape(B, self.G, T, self.d_model)

        x_out = x_grouped.permute(0, 1, 3, 2).reshape(B, self.G * self.d_model, T)

        sim_loss = None
        if self.training and self.sim_loss_weight > 0:
            sim_matrix = self.compute_similarity_matrix(x.detach())
            sim_loss = self.similarity_regularization(assign_probs, sim_matrix)

        if return_info:
            group_usage = assign_probs.mean(dim=0).detach().cpu().numpy()
            group_id = assign_probs.argmax(dim=1).detach().cpu().numpy()
            return {"out": x_out, "assign_probs": assign_probs.detach(), "group_id": group_id, "group_usage": group_usage}
        return x_out, sim_loss


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.num_classes = configs.num_class
        self.num_layers = configs.e_layers
        self.period = configs.period
        self.depths = configs.depths
        self.seq_len = configs.seq_len
        self.group_num = configs.num_groups

        # 关键修改：传入 groups_matrix
        groups_matrix = getattr(configs, 'groups_matrix', None)
        self.enc_embedding_group = GumbelGroupEmbedding(
            configs.enc_in, configs.d_model, max_groups=self.group_num, groups_matrix=groups_matrix
        )
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = TemporalBasicLayer(
                dim=configs.d_model,
                depth=self.depths[i_layer],
                group=self.group_num,
                downsample=PeriodMerging if (i_layer < self.num_layers - 1) else None,
                attn_layer=DilatedConvEncoder(
                    block_num=configs.block_num,
                    kernel_size=configs.kernel_size,
                    d_model=configs.d_model,
                    ratio_ffn=configs.mlp_ratio,
                    num_kernels=configs.num_kernels,
                    group=self.group_num,
                    dropout=configs.dropout
                )
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(configs.d_model * self.group_num)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(configs.d_model * self.group_num, configs.num_class, bias=True)

    def classification(self, x, x_mark_enc):
        B, T, C = x.shape

        if self.group_num > 1:
            x, embeddings_for_loss = self.enc_embedding_group(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.enc_embedding(x, None)
            embeddings_for_loss = torch.tensor(0.0, device=x.device)

        period = self.period

        if self.seq_len % self.period != 0:
            length = ((self.seq_len // period) + 1) * period
            padding = torch.zeros([x.shape[0], length - self.seq_len, x.shape[2]]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = self.seq_len
            out = x

        out = out.reshape(B, length // period, period, x.shape[2]).permute(0, 3, 1, 2).contiguous()

        for layer in self.layers:
            out = layer(out)

        out = out.reshape(B, out.shape[1], -1).permute(0, 2, 1)

        x = self.norm(out)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x, embeddings_for_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None

