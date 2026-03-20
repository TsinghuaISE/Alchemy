"""
InterpGN: Interpretability Gated Networks for Time-Series Classification
Paper: Shedding Light on Time Series Classification using Interpretability Gated Networks (ICLR 2025)
This file consolidates all dependencies so that no extra modules are placed under layers/.
"""

import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange
from torch.autograd import Function


# ==================== Data classes ==================== #
@dataclass
class ModelInfo:
    d: torch.Tensor = None
    p: torch.Tensor = None
    eta: torch.Tensor = None
    shapelet_preds: torch.Tensor = None
    dnn_preds: torch.Tensor = None
    preds: torch.Tensor = None
    loss: torch.Tensor = None


# ==================== Shapelet distance utilities ==================== #
def pearson_corrcoef(x, y, eps=1e-8):
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    numerator = torch.sum(x_centered * y_centered, dim=-1)
    denominator = torch.sqrt(torch.sum(x_centered ** 2, dim=-1) * torch.sum(y_centered ** 2, dim=-1))
    denominator = denominator + eps
    return numerator / denominator


class ShapeletDistanceFunc(Function):
    """Memory efficient shapelet distance."""

    @staticmethod
    def forward(ctx, x, s):
        ctx.save_for_backward(x, s)
        output = torch.cat([
            (s - x[:, :, i:i + s.shape[-1]].unsqueeze(1)).pow(2).mean(-1).unsqueeze(1)
            for i in range(x.shape[-1] - s.shape[-1] + 1)
        ], dim=1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, s = ctx.saved_tensors
        grad_s = torch.zeros_like(s)
        for i in range(grad_output.shape[1]):
            g = grad_output[:, i, :, :].unsqueeze(-1).expand(-1, -1, -1, s.shape[-1])
            xn = x[:, :, i:i + s.shape[-1]].unsqueeze(1)
            grad_s += (g * (s - xn)).sum(0)
        grad_s = grad_s * 2 / s.shape[-1]
        return torch.zeros_like(x), grad_s


def ShapeletDistance(x, s):
    return ShapeletDistanceFunc.apply(x, s)


# ==================== Shapelet modules ==================== #
class Shapelet(nn.Module):
    def __init__(self, dim_data, shapelet_len, num_shapelet=10, stride=1, eps=1.,
                 distance_func='euclidean', memory_efficient=False):
        super().__init__()
        self.dim = dim_data
        self.length = shapelet_len
        self.n = num_shapelet
        self.stride = stride
        self.distance_func = distance_func
        self.memory_efficient = memory_efficient
        self.weights = nn.Parameter(torch.normal(0, 1, (self.n, self.dim, self.length)), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        x = x.unfold(2, self.length, self.stride)
        x = rearrange(x, 'b m t l -> b t 1 m l')

        if self.distance_func == 'cosine':
            d = nn.functional.cosine_similarity(x, self.weights, dim=-1)
            d = torch.ones_like(d) - d
        elif self.distance_func == 'pearson':
            d = pearson_corrcoef(x, self.weights)
            d = torch.ones_like(d) - d
        else:
            if self.memory_efficient:
                d = ShapeletDistance(x, self.weights)
            else:
                d = (x - self.weights).abs().mean(dim=-1)

        p = torch.exp(-torch.pow(self.eps * d, 2))
        hard = torch.zeros_like(p).scatter_(1, p.argmax(dim=1, keepdim=True), 1.)
        soft = torch.softmax(p, dim=1)
        onehot_max = hard + soft - soft.detach()
        max_p = torch.sum(onehot_max * p, dim=1)

        return max_p.flatten(start_dim=1), d.min(dim=1).values.flatten(start_dim=1)

    def derivative(self):
        return torch.diff(self.weights, dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, dim_feature, dim_attn):
        super().__init__()
        self.q_proj = nn.Linear(1, dim_attn)
        self.k_proj = nn.Linear(1, dim_attn)
        self.pos_embed = nn.Embedding(num_embeddings=dim_feature, embedding_dim=dim_attn)

    def forward(self, x):
        pos_embed = self.pos_embed(torch.arange(x.shape[1], device=x.device))
        q = self.q_proj(x.unsqueeze(-1)) + pos_embed
        k = self.k_proj(x.unsqueeze(-1)) + pos_embed
        x = F.scaled_dot_product_attention(q, k, x.unsqueeze(-1))
        return x.squeeze(-1)


# ==================== Shape Bottleneck Model ==================== #
class ShapeBottleneckModel(nn.Module):
    def __init__(self, configs, num_shapelet=[5, 5, 5, 5], shapelet_len=[0.1, 0.2, 0.3, 0.5]):
        super().__init__()
        self.num_shapelet = num_shapelet
        self.num_channel = configs.enc_in
        self.num_class = configs.num_class
        self.shapelet_len = []
        self.normalize = True
        self.configs = configs

        self.shapelets = nn.ModuleList()
        for i, l in enumerate(shapelet_len):
            sl = max(3, np.ceil(l * configs.seq_len).astype(int))
            self.shapelets.append(
                Shapelet(
                    dim_data=self.num_channel,
                    shapelet_len=sl,
                    num_shapelet=num_shapelet[i],
                    eps=configs.epsilon,
                    distance_func=configs.distance_func,
                    memory_efficient=configs.memory_efficient,
                    stride=1 if configs.seq_len < 3000 else max(1, int(np.log2(sl)))
                )
            )
            self.shapelet_len.append(sl)

        self.total_shapelets = sum(num_shapelet) * self.num_channel

        if configs.sbm_cls == 'linear':
            self.output_layer = nn.Linear(self.total_shapelets, self.num_class, bias=False)
        elif configs.sbm_cls == 'bilinear':
            self.output_layer = nn.Linear(self.total_shapelets, self.num_class, bias=False)
            self.output_bilinear = nn.Bilinear(self.total_shapelets, self.total_shapelets, self.num_class, bias=False)
        elif configs.sbm_cls == 'attention':
            self.attention = SelfAttention(self.total_shapelets, 16)
            self.output_layer = nn.Linear(self.total_shapelets, self.num_class, bias=False)

        self.dropout = nn.Dropout(p=configs.dropout)
        self.distance_func_module = nn.PairwiseDistance(p=2)
        self.lambda_reg = configs.lambda_reg
        self.lambda_div = configs.lambda_div

    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b t c -> b c t')
        x = (x - x.mean(dim=-1, keepdims=True)) / (x.std(dim=-1, keepdims=True) + 1e-8)

        shapelet_probs, shapelet_dists = [], []
        for shapelet in self.shapelets:
            p, d = shapelet(x)
            shapelet_probs.append(p)
            shapelet_dists.append(d)
        shapelet_probs = torch.cat(shapelet_probs, dim=-1)
        shapelet_dists = torch.cat(shapelet_dists, dim=-1)

        if self.configs.sbm_cls == 'linear':
            out = self.output_layer(self.dropout(shapelet_probs))
        elif self.configs.sbm_cls == 'bilinear':
            out = self.output_layer(self.dropout(shapelet_probs)) + \
                  self.output_bilinear(self.dropout(shapelet_probs), self.dropout(shapelet_probs))
        elif self.configs.sbm_cls == 'attention':
            out = self.attention(shapelet_probs)
            out = self.output_layer(self.dropout(out))

        return out, ModelInfo(
            d=shapelet_dists,
            p=shapelet_probs,
            shapelet_preds=out,
            preds=out,
            loss=self.loss().unsqueeze(0)
        )

    def step(self):
        with torch.no_grad():
            self.output_layer.weight.clamp_(0.)

    def loss(self):
        loss_reg = self.output_layer.weight.abs().mean()
        loss_div = self.diversity() if self.lambda_div > 0. else 0.
        return loss_reg * self.lambda_reg + loss_div * self.lambda_div

    def diversity(self):
        loss = 0.
        for s in self.shapelets:
            sh = s.weights.permute(1, 0, 2)
            dist = self.distance_func_module(sh.unsqueeze(1), sh.unsqueeze(2))
            mask = torch.ones_like(dist) - torch.eye(sh.shape[1], device=dist.device).unsqueeze(0)
            loss += (torch.exp(-dist) * mask).mean()
        return loss

    def get_shapelets(self):
        shapelets = []
        for s in self.shapelets:
            for k in range(s.weights.data.shape[0]):
                for c in range(s.weights.data.shape[1]):
                    shapelets.append((s.weights.data[k, c, :].cpu().numpy(), c))
        return shapelets


# ==================== DNN components ==================== #
class FullyConvNetwork(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.seq_len <= 10:
            self.block1 = nn.Sequential(nn.Conv1d(configs.enc_in, 128, 3), nn.BatchNorm1d(128), nn.ReLU())
            self.block2 = nn.Sequential(nn.Conv1d(128, 256, 3), nn.BatchNorm1d(256), nn.ReLU())
            self.block3 = nn.Sequential(nn.Conv1d(256, 128, 2), nn.BatchNorm1d(128), nn.ReLU())
        else:
            self.block1 = nn.Sequential(nn.Conv1d(configs.enc_in, 128, 8), nn.BatchNorm1d(128), nn.ReLU())
            self.block2 = nn.Sequential(nn.Conv1d(128, 256, 5), nn.BatchNorm1d(256), nn.ReLU())
            self.block3 = nn.Sequential(nn.Conv1d(256, 128, 3), nn.BatchNorm1d(128), nn.ReLU())
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, configs.num_class)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = rearrange(x, 'b t c -> b c t')
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pooling(x)
        x = self.fc(x.flatten(start_dim=1))
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class InterpGN_ResNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(configs.enc_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 * BasicBlock.expansion, configs.num_class)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = rearrange(x, 'b t c -> b c t')
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.fc(torch.flatten(self.avgpool(x), 1))
        return x


def get_dnn_model(configs):
    dnn_type = getattr(configs, 'dnn_type', 'FCN')
    if dnn_type == 'FCN':
        return FullyConvNetwork(configs)
    if dnn_type == 'ResNet':
        return InterpGN_ResNet(configs)
    if dnn_type in ['TimesNet', 'Transformer', 'PatchTST']:
        module = importlib.import_module(f'models.{dnn_type}')
        return module.Model(configs)
    raise ValueError(f"Unknown dnn_type: {dnn_type}")


# ==================== InterpGN main model ==================== #
class Model(nn.Module):
    """
    InterpGN combines a shapelet bottleneck with a deep model and gates them by Gini-based confidence.
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        shapelet_lengths = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
        num_shapelet = [getattr(configs, 'num_shapelet', 10)] * len(shapelet_lengths)

        self.sbm = ShapeBottleneckModel(
            configs=configs,
            num_shapelet=num_shapelet,
            shapelet_len=shapelet_lengths
        )
        self.deep_model = get_dnn_model(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, gating_value=None):
        if gating_value is None:
            gating_value = getattr(self.configs, 'gating_value', None)

        sbm_out, model_info = self.sbm(x)
        deep_out = self.deep_model(x, x_mark_enc, x_dec, x_mark_dec, mask)

        p = nn.functional.softmax(sbm_out, dim=-1)
        c = sbm_out.shape[-1]
        gini = p.pow(2).sum(-1, keepdim=True)
        sbm_util = (c * gini - 1) / (c - 1)

        if gating_value is not None:
            gate_mask = (sbm_util > gating_value).float()
            sbm_util = torch.ones_like(sbm_util) * gate_mask + sbm_util * (1 - gate_mask)

        deep_util = torch.ones_like(sbm_util) - sbm_util
        output = sbm_util * sbm_out + deep_util * deep_out

        return output, ModelInfo(
            d=model_info.d,
            p=model_info.p,
            eta=sbm_util,
            shapelet_preds=sbm_out,
            dnn_preds=deep_out,
            preds=output,
            loss=self.loss().unsqueeze(0)
        )

    def loss(self):
        return self.sbm.loss()

    def step(self):
        self.sbm.step()

