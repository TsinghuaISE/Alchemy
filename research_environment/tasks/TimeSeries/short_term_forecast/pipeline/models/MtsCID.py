"""
MtsCID: Multivariate Time Series Anomaly Detection by Capturing Coarse-Grained
Intra- and Inter-Variate Dependencies

This file implements the MtsCID model for Time_Series_Library.
Components have been extracted to layers/ modules for better modularity and reusability.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import components from layers
from layers.MtsCID_Utils import (
    complex_operator,
    complex_einsum,
    complex_softmax,
    complex_dropout,
    harmonic_loss_compute,
)
from layers.MtsCID_Scheduler import PolynomialDecayLR
from layers.MtsCID_Losses import EntropyLoss, GatheringLoss
from layers.MtsCID_Normalization import RevIN
from layers.MtsCID_Attention import PositionalEmbedding, Attention, AttentionLayer
from layers.MtsCID_Conv import Inception_Block, Inception_Attention_Block
from layers.MtsCID_Memory import generate_rolling_matrix, create_memory_matrix
from layers.MtsCID_Metrics import _get_best_f1, ts_metrics_enhanced, point_adjustment

# Import for metrics
from sklearn import metrics  # noqa: E402


# ===================== Encoder / Embedding =====================

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super().__init__()
        self.attn_layer = attn
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        out, _ = self.attn_layer(x)
        y = complex_dropout(self.dropout, out)
        return y


class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, n_window=100, n_layers=1,
                 branch_layers=None, group_embedding='False', match_dimension='first',
                 kernel_size=None, multiscale_patch_size=None,
                 init_type='normal', gain=0.02, dropout=0.1):
        super().__init__()
        branch_layers = branch_layers or ['fc_linear', 'intra_fc_transformer']
        kernel_size = kernel_size or [5]
        multiscale_patch_size = multiscale_patch_size or [10, 20]

        self.window_size = n_window
        self.d_model = d_model
        self.n_layers = n_layers
        self.branch_layers = branch_layers
        self.group_embedding = group_embedding
        self.match_dimension = match_dimension
        self.kernel_size = kernel_size
        self.multiscale_patch_size = multiscale_patch_size

        component_network = ['real_part', 'imaginary_part']
        num_in_fc_networks = len(component_network)

        self.encoder_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])

        for i, e_layer in enumerate(branch_layers):
            if self.match_dimension == 'none':
                updated_in_dim = in_dim
                extended_dim = in_dim
            elif (i == 0 and self.match_dimension == 'first') or (len(branch_layers) < 2):
                updated_in_dim = in_dim
                extended_dim = d_model
            elif (i == 0) and (not self.match_dimension == 'first'):
                updated_in_dim = in_dim
                extended_dim = in_dim
            elif (i + 1 < len(branch_layers)) and (self.match_dimension == 'middle'):
                updated_in_dim = extended_dim
                extended_dim = d_model
            elif i + 1 == len(branch_layers):
                updated_in_dim = extended_dim
                extended_dim = d_model
            else:
                updated_in_dim = extended_dim
                extended_dim = extended_dim

            if 'conv1d' in e_layer or 'deconv1d' in e_layer:
                if self.group_embedding == 'False':
                    groups = 1
                elif extended_dim >= updated_in_dim and extended_dim % updated_in_dim == 0:
                    groups = updated_in_dim
                elif extended_dim < updated_in_dim and updated_in_dim % extended_dim == 0:
                    groups = extended_dim
                else:
                    groups = 1

            if e_layer == 'dropout':
                self.encoder_layers.append(nn.Dropout(p=dropout))
                self.norm_layers.append(nn.Identity())
            elif e_layer == 'fc_linear':
                self.encoder_layers.append(nn.ModuleList([nn.Linear(updated_in_dim, extended_dim, bias=False)
                                                         for _ in range(num_in_fc_networks)]))
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(extended_dim) for _ in range(num_in_fc_networks)]))
            elif e_layer == 'linear':
                self.encoder_layers.append(nn.ModuleList([nn.Linear(updated_in_dim, extended_dim, bias=False)
                                                          for _ in range(num_in_fc_networks)]))
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(extended_dim) for _ in range(num_in_fc_networks)]))
            elif e_layer == 'multiscale_conv1d':
                for _ in range(n_layers):
                    self.encoder_layers.append(Inception_Block(in_channels=updated_in_dim,
                                                               out_channels=extended_dim,
                                                               kernel_list=kernel_size,
                                                               groups=groups))
                    self.norm_layers.append(nn.ModuleList([nn.LayerNorm(self.window_size)
                                                           for _ in range(num_in_fc_networks)]))
            elif e_layer == 'inter_fc_transformer':
                w_model = self.window_size // 2 + 1
                attention_layer = AttentionLayer(w_size=extended_dim, d_model=w_model, n_heads=1, dropout=dropout)
                self.encoder_layers.append(nn.ModuleList([EncoderLayer(attn=attention_layer,
                                                                       d_model=w_model,
                                                                       d_ff=128,
                                                                       dropout=dropout,
                                                                       activation='gelu')
                                                          for _ in range(num_in_fc_networks)]))
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(self.window_size)
                                                       for _ in range(num_in_fc_networks)]))

            elif e_layer == 'intra_fc_transformer':
                w_model = self.window_size // 2 + 1
                attention_layer = AttentionLayer(w_size=w_model, d_model=extended_dim, n_heads=1, dropout=dropout)
                self.encoder_layers.append(nn.ModuleList([EncoderLayer(attn=attention_layer,
                                                                       d_model=extended_dim,
                                                                       d_ff=128,
                                                                       dropout=dropout,
                                                                       activation='gelu')
                                                          for _ in range(num_in_fc_networks)]))
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(self.window_size)
                                                       for _ in range(num_in_fc_networks)]))

            elif e_layer == 'multiscale_ts_attention':
                self.encoder_layers.append(Inception_Attention_Block(w_size=self.window_size,
                                                                     in_dim=extended_dim,
                                                                     d_model=extended_dim,
                                                                     patch_list=multiscale_patch_size))
                self.norm_layers.append(nn.Identity())
            else:
                raise ValueError(f'The specified model {e_layer} is not supported!')

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        B, L, _ = x.size()
        latent_list = []
        residual = None

        for i, (embedding_layer, norm_layer) in enumerate(zip(self.encoder_layers, self.norm_layers)):
            if self.branch_layers[i] not in ['linear', 'fc_linear', 'multiscale_ts_attention']:
                x = x.permute(0, 2, 1)

            if self.branch_layers[i] in ['multiscale_conv1d', 'multiscale_ts_attention']:
                x = complex_operator(embedding_layer, x)
            elif self.branch_layers[i] in ['fc_linear']:
                x = torch.fft.rfft(x, dim=-2)
                x = complex_operator(embedding_layer, x)
                x = torch.fft.irfft(x, dim=-2)
            elif self.branch_layers[i] in ['inter_fc_transformer']:
                x = torch.fft.rfft(x, dim=-1)
                x = complex_operator(embedding_layer, x)
                x = torch.fft.irfft(x, dim=-1)
            elif self.branch_layers[i] in ['intra_fc_transformer']:
                x = torch.fft.rfft(x, dim=-1)
                x = x.permute(0, 2, 1)
                x = complex_operator(embedding_layer, x)
                x = x.permute(0, 2, 1)
                x = torch.fft.irfft(x, dim=-1)
            else:
                x = complex_operator(embedding_layer, x)

            x = complex_operator(norm_layer, x)

            if self.branch_layers[i] not in ['linear', 'fc_linear', 'multiscale_ts_attention']:
                x = x.permute(0, 2, 1)

            latent_list.append(x)

            if residual is not None and x.shape == residual.shape and 'transformer' in self.branch_layers[i]:
                x += residual
            if self.branch_layers[i] in ['linear', 'fc_linear']:
                residual = x

        return x, latent_list


class InputEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, n_window, device, dropout=0.1, n_layers=1, use_pos_embedding='False',
                 group_embedding='False', kernel_size=5, init_type='kaiming', match_dimension='first',
                 branch_layers=None):
        super().__init__()
        self.device = device
        branch_layers = branch_layers or ['linear']
        self.token_embedding = TokenEmbedding(in_dim=in_dim, d_model=d_model, n_window=n_window,
                                              n_layers=n_layers, branch_layers=branch_layers,
                                              group_embedding=group_embedding, match_dimension=match_dimension,
                                              init_type=init_type, kernel_size=kernel_size,
                                              dropout=0.1)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.use_pos_embedding = use_pos_embedding
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.to(self.device)
        x, latent_list = self.token_embedding(x)
        if self.use_pos_embedding == 'True':
            x = x + self.pos_embedding(x).to(self.device)
        return self.dropout(x), latent_list


# ===================== Decoder =====================

class Decoder(nn.Module):
    def __init__(self, w_size, d_model, c_out, networks=None, n_layers=1,
                 group_embedding='False', kernel_size=None, patch_size=-1,
                 activation='gelu', dropout=0.0, device='cpu'):
        super().__init__()
        networks = networks or ['linear']
        kernel_size = kernel_size or [1]
        self.decoder = InputEmbedding(in_dim=d_model, d_model=c_out, n_window=w_size,
                                      dropout=dropout, n_layers=n_layers,
                                      branch_layers=networks,
                                      match_dimension='last',
                                      group_embedding=group_embedding,
                                      kernel_size=kernel_size, init_type='normal',
                                      device=device)

    def forward(self, x):
        return self.decoder(x)


# ===================== Core Model =====================

class DecoderWrapper(nn.Module):
    def __init__(self, w_size, d_model, c_out, networks, n_layers,
                 group_embedding, kernel_size, device):
        super().__init__()
        self.decoder = Decoder(w_size=w_size,
                               d_model=d_model,
                               c_out=c_out,
                               networks=networks,
                               n_layers=n_layers,
                               group_embedding=group_embedding,
                               kernel_size=kernel_size,
                               device=device)

    def forward(self, x):
        out, latent = self.decoder(x)
        return out, latent


class TransformerVar(nn.Module):
    DEFAULTS = {}

    def __init__(self, config, n_heads=1, d_ff=128, dropout=0.3, activation='gelu', gain=0.02):
        super().__init__()
        self.__dict__.update(TransformerVar.DEFAULTS, **config)

        branch1_group = self.branches_group_embedding.split('_')[0]
        branch2_group = self.branches_group_embedding.split('_')[1]

        branch1_dim = self.input_c if self.branch1_match_dimension == 'none' else self.d_model
        branch2_dim = self.input_c if self.branch2_match_dimension == 'none' else self.d_model

        self.encoder_branch1 = InputEmbedding(in_dim=self.input_c, d_model=branch1_dim, n_window=self.win_size,
                                              dropout=dropout, n_layers=self.encoder_layers,
                                              branch_layers=self.branch1_networks,
                                              match_dimension=self.branch1_match_dimension,
                                              group_embedding=branch1_group,
                                              kernel_size=self.multiscale_kernel_size, init_type=self.embedding_init,
                                              device=self.device)

        self.encoder_branch2 = InputEmbedding(in_dim=self.input_c, d_model=branch2_dim, n_window=self.win_size,
                                              dropout=dropout, n_layers=self.encoder_layers,
                                              branch_layers=self.branch2_networks,
                                              match_dimension=self.branch2_match_dimension,
                                              group_embedding=branch2_group,
                                              kernel_size=self.multiscale_kernel_size,
                                              init_type=self.embedding_init, device=self.device)

        self.activate_func = nn.GELU()
        self.dropout = nn.AlphaDropout(p=dropout)
        self.loss_func = nn.MSELoss(reduction='none')

        self.mem_R, self.mem_I = create_memory_matrix(N=branch2_dim,
                                                      L=self.win_size,
                                                      mem_type=self.memory_guided,
                                                      option='options2')

        branch1_out_dim = self.output_c if self.branch1_match_dimension == 'none' else self.d_model
        model_dim = branch1_out_dim

        self.weak_decoder = DecoderWrapper(w_size=self.win_size,
                                           d_model=model_dim,
                                           c_out=self.output_c,
                                           networks=self.decoder_networks,
                                           n_layers=self.decoder_layers,
                                           group_embedding=self.decoder_group_embedding,
                                           kernel_size=self.multiscale_kernel_size,
                                           device=self.device)

        if self.branch1_match_dimension == 'none':
            self.feature_prj = lambda x: x
        else:
            self.feature_prj = nn.Linear(branch1_out_dim, self.output_c)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.embedding_init == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif self.embedding_init == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif self.embedding_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.embedding_init == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                if self.embedding_init == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif self.embedding_init == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif self.embedding_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.embedding_init == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input_data, mode='train'):
        z1 = z2 = input_data
        t_query, t_latent_list = self.encoder_branch1(z1)
        i_query, _ = self.encoder_branch2(z2)

        mem = self.mem_R.T.to(self.device)
        attn = torch.einsum('blf,jl->bfj', i_query, self.mem_R.to(self.device).detach())
        attn = torch.softmax(attn / self.temperature, dim=-1)

        queries = i_query
        combined_z = t_query
        combined_z = self.feature_prj(combined_z)
        out, _ = self.weak_decoder(combined_z)

        return {"out": out, "queries": queries, "mem": mem, "attn": attn}


# ===================== Library Wrapper =====================

class Model(nn.Module):
    """
    Time_Series_Library compatible wrapper.
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len

        # MtsCID specific parameters
        self.temperature = getattr(configs, 'mtscid_temperature', 0.1)
        self.alpha = getattr(configs, 'mtscid_alpha', 1.0)
        self.aggregation = getattr(configs, 'mtscid_aggregation', 'normal_mean')
        self.warmup_epoch = getattr(configs, 'mtscid_warmup_epoch', 0)
        self.peak_lr = getattr(configs, 'mtscid_peak_lr', 2e-3)
        self.end_lr = getattr(configs, 'mtscid_end_lr', 5e-5)
        self.weight_decay = getattr(configs, 'mtscid_weight_decay', 5e-5)

        device = configs.device if hasattr(configs, 'device') else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        config_dict = self._build_config(configs, device)
        self.mtscid_model = TransformerVar(config_dict)

        self.entropy_loss = EntropyLoss()
        self.gathering_loss = GatheringLoss(reduction='none', memto_framework=True)
        self.mtscid_device = device

    def _build_config(self, configs, device):
        return {
            'input_c': configs.enc_in,
            'output_c': configs.c_out,
            'd_model': configs.d_model,
            'win_size': configs.seq_len,
            'encoder_layers': getattr(configs, 'e_layers', 1),
            'decoder_layers': getattr(configs, 'd_layers', 1),
            'temperature': getattr(configs, 'mtscid_temperature', 0.1),
            'device': device,
            'branch1_networks': getattr(configs, 'mtscid_branch1_networks',
                                        ['fc_linear', 'intra_fc_transformer', 'multiscale_ts_attention']),
            'branch1_match_dimension': getattr(configs, 'mtscid_branch1_match_dimension', 'first'),
            'branch2_networks': getattr(configs, 'mtscid_branch2_networks',
                                        ['multiscale_conv1d', 'inter_fc_transformer']),
            'branch2_match_dimension': getattr(configs, 'mtscid_branch2_match_dimension', 'first'),
            'decoder_networks': getattr(configs, 'mtscid_decoder_networks', ['linear']),
            'decoder_group_embedding': getattr(configs, 'mtscid_decoder_group_embedding', 'False'),
            'multiscale_patch_size': getattr(configs, 'mtscid_multiscale_patch_size', [10, 20]),
            'multiscale_kernel_size': getattr(configs, 'mtscid_multiscale_kernel_size', [5]),
            'branches_group_embedding': getattr(configs, 'mtscid_branches_group_embedding', 'False_False'),
            'memory_guided': getattr(configs, 'mtscid_memory_guided', 'sinusoid'),
            'embedding_init': getattr(configs, 'mtscid_embedding_init', 'normal'),
        }

    def anomaly_detection(self, x_enc):
        output_dict = self.mtscid_model(x_enc)
        return output_dict['out']

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        return None

    # ============= Custom helpers for exp_anomaly_detection =============
    def forward_with_loss(self, x_enc):
        output_dict = self.mtscid_model(x_enc)
        output = output_dict['out']
        attn = output_dict['attn']
        entropy_loss = self.entropy_loss(attn) if attn is not None else torch.tensor(0.0, device=output.device)
        return output, entropy_loss

    def (selcompute_anomaly_scoref, x_enc):
        output_dict = self.mtscid_model(x_enc)
        output = output_dict['out']
        queries = output_dict['queries']
        mem_items = output_dict['mem']

        rec_loss = F.mse_loss(x_enc, output, reduction='none')
        latent_score = torch.softmax(self.gathering_loss(queries, mem_items) / self.temperature, dim=-1)
        score = harmonic_loss_compute(rec_loss, latent_score, self.aggregation)
        return score

__all__ = [
    # 主模型类
    'Model',
    
    # 从 layers 重新导出的组件（保持向后兼容）
    'PolynomialDecayLR',
    'EntropyLoss',
    'GatheringLoss',
    'RevIN',
    'PositionalEmbedding',
    'Attention',
    'AttentionLayer',
    'Inception_Block',
    'Inception_Attention_Block',
    'create_memory_matrix',
    'generate_rolling_matrix',
    '_get_best_f1',
    'ts_metrics_enhanced',
    'point_adjustment',
    'complex_operator',
    'complex_einsum',
    'complex_softmax',
    'complex_dropout',
    'harmonic_loss_compute',
]