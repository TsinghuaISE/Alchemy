import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Attention Blocks ==================== #
class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout: float = 0.0):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        bs, n_heads, q_len, d_k = q.size()
        scale = 1.0 / math.sqrt(d_k)
        attn_scores = torch.matmul(q, k) * scale  # [bs, n_heads, q_len, k_len]
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output.contiguous(), attn_weights


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.attn = attention
        self.proj = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q, K, V, attn_mask):
        bs, _, _ = Q.size()
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).permute(0, 2, 1, 3)

        out, attn_weights = self.attn(q_s, k_s, v_s, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.n_heads * self.d_v)
        out = self.proj(out)
        attn_weights = attn_weights.mean(dim=1)  # [bs, q_len, k_len]
        return out, attn_weights


# ==================== Encoder / Decoder Blocks ==================== #
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous: bool = False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, norm: str = "batchnorm", dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        x_, attn_weights = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(x_)
        x = self.norm1(x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn_weights


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, attn_weights = layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, norm: str = "batchnorm", dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm3 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x_, x_attn_weights = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x + self.dropout(x_)
        x = self.norm1(x)

        x_, cross_attn_weights = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x + self.dropout(x_)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y), x_attn_weights, cross_attn_weights


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x, x_attn_weights, cross_attn_weights = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, x_attn_weights, cross_attn_weights


# ==================== Context Blocks ==================== #
class ContextNet(nn.Module):
    def __init__(self, router, querys, extractor):
        super().__init__()
        self.router = router
        self.querys = querys
        self.extractor = extractor

    def forward(self, x_enc, local_repr, mask=None):
        q_indices = self.router(x_enc)
        q = torch.einsum("bn,nqd->bqd", q_indices, self.querys)  # [bs, query_len, d_model]
        query_latent_distances, context = self.extractor(q, local_repr, mask)
        return query_latent_distances, context


class Router(nn.Module):
    def __init__(self, seq_len, n_vars, n_query, topk: int = 5):
        super().__init__()
        self.k = topk
        self.fc = nn.Sequential(nn.Flatten(-2), nn.Linear(seq_len * n_vars, n_query))

    def forward(self, x):
        bs, t, c = x.shape
        x_freq = torch.fft.rfft(x, dim=1, n=t)
        _, indices = torch.topk(x_freq.abs(), self.k, dim=1)  # [bs, k, c]
        mesh_a, mesh_b = torch.meshgrid(
            torch.arange(x_freq.size(0), device=x_freq.device),
            torch.arange(x_freq.size(2), device=x_freq.device),
            indexing="ij",
        )
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        mask = torch.zeros_like(x_freq, dtype=torch.bool)
        mask[index_tuple] = True
        x_freq[~mask] = torch.tensor(0.0 + 0j, device=x_freq.device)
        x = torch.fft.irfft(x_freq, dim=1, n=t)
        logits = self.fc(x)  # [bs, n_query]
        q_indices = F.gumbel_softmax(logits, tau=1, hard=True)
        return q_indices


class ExtractorLayer(nn.Module):
    def __init__(self, cross_attention, d_model, d_ff=None, norm: str = "batchnorm", dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, q, local_repr, mask=None):
        q = q + self.dropout(self.cross_attention(q, local_repr, local_repr, attn_mask=mask)[0])
        q = self.norm1(q)
        y = self.dropout(self.activation(self.conv1(q.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(q + y)


class Extractor(nn.Module):
    def __init__(self, layers, context_size: int = 64, query_len: int = 5, d_model: int = 128, decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self.context_size = context_size
        self.query_len = query_len
        self.d_model = d_model
        self.register_buffer("context", torch.randn(context_size, query_len, d_model))
        self.register_buffer("ema_count", torch.ones(context_size))
        self.register_buffer("ema_dw", torch.zeros(context_size, query_len, d_model))
        self.decay = decay
        self.epsilon = epsilon
        self.extractor = nn.ModuleList(layers)

    def update_context(self, q):
        _, q_len, d = q.shape
        q_flat = q.reshape(-1, q_len * d)
        g_flat = self.context.reshape(-1, q_len * d)
        N, D = g_flat.shape

        distances = (
            torch.sum(q_flat ** 2, dim=1, keepdim=True)
            + torch.sum(g_flat ** 2, dim=1)
            - 2 * torch.matmul(q_flat, g_flat.t())
        )
        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, N).float()
        q_context = torch.einsum("bn,nqd->bqd", [encodings, self.context])
        q_hat = torch.einsum("bn,bqd->nqd", [encodings, q])

        query_latent_distances = torch.mean(F.mse_loss(q_context.detach(), q, reduction="none"), dim=(1, 2))

        if self.training:
            with torch.no_grad():
                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
                n = torch.sum(self.ema_count)
                self.ema_count = (self.ema_count + self.epsilon) / (n + D * self.epsilon) * n

                dw = torch.einsum("bn,bqd->nqd", [encodings, q])
                self.ema_dw = self.decay * self.ema_dw + (1 - self.decay) * dw
                self.context = self.ema_dw / self.ema_count.unsqueeze(-1).unsqueeze(-1)
        return query_latent_distances, q_hat

    def concat_context(self, context):
        return context.view(-1, self.d_model)

    def forward(self, q, local_repr, mask=None):
        for layer in self.extractor:
            q = layer(q, local_repr, mask)
        query_latent_distances, q_hat = self.update_context(q)
        context = self.concat_context(q_hat + self.context.detach() - q_hat.detach())
        return query_latent_distances, context


# ==================== Multi-scale Utils ==================== #
class MS_Utils(nn.Module):
    def __init__(self, kernels, method: str = "interval_sampling"):
        super().__init__()
        self.kernels = kernels
        self.method = method

    def concat_sampling_list(self, x_enc_sampling_list):
        return torch.concat(x_enc_sampling_list, dim=1)

    def split_2_list(self, ms_x_enc, ms_t_lens, mode: str = "encoder"):
        if mode == "encoder":
            return list(torch.split(ms_x_enc, split_size_or_sections=ms_t_lens[:-1], dim=1))
        elif mode == "decoder":
            return list(torch.split(ms_x_enc, split_size_or_sections=ms_t_lens[1:], dim=1))

    def scale_ind_mask(self, ms_t_lens):
        L = sum(t_len for t_len in ms_t_lens[:-1])
        d = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(ms_t_lens[:-1])]).view(1, L, 1)
        dT = d.transpose(1, 2)
        return torch.where(d == dT, 0.0, -torch.inf).reshape(1, 1, L, L).contiguous().bool()

    def next_scale_mask(self, ms_t_lens):
        L = sum(t_len for t_len in ms_t_lens[1:])
        d = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(ms_t_lens[1:])]).view(1, L, 1)
        dT = d.transpose(1, 2)
        return torch.where(d >= dT, 0.0, -torch.inf).reshape(1, 1, L, L).contiguous().bool()

    def down(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)  # [b, c, t]
        x_enc_sampling_list = []
        for kernel in self.kernels:
            pad_x_enc = F.pad(x_enc, pad=(0, kernel - 1), mode="replicate")
            x_enc_i = pad_x_enc.unfold(dimension=-1, size=kernel, step=kernel)  # [b, c, t_i, kernel]
            if self.method == "average_pooling":
                x_enc_i = torch.mean(x_enc_i, dim=-1)
            elif self.method == "interval_sampling":
                x_enc_i = x_enc_i[:, :, :, 0]
            x_enc_sampling_list.append(x_enc_i.permute(0, 2, 1))  # [b, t_i, c]
        return x_enc_sampling_list

    def up(self, x_enc_sampling_list, ms_t_lens):
        for i in range(len(ms_t_lens) - 1):
            x_enc = x_enc_sampling_list[i].permute(0, 2, 1)
            up_x_enc = F.interpolate(x_enc, size=ms_t_lens[i + 1], mode="nearest").permute(0, 2, 1)
            x_enc_sampling_list[i] = up_x_enc
        return x_enc_sampling_list

    @torch.no_grad()
    def _dummy_forward(self, input_len):
        dummy_x = torch.ones((1, input_len, 1))
        dummy_sampling_list = self.down(dummy_x)
        ms_t_lens = []
        for i in range(len(dummy_sampling_list)):
            ms_t_lens.append(dummy_sampling_list[i].shape[1])
        ms_t_lens.append(input_len)
        return ms_t_lens

    def forward(self, x_enc):
        return self.down(x_enc)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, t_len):
        return self.pe[:, :t_len]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def _dummy_forward(self, input_lens):
        ms_p_lens = []
        for input_len in input_lens:
            dummy_x = torch.ones((1, 1, input_len))
            dummy_x = self.padding_patch_layer(dummy_x)
            dummy_x = dummy_x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            ms_p_lens.append(dummy_x.shape[2])
        return ms_p_lens

    def forward(self, x):
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_patch = x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), x_patch


# ==================== Main Model ==================== #
class Model(nn.Module):
    """CrossAD model for time_series (faithful to original)."""

    def __init__(self, configs):
        super().__init__()
        seq_len = configs.seq_len
        patch_len = getattr(configs, "crossad_patch_len", getattr(configs, "patch_len", 6))
        d_model = configs.d_model
        ms_kernels = getattr(configs, "crossad_ms_kernels", getattr(configs, "ms_kernels", [16, 8, 4, 2]))
        ms_method = getattr(configs, "crossad_ms_method", getattr(configs, "ms_method", "average_pooling"))
        n_heads = configs.n_heads
        e_layers = configs.e_layers
        d_layers = configs.d_layers
        m_layers = getattr(configs, "crossad_m_layers", getattr(configs, "m_layers", 2))
        d_ff = configs.d_ff if configs.d_ff else 4 * d_model
        attn_dropout = getattr(configs, "crossad_attn_dropout", getattr(configs, "attn_dropout", 0.1))
        proj_dropout = getattr(configs, "crossad_proj_dropout", getattr(configs, "proj_dropout", 0.1))
        ff_dropout = getattr(configs, "crossad_ff_dropout", getattr(configs, "ff_dropout", 0.1))
        norm = getattr(configs, "crossad_norm", getattr(configs, "norm", "layernorm"))
        activation = configs.activation
        topk = getattr(configs, "crossad_topk", getattr(configs, "topk", 10))
        n_query = getattr(configs, "crossad_n_query", getattr(configs, "n_query", 5))
        query_len = getattr(configs, "crossad_query_len", getattr(configs, "query_len", 5))
        bank_size = getattr(configs, "crossad_bank_size", getattr(configs, "bank_size", 32))
        decay = getattr(configs, "crossad_decay", getattr(configs, "decay", 0.95))
        epsilon = getattr(configs, "crossad_epsilon", getattr(configs, "epsilon", 1e-5))

        self.seq_len = seq_len
        self.n_scales = len(ms_kernels)
        self.ms_utils = MS_Utils(ms_kernels, ms_method)
        self.pos_embedding = PositionalEmbedding(d_model)
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len=patch_len, stride=patch_len, padding=(patch_len - 1), dropout=0.0
        )
        self.ms_t_lens = self.ms_utils._dummy_forward(seq_len)
        self.ms_p_lens = self.patch_embedding._dummy_forward(self.ms_t_lens)
        self.ms_t_lens_ = [pn * patch_len for pn in self.ms_p_lens]

        # Key fix: avoid hard-coded .cuda(); buffers follow model device
        self.register_buffer("scale_ind_mask", self.ms_utils.scale_ind_mask(self.ms_p_lens))
        self.register_buffer("next_scale_mask", self.ms_utils.next_scale_mask(self.ms_p_lens))

        if "batch" in norm.lower():
            encoder_norm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            decoder_norm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            encoder_norm = nn.LayerNorm(d_model)
            decoder_norm = nn.LayerNorm(d_model)

        self.encoder = Encoder(
            layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        ScaledDotProductAttention(attn_dropout=attn_dropout),
                        d_model=d_model,
                        n_heads=n_heads,
                        proj_dropout=proj_dropout,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    norm=norm,
                    dropout=ff_dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=encoder_norm,
        )
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    self_attention=AttentionLayer(
                        ScaledDotProductAttention(attn_dropout=attn_dropout),
                        d_model=d_model,
                        n_heads=n_heads,
                        proj_dropout=proj_dropout,
                    ),
                    cross_attention=AttentionLayer(
                        ScaledDotProductAttention(attn_dropout=attn_dropout),
                        d_model=d_model,
                        n_heads=n_heads,
                        proj_dropout=proj_dropout,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    norm=norm,
                    dropout=ff_dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=decoder_norm,
            projection=nn.Sequential(nn.Linear(d_model, patch_len), nn.Flatten(-2)),
        )
        self.context_net = ContextNet(
            router=Router(seq_len=self.ms_t_lens[-1], n_vars=1, n_query=n_query, topk=topk),
            querys=nn.Parameter(torch.randn(n_query, query_len, d_model)),
            extractor=Extractor(
                layers=[
                    ExtractorLayer(
                        cross_attention=AttentionLayer(
                            ScaledDotProductAttention(attn_dropout=attn_dropout),
                            d_model=d_model,
                            n_heads=n_heads,
                            proj_dropout=proj_dropout,
                        ),
                        d_model=d_model,
                        d_ff=d_ff,
                        norm=norm,
                        dropout=ff_dropout,
                        activation=activation,
                    )
                    for _ in range(m_layers)
                ],
                context_size=bank_size,
                query_len=query_len,
                d_model=d_model,
                decay=decay,
                epsilon=epsilon,
            ),
        )

    def _forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        bs, t, c = x_enc.shape

        # channel independence reshape
        x_enc = x_enc.permute(0, 2, 1).reshape(bs * c, t, 1)
        router_input = x_enc

        # multi-scale downsample
        ms_x_enc_list = self.ms_utils(x_enc)
        ms_gt = self.ms_utils.concat_sampling_list(ms_x_enc_list[1:] + [x_enc])
        ms_gt = ms_gt.reshape(bs, c, -1).permute(0, 2, 1)  # [bs, ms_t, c]

        # patch + pos embedding
        x_enc = x_enc.permute(0, 2, 1)
        _, x_enc = self.patch_embedding(x_enc)
        for i in range(self.n_scales):
            x_enc_i = ms_x_enc_list[i]
            x_enc_i = x_enc_i.permute(0, 2, 1)
            x_enc_emb_i, _ = self.patch_embedding(x_enc_i)
            ms_x_enc_list[i] = x_enc_emb_i
        ms_x_enc = self.ms_utils.concat_sampling_list(ms_x_enc_list)

        pos_emb = self.pos_embedding(self.ms_p_lens[-1])
        ms_pos_emb_list = self.ms_utils(pos_emb)
        ms_pos_emb = self.ms_utils.concat_sampling_list(ms_pos_emb_list)
        ms_x_enc = ms_x_enc + ms_pos_emb

        # encoder
        ms_x_enc, _ = self.encoder(ms_x_enc, self.scale_ind_mask)

        # context
        if self.training:
            query_latent_distances, context = self.context_net(router_input, ms_x_enc)
            context = context.unsqueeze(0).expand(bs * c, -1, -1)
            query_latent_distances = query_latent_distances.reshape(bs, c, 1).permute(0, 2, 1)
        else:
            query_latent_distances = torch.zeros(bs, 1, c, device=x_enc.device)
            context = self.context_net.extractor.concat_context(self.context_net.extractor.context)
            context = context.unsqueeze(0).expand(bs * c, -1, -1)

        # upsample & decoder
        ms_x_enc_list = self.ms_utils.split_2_list(ms_x_enc, ms_t_lens=self.ms_p_lens, mode="encoder")
        ms_x_dec_list = self.ms_utils.up(ms_x_enc_list, ms_t_lens=self.ms_p_lens)
        ms_x_dec = self.ms_utils.concat_sampling_list(ms_x_dec_list)

        ms_x_dec, _, _ = self.decoder(ms_x_dec, context, self.next_scale_mask)
        ms_x_dec = ms_x_dec.reshape(bs * c, -1, 1).reshape(bs, c, -1).permute(0, 2, 1)
        ms_x_dec_list = self.ms_utils.split_2_list(ms_x_dec, ms_t_lens=self.ms_t_lens_, mode="decoder")
        for i in range(len(ms_x_dec_list)):
            ms_x_dec_list[i] = ms_x_dec_list[i][:, : self.ms_t_lens[i + 1]]
        ms_x_dec = self.ms_utils.concat_sampling_list(ms_x_dec_list)

        return ms_gt, ms_x_dec, query_latent_distances

    def _ms_anomaly_score(self, ms_x_dec, ms_gt):
        ms_score = F.mse_loss(ms_x_dec, ms_gt, reduction="none")
        ms_score_list = self.ms_utils.split_2_list(ms_score, ms_t_lens=self.ms_t_lens, mode="decoder")
        for i in range(len(ms_score_list) - 1):
            loss_i = ms_score_list[i].permute(0, 2, 1)
            up_loss_i = F.interpolate(loss_i, size=ms_score_list[-1].shape[1], mode="linear").permute(0, 2, 1)
            ms_score_list[-1] = ms_score_list[-1] + up_loss_i
        return ms_score_list[-1]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Return reconstruction for compatibility with default vali
        _, ms_x_dec, _ = self._forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return ms_x_dec[:, -self.seq_len :, :]

    def forward_with_loss(self, x_enc):
        ms_gt, ms_x_dec, q_distance = self._forward(x_enc, None, None, None)
        return F.mse_loss(ms_x_dec, ms_gt), torch.mean(q_distance)

    def compute_anomaly_score(self, x_enc):
        ms_gt, ms_x_dec, _ = self._forward(x_enc, None, None, None)
        score = self._ms_anomaly_score(ms_x_dec, ms_gt)  # [bs, t, c]
        return score.mean(dim=-1)  # [bs, t]

