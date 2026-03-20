from math import sqrt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from layers.Embed import DataEmbedding
from utils.masking import TriangularCausalMask


# ===================== MSGBlock =====================
class Predict(nn.Module):
    def __init__(self,  individual, c_out, seq_len, pred_len, dropout):
        super(Predict, self).__init__()
        self.individual = individual
        self.c_out = c_out

        if self.individual:
            self.seq2pred = nn.ModuleList()
            self.dropout = nn.ModuleList()
            for i in range(self.c_out):
                self.seq2pred.append(nn.Linear(seq_len , pred_len))
                self.dropout.append(nn.Dropout(dropout))
        else:
            self.seq2pred = nn.Linear(seq_len , pred_len)
            self.dropout = nn.Dropout(dropout)

    #(B,  c_out , seq)
    def forward(self, x):
        if self.individual:
            out = []
            for i in range(self.c_out):
                per_out = self.seq2pred[i](x[:,i,:])
                per_out = self.dropout[i](per_out)
                out.append(per_out)
            out = torch.stack(out,dim=1)
        else:
            out = self.seq2pred(x)
            out = self.dropout(out)

        return out


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # return V.contiguous()
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class self_attention(nn.Module):
    def __init__(self, attention, d_model ,n_heads):
        super(self_attention, self).__init__()
        d_keys =  d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention( attention_dropout = 0.1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads


    def forward(self, queries ,keys ,values, attn_mask= None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
                    queries,
                    keys,
                    values,
                    attn_mask
                )
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out , attn


class Attention_Block(nn.Module):
    def __init__(self,  d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super(Attention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = self_attention(FullAttention, d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        # x = torch.einsum('ncwl,wv->nclv',(x,A)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class GraphBlock(nn.Module):
    def __init__(self, c_out , d_model , conv_channel, skip_channel,
                        gcn_depth , dropout, propalpha ,seq_len , node_dim):
        super(GraphBlock, self).__init__()

        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        self.start_conv = nn.Conv2d(1, conv_channel, (d_model - c_out + 1, 1))
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv2d(skip_channel, seq_len , (1, seq_len ))
        self.linear = nn.Linear(c_out, d_model)
        self.norm = nn.LayerNorm(d_model)
    # x in (B, T, d_model)
    # Here we use a mlp to fit a complex mapping f (x)
    def forward(self, x):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        out = x.unsqueeze(1).transpose(2, 3)
        out = self.start_conv(out)
        out = self.gelu(self.gconv1(out , adp))
        out = self.end_conv(out).squeeze(-1)
        out = self.linear(out)

        return self.norm(x + out)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class simpleVIT(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size=2, depth=1, num_heads=4, dropout=0.1,init_weight =True):
        super(simpleVIT, self).__init__()
        self.emb_size = emb_size
        self.depth = depth
        self.to_patch = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, 2 * patch_size + 1, padding= patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, dropout),
                FeedForward(emb_size,  emb_size)
            ]))

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        B , N ,_ ,P = x.shape
        x = self.to_patch(x)
        # x = x.permute(0, 2, 3, 1).reshape(B,-1, N)
        for  norm ,attn, ff in self.layers:
            x = attn(norm(x)) + x
            x = ff(x) + x

        x = x.transpose(1,2).reshape(B, self.emb_size ,-1, P)
        return x


# ===================== Model =====================
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class ScaleGraphBlock(nn.Module):
    def __init__(self, configs):
        super(ScaleGraphBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.att0 = Attention_Block(configs.d_model, configs.d_ff,
                                   n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList()
        for i in range(self.k):
            self.gconv.append(
                GraphBlock(configs.c_out , configs.d_model , configs.conv_channel, configs.skip_channel,
                        configs.gcn_depth , configs.dropout, configs.propalpha ,configs.seq_len,
                           configs.node_dim))


    def forward(self, x):
        B, T, N = x.size()
        scale_list, scale_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            scale = scale_list[i]
            #Gconv
            x = self.gconv[i](x)
            # paddng
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // scale, scale, N)

        #for Mul-attetion
            out = out.reshape(-1 , scale , N)
            out = self.norm(self.att0(out))
            out = self.gelu(out)
            out = out.reshape(B, -1 , scale , N).reshape(B ,-1 ,N)
        # #for simpleVIT
        #     out = self.att(out.permute(0, 3, 1, 2).contiguous()) #return
        #     out = out.permute(0, 2, 3, 1).reshape(B, -1 ,N)

            out = out[:, :self.seq_len, :]
            res.append(out)

        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # for graph
        # self.num_nodes = configs.c_out
        # self.subgraph_size = configs.subgraph_size
        # self.node_dim = configs.node_dim
        # to return adj (node , node)
        # self.graph = constructor_graph()

        self.model = nn.ModuleList([ScaleGraphBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                           configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.seq2pred = Predict(configs.individual, configs.c_out,
                                configs.seq_len, configs.pred_len, configs.dropout)

        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # adp = self.graph(torch.arange(self.num_nodes).to(self.device))
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)
        dec_out = self.seq2pred(dec_out.transpose(1, 2)).transpose(1, 2)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # adp = self.graph(torch.arange(self.num_nodes).to(self.device))
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)
        # dec_out = self.seq2pred(dec_out.transpose(1, 2)).transpose(1, 2)
        # print(dec_out.shape)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, L, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, L, 1))

        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # adp = self.graph(torch.arange(self.num_nodes).to(self.device))
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)
        # dec_out = self.seq2pred(dec_out.transpose(1, 2)).transpose(1, 2)
        # print(dec_out.shape)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, L, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, L, 1))

        return dec_out


    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # adp = self.graph(torch.arange(self.num_nodes).to(self.device))
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
