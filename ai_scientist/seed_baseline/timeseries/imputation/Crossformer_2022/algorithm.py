import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple

# ===== Inlined Components =====

class PositionalEmbedding(nn.Module):
    """Positional embedding for dimension and segment"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space
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

class DSWEmbedding(nn.Module):
    """Dimension-Segment-Wise Embedding"""
    def __init__(self, seq_len, d_model, enc_in, seg_len, dropout=0.1):
        super(DSWEmbedding, self).__init__()
        self.seg_len = seg_len
        self.d_model = d_model
        self.enc_in = enc_in
        
        # Calculate number of segments
        self.seg_num = seq_len // seg_len
        
        # Linear projection for each segment
        self.projection = nn.Linear(seg_len, d_model)
        
        # Position embedding for 2D array (segment_idx, dimension_idx)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seg_num, enc_in, d_model) * 0.02
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        
        # Reshape to segments: [B, seg_num, D, seg_len]
        x = x.reshape(B, self.seg_num, self.seg_len, D)
        x = x.permute(0, 1, 3, 2)  # [B, seg_num, D, seg_len]
        
        # Project each segment: [B, seg_num, D, d_model]
        x = self.projection(x)
        
        # Add position embedding
        x = x + self.pos_embedding
        
        x = self.dropout(x)
        
        return x  # [B, seg_num, D, d_model]

class RouterMechanism(nn.Module):
    """Router mechanism for cross-dimension stage"""
    def __init__(self, d_model, n_routers=4, n_heads=8):
        super(RouterMechanism, self).__init__()
        self.n_routers = n_routers
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Learnable routers
        self.routers = nn.Parameter(torch.randn(1, 1, n_routers, d_model) * 0.02)
        
        # MSA layers
        self.msa1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.msa2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
    def forward(self, x):
        # x: [B, L, D, d_model]
        B, L, D, d_model = x.shape
        
        # Process each time step
        outputs = []
        for i in range(L):
            x_i = x[:, i, :, :]  # [B, D, d_model]
            routers = self.routers.expand(B, -1, -1, -1).squeeze(1)  # [B, n_routers, d_model]
            
            # Aggregate: routers gather from all dimensions
            aggregated, _ = self.msa1(routers, x_i, x_i)  # [B, n_routers, d_model]
            
            # Distribute: dimensions receive from routers
            output, _ = self.msa2(x_i, aggregated, aggregated)  # [B, D, d_model]
            
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)  # [B, L, D, d_model]
        return outputs

class TSALayer(nn.Module):
    """Two-Stage Attention Layer"""
    def __init__(self, d_model, n_heads, d_ff, enc_in, n_routers=4, dropout=0.1):
        super(TSALayer, self).__init__()
        
        # Cross-Time Stage
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.time_norm1 = nn.LayerNorm(d_model)
        self.time_norm2 = nn.LayerNorm(d_model)
        self.time_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Cross-Dimension Stage
        self.router = RouterMechanism(d_model, n_routers, n_heads)
        self.dim_norm1 = nn.LayerNorm(d_model)
        self.dim_norm2 = nn.LayerNorm(d_model)
        self.dim_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.enc_in = enc_in
        
    def forward(self, x):
        # x: [B, L, D, d_model]
        B, L, D, d_model = x.shape
        
        # Cross-Time Stage: process each dimension separately
        time_outputs = []
        for d in range(D):
            x_d = x[:, :, d, :]  # [B, L, d_model]
            
            # Self-attention on time dimension
            attn_out, _ = self.time_attn(x_d, x_d, x_d)
            x_d = self.time_norm1(x_d + attn_out)
            
            # FFN
            ffn_out = self.time_ffn(x_d)
            x_d = self.time_norm2(x_d + ffn_out)
            
            time_outputs.append(x_d)
        
        x_time = torch.stack(time_outputs, dim=2)  # [B, L, D, d_model]
        
        # Cross-Dimension Stage: router mechanism
        router_out = self.router(x_time)  # [B, L, D, d_model]
        x_dim = self.dim_norm1(x_time + router_out)
        
        # FFN
        B, L, D, d_model = x_dim.shape
        x_dim_flat = x_dim.reshape(B * L * D, d_model)
        ffn_out = self.dim_ffn(x_dim_flat)
        ffn_out = ffn_out.reshape(B, L, D, d_model)
        x_dim = self.dim_norm2(x_dim + ffn_out)
        
        return x_dim

class SegmentMerging(nn.Module):
    """Merge adjacent segments in time dimension"""
    def __init__(self, d_model):
        super(SegmentMerging, self).__init__()
        self.merge = nn.Linear(2 * d_model, d_model)
        
    def forward(self, x):
        # x: [B, L, D, d_model]
        B, L, D, d_model = x.shape
        
        # Ensure L is even, pad if necessary
        if L % 2 != 0:
            padding = x[:, -1:, :, :]
            x = torch.cat([x, padding], dim=1)
            L = L + 1
        
        # Merge pairs: [B, L//2, D, 2*d_model]
        x = x.reshape(B, L // 2, 2, D, d_model)
        x = x.permute(0, 1, 3, 2, 4)  # [B, L//2, D, 2, d_model]
        x = x.reshape(B, L // 2, D, 2 * d_model)
        
        # Linear projection
        x = self.merge(x)  # [B, L//2, D, d_model]
        
        return x

class HierarchicalEncoder(nn.Module):
    """Hierarchical Encoder with TSA layers and segment merging"""
    def __init__(self, configs, seg_len):
        super(HierarchicalEncoder, self).__init__()
        
        self.seg_len = seg_len
        self.seg_num = configs.seq_len // seg_len
        
        # Embedding
        self.embedding = DSWEmbedding(
            configs.seq_len, configs.d_model, configs.enc_in, 
            seg_len, configs.dropout
        )
        
        # Encoder layers
        self.layers = nn.ModuleList([
            TSALayer(
                configs.d_model, configs.n_heads, configs.d_ff,
                configs.enc_in, n_routers=4, dropout=configs.dropout
            ) for _ in range(configs.e_layers)
        ])
        
        # Segment merging layers (except for first layer)
        self.merging = nn.ModuleList([
            SegmentMerging(configs.d_model) for _ in range(configs.e_layers - 1)
        ])
        
    def forward(self, x):
        # x: [B, L, D]
        # Embedding
        x = self.embedding(x)  # [B, seg_num, D, d_model]
        
        # Store outputs of all layers
        layer_outputs = [x]
        
        # Process through encoder layers
        for i, layer in enumerate(self.layers):
            x = layer(x)  # [B, L_i, D, d_model]
            
            # Merge segments (except last layer)
            if i < len(self.layers) - 1:
                x = self.merging[i](x)
            
            layer_outputs.append(x)
        
        return layer_outputs

class HierarchicalDecoder(nn.Module):
    """Hierarchical Decoder"""
    def __init__(self, configs, seg_len):
        super(HierarchicalDecoder, self).__init__()
        
        self.seg_len = seg_len
        self.seg_num = configs.seq_len // seg_len
        self.n_layers = configs.e_layers
        
        # Learnable position embeddings for decoder
        self.pos_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.seg_num // (2 ** i), configs.enc_in, configs.d_model) * 0.02)
            for i in range(self.n_layers + 1)
        ])
        
        # TSA layers for decoder
        self.tsa_layers = nn.ModuleList([
            TSALayer(
                configs.d_model, configs.n_heads, configs.d_ff,
                configs.enc_in, n_routers=4, dropout=configs.dropout
            ) for _ in range(self.n_layers + 1)
        ])
        
        # Cross attention layers (decoder-encoder)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(configs.d_model, configs.n_heads, dropout=configs.dropout, batch_first=True)
            for _ in range(self.n_layers + 1)
        ])
        
        self.norms1 = nn.ModuleList([nn.LayerNorm(configs.d_model) for _ in range(self.n_layers + 1)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(configs.d_model) for _ in range(self.n_layers + 1)])
        
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_ff, configs.d_model),
                nn.Dropout(configs.dropout)
            ) for _ in range(self.n_layers + 1)
        ])
        
        # Projection layers for each scale
        self.projections = nn.ModuleList([
            nn.Linear(configs.d_model, seg_len) for _ in range(self.n_layers + 1)
        ])
        
        self.enc_in = configs.enc_in
        
    def forward(self, encoder_outputs):
        # encoder_outputs: list of [B, L_i, D, d_model]
        predictions = []
        
        for layer_idx in range(self.n_layers + 1):
            enc_out = encoder_outputs[layer_idx]
            B, L, D, d_model = enc_out.shape
            
            # Initialize decoder with position embedding
            if layer_idx == 0:
                dec_input = self.pos_embeddings[layer_idx].expand(B, -1, -1, -1)
            else:
                # Use previous decoder output
                dec_input = dec_out
            
            # TSA layer
            dec_out = self.tsa_layers[layer_idx](dec_input)
            
            # Cross attention with encoder (process each dimension)
            cross_outputs = []
            for d in range(D):
                dec_d = dec_out[:, :, d, :]  # [B, L, d_model]
                enc_d = enc_out[:, :, d, :]  # [B, L, d_model]
                
                cross_out, _ = self.cross_attns[layer_idx](dec_d, enc_d, enc_d)
                dec_d = self.norms1[layer_idx](dec_d + cross_out)
                
                # FFN
                ffn_out = self.ffns[layer_idx](dec_d)
                dec_d = self.norms2[layer_idx](dec_d + ffn_out)
                
                cross_outputs.append(dec_d)
            
            dec_out = torch.stack(cross_outputs, dim=2)  # [B, L, D, d_model]
            
            # Project to time series segments
            B, L, D, d_model = dec_out.shape
            dec_out_flat = dec_out.reshape(B * L * D, d_model)
            pred_seg = self.projections[layer_idx](dec_out_flat)  # [B*L*D, seg_len]
            pred_seg = pred_seg.reshape(B, L, D, self.seg_len)
            
            # Reshape to original sequence: [B, L*seg_len, D]
            pred_seg = pred_seg.permute(0, 2, 1, 3)  # [B, D, L, seg_len]
            pred_seg = pred_seg.reshape(B, D, -1)  # [B, D, L*seg_len]
            pred_seg = pred_seg.permute(0, 2, 1)  # [B, L*seg_len, D]
            
            predictions.append(pred_seg)
        
        return predictions

# ===== End Inlined Components =====

class Model(nn.Module):
    """
    Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len  # For imputation, output length equals input length
        
        # Segment length
        self.seg_len = getattr(configs, 'seg_len', 6)
        
        # Ensure seq_len is divisible by seg_len
        if self.seq_len % self.seg_len != 0:
            # Pad to make it divisible
            self.pad_len = self.seg_len - (self.seq_len % self.seg_len)
        else:
            self.pad_len = 0
        
        self.padded_seq_len = self.seq_len + self.pad_len
        
        # Update configs for padded length
        configs.seq_len = self.padded_seq_len
        
        # Hierarchical Encoder-Decoder
        self.encoder = HierarchicalEncoder(configs, self.seg_len)
        self.decoder = HierarchicalDecoder(configs, self.seg_len)
        
        # Output projection
        self.output_projection = nn.Linear(configs.enc_in, configs.c_out)
        
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # x_enc: [B, L, D]
        B, L, D = x_enc.shape
        
        # Pad if necessary
        if self.pad_len > 0:
            padding = x_enc[:, -self.pad_len:, :]
            x_enc = torch.cat([x_enc, padding], dim=1)
        
        # Normalize
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev
        
        # Encoder
        encoder_outputs = self.encoder(x_enc)
        
        # Decoder
        decoder_outputs = self.decoder(encoder_outputs)
        
        # Sum predictions from all scales
        dec_out = torch.zeros_like(decoder_outputs[0])
        for pred in decoder_outputs:
            # Truncate or pad to match target length
            if pred.shape[1] > dec_out.shape[1]:
                pred = pred[:, :dec_out.shape[1], :]
            elif pred.shape[1] < dec_out.shape[1]:
                padding = torch.zeros(B, dec_out.shape[1] - pred.shape[1], D, device=pred.device)
                pred = torch.cat([pred, padding], dim=1)
            dec_out = dec_out + pred
        
        # Remove padding
        if self.pad_len > 0:
            dec_out = dec_out[:, :L, :]
        
        # Denormalize
        dec_out = dec_out * stdev + means
        
        # Project to output dimension
        if D != dec_out.shape[-1]:
            dec_out = self.output_projection(dec_out)
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return dec_out  # [B, L, D]