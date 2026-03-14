# coding: utf-8
"""
BM3: Bootstrapped Multi-modal Model for Multimedia Recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender


class BM3(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BM3, self).__init__(config, dataset)
        
        # Model configurations
        self.embedding_size = config['embedding_size']  # d
        self.n_layers = config['n_layers']  # L
        self.dropout_ratio = config['dropout_ratio']  # p for Bernoulli dropout
        self.cl_weight = config['cl_weight']  # weight for contrastive losses
        self.reg_weight = config['reg_weight']  # λ for regularization
        
        # Initialize ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Multi-modal feature projectors (Section 3.1.1)
        if self.v_feat is not None:
            self.visual_feat_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.visual_projector = nn.Linear(self.v_feat.shape[1], self.embedding_size)
        
        if self.t_feat is not None:
            self.text_feat_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_projector = nn.Linear(self.t_feat.shape[1], self.embedding_size)
        
        # Shared predictor for contrastive learning (Section 3.2.1)
        self.predictor = nn.Linear(self.embedding_size, self.embedding_size)
        
        # Build interaction graph for LightGCN
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self._build_normalized_adj()
        self.norm_adj = self._sparse_mx_to_torch_sparse_tensor(self.norm_adj).to(self.device)
        
    def _build_normalized_adj(self):
        """Build normalized adjacency matrix for LightGCN propagation"""
        # Create user-item bipartite graph
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), 
                               dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        
        # Fill adjacency matrix
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        rowsum = np.array(adj_mat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        return norm_adj.tocoo()
    
    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert scipy sparse matrix to torch sparse tensor"""
        sparse_mx = sparse_mx.astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)
    
    def lightgcn_propagation(self):
        """LightGCN propagation for ID embeddings (Section 3.1.2)"""
        # Stack user and item embeddings: H^0
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]
        
        # Multi-layer propagation
        for layer in range(self.n_layers):
            # H^(l+1) = (D^(-1/2) * A * D^(-1/2)) * H^l
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # READOUT: mean aggregation across layers
        final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        
        # Split user and item embeddings
        user_embeddings, item_embeddings = torch.split(final_embeddings, 
                                                      [self.n_users, self.n_items], dim=0)
        
        # Add residual connection for items (Section 3.1.2)
        item_embeddings = item_embeddings + self.item_embedding.weight
        
        return user_embeddings, item_embeddings
    
    def project_multimodal_features(self):
        """Project multi-modal features to shared latent space (Section 3.1.1)"""
        modal_embeddings = {}
        
        if self.v_feat is not None:
            # h_v = e_v * W_v + b_v
            visual_features = self.visual_feat_embedding.weight
            visual_embeddings = self.visual_projector(visual_features)
            modal_embeddings['visual'] = visual_embeddings
        
        if self.t_feat is not None:
            # h_t = e_t * W_t + b_t
            text_features = self.text_feat_embedding.weight
            text_embeddings = self.text_projector(text_features)
            modal_embeddings['text'] = text_embeddings
        
        return modal_embeddings
    
    def contrastive_view_generator(self, embeddings):
        """Generate contrastive views via dropout (Section 3.2.1)"""
        # Original view through predictor: h_tilde = h * W_p + b_p
        online_view = self.predictor(embeddings)
        
        # Target view through dropout: h_dot = h * Bernoulli(p)
        dropout_mask = torch.bernoulli(torch.full_like(embeddings, 1 - self.dropout_ratio))
        target_view = embeddings * dropout_mask
        
        return online_view, target_view.detach()  # Stop gradient on target view
    
    def cosine_loss(self, x, y):
        """Negative cosine similarity loss (Section 3.2.2)"""
        # C(h_u, h_i) = -h_u^T * h_i / (||h_u||_2 * ||h_i||_2)
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        return -(x_norm * y_norm).sum(dim=1).mean()
    
    def forward(self, interaction):
        """Forward pass for training"""
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        
        # Get ID embeddings through LightGCN
        user_embeddings, item_embeddings = self.lightgcn_propagation()
        
        # Get multi-modal embeddings
        modal_embeddings = self.project_multimodal_features()
        
        # Store embeddings for loss calculation
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.modal_embeddings = modal_embeddings
        
        # Get user and item embeddings for current batch
        batch_user_emb = user_embeddings[users]
        batch_pos_item_emb = item_embeddings[pos_items]
        batch_neg_item_emb = item_embeddings[neg_items]
        
        # Score calculation for BPR (not used in BM3's self-supervised training)
        pos_scores = (batch_user_emb * batch_pos_item_emb).sum(dim=1)
        neg_scores = (batch_user_emb * batch_neg_item_emb).sum(dim=1)
        
        return pos_scores, neg_scores
    
    def calculate_loss(self, interaction):
        """Calculate multi-modal contrastive loss (Section 3.2)"""
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        
        # Forward pass to get embeddings
        self.forward(interaction)
        
        total_loss = 0.0
        
        # Get batch embeddings
        batch_user_emb = self.user_embeddings[users]
        batch_pos_item_emb = self.item_embeddings[pos_items]
        
        # 1. Graph Reconstruction Loss (Section 3.2.2)
        user_online, user_target = self.contrastive_view_generator(batch_user_emb)
        item_online, item_target = self.contrastive_view_generator(batch_pos_item_emb)
        
        # L_rec = C(h_tilde_u, sg(h_dot_i)) + C(sg(h_dot_u), h_tilde_i)
        loss_rec = (self.cosine_loss(user_online, item_target) + 
                   self.cosine_loss(user_target, item_online))
        total_loss += loss_rec
        
        # 2. Inter-modality Feature Alignment Loss (Section 3.2.3)
        loss_align = 0.0
        if self.modal_embeddings:
            for modality, modal_emb in self.modal_embeddings.items():
                batch_modal_emb = modal_emb[pos_items]
                modal_online, modal_target = self.contrastive_view_generator(batch_modal_emb)
                
                # L_align = C(h_tilde_m^i, h_dot_i)
                align_loss = self.cosine_loss(modal_online, item_target)
                loss_align += align_loss
            
            total_loss += self.cl_weight * loss_align
        
        # 3. Intra-modality Feature Masked Loss (Section 3.2.4)
        loss_mask = 0.0
        if self.modal_embeddings:
            for modality, modal_emb in self.modal_embeddings.items():
                batch_modal_emb = modal_emb[pos_items]
                modal_online, modal_target = self.contrastive_view_generator(batch_modal_emb)
                
                # L_mask = C(h_tilde_m^i, h_dot_m^i)
                mask_loss = self.cosine_loss(modal_online, modal_target)
                loss_mask += mask_loss
            
            total_loss += self.cl_weight * loss_mask
        
        # 4. Regularization Loss
        reg_loss = self.reg_weight * (
            (batch_user_emb ** 2).mean() + 
            (batch_pos_item_emb ** 2).mean()
        )
        total_loss += reg_loss
        
        return total_loss.squeeze()
    
    def full_sort_predict(self, interaction):
        """Predict scores for all items for given users (Section 3.3)"""
        users = interaction[0]
        
        # Get final embeddings
        user_embeddings, item_embeddings = self.lightgcn_propagation()
        
        # Use predictor-transformed embeddings for scoring
        user_emb = self.predictor(user_embeddings[users])
        item_emb = self.predictor(item_embeddings)
        
        # s(h_u, h_i) = h_tilde_u · h_tilde_i^T
        scores = torch.matmul(user_emb, item_emb.transpose(0, 1))
        
        return scores