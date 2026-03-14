# Domain Knowledge

- Domain: Recsys
- Task: MultiModal

## signal_decomposition_fusion

多模态推荐的核心思想是将协同信号与内容信号解耦后分别编码再融合。协同信号来自用户-物品交互图，捕获行为模式；内容信号来自模态特征，提供语义信息。两类信号通过不同的图结构独立传播：交互图采用 LightGCN 风格的二部图传播，模态图采用物品-物品 kNN 相似度图传播。融合策略包括加权求和（可学习权重）、门控机制、注意力以及简单的残差相加。这种解耦-编码-融合的范式使模型能够分别捕获不同类型的用户偏好，优于端到端混合建模。

## simplified_graph_propagation


多模态推荐普遍采用 LightGCN 风格的简化图传播：去除 GCN 中的特征变换矩阵和非线性激活函数，仅保留邻居聚合操作。这一简化基于推荐场景的特殊性——用户和物品的 ID 嵌入已足够表达个性化信息，额外的非线性变换反而引入噪声。标准实现：多层传播后将各层输出堆叠取均值（torch.stack + mean）作为最终表示，而非仅用最后一层。残差连接（如 item_emb = item_emb + modal_h）是防止深层传播过平滑的关键技术。邻接矩阵使用对称归一化 D^(-1/2)AD^(-1/2)，传播层数通常为 1-3 层。

## modality_as_item_view


多模态特征本质上是物品的内容描述，因此模态信息的传播路径是：物品特征 → 物品-物品模态图传播 → 增强物品表示 → 通过交互图间接影响用户表示。用户并不直接拥有模态特征，而是通过其交互的物品「继承」模态偏好。模态图的构建方式为：计算物品间的余弦相似度，取 Top-K 邻居构成 kNN 图，再进行归一化。该图可在初始化时预计算并缓存到文件（torch.save/load），避免重复计算。部分方法（如 LATTICE）会在训练中动态更新模态图，结合原始相似度图和学习到的图。用户模态偏好的建模通常通过可学习的 preference 参数或通过交互矩阵聚合物品模态特征实现。

## contrastive_self_supervision
- Title: 对比学习作为辅助监督

对比学习在多模态推荐中扮演两个角色：跨模态对齐和表示增强。跨模态对比（如 BM3 的 visual-ID 对齐）使不同模态的表示在语义空间中对齐；跨视图对比增强表示的鲁棒性。实现要点：通过 F.dropout 生成增强视图，使用 .detach() 实现 stop-gradient 防止表示坍缩，采用 1 - cosine_similarity().mean() 作为损失。BYOL 风格的方法（如 BM3）需配合 predictor 使用：online 视图过 predictor，target 视图。 不过, 温度系数 tau 控制分布平滑程度，典型值 0.1-0.5。

## loss_composition_pattern

多模态推荐的损失函数通常由主损失和辅助损失组成，权重分配直接影响模型性能。主损失负责建模用户-物品交互（如 BPR 损失、用户-物品对比损失），权重固定为 1；辅助损失负责模态对齐或正则化，需乘以权重系数（cl_weight、reg_weight）。典型模式：loss = main_loss + cl_weight * modal_loss + reg_weight * reg_loss。常见错误是将所有损失都乘以 cl_weight，导致主损失信号被大幅削弱。以 BM3 为例，loss_ui + loss_iu 是主损失（权重为 1），而模态对齐损失 loss_t + loss_v 才乘以 cl_weight。
