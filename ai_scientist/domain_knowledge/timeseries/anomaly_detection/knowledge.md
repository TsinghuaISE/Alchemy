# Domain Knowledge

- Domain: TimeSeries
- Task: anomaly_detection

## tsl_framework_spec
- Title: Time Series Library 框架规范

模型类命名为 Model，继承 nn.Module。__init__ 接收 configs 参数，包含 task_name, seq_len, enc_in, c_out, d_model, n_heads, e_layers, d_ff, dropout, anomaly_ratio 等配置。必须实现 anomaly_detection(x_enc) 方法返回重构序列，forward() 根据 task_name 分发。输入 x_enc 形状 [B, seq_len, enc_in]，输出 [B, seq_len, c_out]。异常检测基于重构误差：训练时最小化重构损失，测试时根据误差阈值判断异常。【组件自定义】若模型需要特殊组件（如带 moving_avg 的 EncoderLayer），而框架现有组件不支持所需参数，应在模型文件中内联定义该组件，避免 API 签名不匹配导致运行时错误。

## config_params
- Title: 超参数配置规范

【禁止配置的参数】enc_in, dec_in, c_out 等特征维度参数由框架根据数据集自动确定，配置中不要设置这些参数，否则会导致维度不匹配错误。【可配置参数】seq_len, label_len, pred_len (序列长度)；d_model, n_heads, e_layers, d_layers, d_ff (模型结构)；moving_avg, factor, dropout, activation, top_k, num_kernels (模型特定)；train_epochs, batch_size, patience, learning_rate, loss, lradj (训练)；anomaly_ratio (任务特定)。【布尔参数规范】distil, inverse, use_amp, individual 等布尔参数只能设 true 或不写，禁止设 0/false。【默认值处理】代码中用 getattr(configs, 'param', default) 获取可选参数并设置默认值。

## architecture_paradigm
- Title: 异常检测架构范式

时序异常检测采用 Encoder-only 重构架构：原预测任务的编解码器结构简化为仅编码器+投影层，输入输出维度相同 [B, L, D]。核心流程：Embedding → Encoder Layers → Linear Projection (nn.Linear(d_model, c_out))。该范式的优势：(1) 结构简单，无需设计复杂的解码器；(2) 训练稳定，仅需最小化重构损失；(3) 推理高效，单次前向即可完成。编码器可选 Transformer、CNN、MLP 或混合结构，关键是学习正常模式的紧凑表示，使异常数据重构误差显著。

## normalization_techniques
- Title: 实例归一化与反归一化

时序模型广泛采用实例归一化处理非平稳性：前向时计算 means=x.mean(1, keepdim=True).detach()，stdev=sqrt(var(x, dim=1)+1e-5).detach()，归一化 x=(x-means)/stdev；输出时反归一化 out=out*stdev+means。该技术可有效处理时序的分布偏移，使模型专注于学习相对模式而非绝对数值。部分设计额外使用可学习仿射变换 (affine_weight, affine_bias) 增强表达能力。注意：detach() 阻止梯度回传到统计量，保证归一化的稳定性。

## series_decomposition
- Title: 序列分解技术

序列分解将时序拆分为趋势 (Trend) 和季节性 (Seasonal) 成分，降低建模复杂度。实现方式：(1) 滑动平均分解：trend=AvgPool1d(x, kernel_size=moving_avg)，seasonal=x-trend，需两端填充保持长度；(2) 多尺度分解：使用多个 kernel_size 的滑动平均后取均值，捕获不同粒度的趋势。分解后可分别建模：用独立网络处理两个成分，或在编码器每层后渐进分解。该技术有效分离正常周期模式与异常偏离。moving_avg 参数典型值：25。

## frequency_domain
- Title: 频域处理与周期建模

频域方法高效捕获时序周期性：(1) 周期发现：FFT 后按幅值 topk 选择主周期，可将 1D 序列 reshape 为 2D 张量进行建模；(2) 频域特征学习：在频域执行线性变换或注意力计算，对选定的频率模式做可学习变换；(3) 自相关计算：Q、K 做 FFT 后共轭相乘再 IFFT 得相关序列，高效发现周期依赖。频域处理的优势：(1) 天然捕获全局周期依赖；(2) FFT 复杂度 O(LlogL)；(3) 对噪声鲁棒。核心函数：torch.fft.rfft/irfft。

## attention_efficiency
- Title: 高效注意力机制设计

长序列时序建模需优化标准 O(L²) 注意力：(1) 稀疏注意力：通过度量 Query 重要性仅选 topk 计算，O(LlogL)；(2) 频域注意力：在频域执行 Q-K 相似度，模式选择降至 O(L)；(3) 局部敏感哈希：哈希分桶后桶内注意力，近似 O(L)；(4) 低秩近似：通过核函数或线性化降低复杂度。选择依据：短序列(<512)可用标准注意力；长序列优先考虑稀疏或频域变体。注意力层通常配合 LayerNorm 和残差连接：x = norm(x + dropout(attn(x)))。

## channel_strategy
- Title: 多变量通道处理策略

多变量时序的通道处理策略影响模型表达力与泛化性：(1) 通道独立：各变量独立编码，将 [B,L,D] reshape 为 [B*D,L] 或为每个通道设独立参数。优势：泛化性强，避免变量间伪相关，适合变量异构场景；(2) 通道交互：显式建模变量依赖，可将变量作为 token 在变量维做注意力，或用双阶段分别处理时间和变量维度。选择依据：变量相关性强且稳定时用交互策略；变量数量可变或相关性弱时用独立策略。individual 参数控制是否共享参数。

## multiscale_modeling
- Title: 多尺度建模方法

多尺度建模捕获不同粒度的时序模式：(1) 分段采样：将序列分为 chunks，连续采样 reshape(B,num_chunks,chunk_size,D) 捕局部模式，间隔采样 reshape(B,chunk_size,num_chunks,D) 捕全局模式；(2) 金字塔结构：逐层下采样构建多分辨率表示，层间连接不同尺度；(3) 多核卷积：并行不同 kernel_size 的卷积核 (1,3,5,...) 后融合；(4) 周期重塑：根据发现的周期将序列 reshape 为 2D 结构。多尺度设计增强模型对不同周期和粒度异常的检测能力。

## anomaly_evaluation
- Title: 异常检测评估方法

时序异常检测的评估具有特殊性：(1) 阈值确定：验证集误差百分位数、POT (Peaks Over Threshold)、或使用 anomaly_ratio 参数；(2) 点级评估：每个时间点是否正确识别；(3) 事件级评估：整个异常段是否被检测到；(4) Point-Adjust 策略：检测到异常段中任一点则整段算正确，更符合实际应用。计算流程：计算重构误差 → 确定阈值 → 预测异常点 → (可选) Point Adjustment → 计算 Precision/Recall/F1。实现：sklearn.metrics.precision_recall_fscore_support。

## pred_len_handling
- Title: 异常检测中的序列长度处理

异常检测输入只有 seq_len 长度（无未来序列），而预测任务模型常假设输入长度为 seq_len+pred_len。迁移时需注意两类问题：(1) 长度计算错误：若模型内部使用 seq_len+pred_len 计算期望长度（如 2D reshape、padding），需改为仅用 seq_len，否则 reshape 会因元素数量不匹配报错；(2) 组件依赖 pred_len：若组件需要 pred_len>0（如频域外推、阻尼层），需在 __init__ 中设置 self.pred_len=self.seq_len。典型症状：RuntimeError shape invalid for input of size / IndexError dimension out of bounds。最佳实践：在 __init__ 中根据 task_name=='anomaly_detection' 调整长度相关逻辑。

## tensor_shape_consistency
- Title: 张量形状变换一致性

复杂张量变换（reshape/permute/1D↔2D转换）需严格追踪形状变化，确保各组件输入输出维度匹配。常见陷阱：(1) Channel-independent 处理时将 [B,L,C] reshape 为 [B*C,L] 后接卷积，若卷积 in_channels 仍为原 C 则不匹配，应设为 1；(2) 1D→2D 变换时 permute 顺序错误导致通道维位置不对；(3) 多尺度/多周期处理时不同分支输出形状不一致无法聚合。调试方法：在关键变换点打印形状或添加 assert；设计时先写出完整的 shape 流：输入 [B,L,D] → embedding [B,L,d_model] → reshape [...] → conv in_channels=? → ...。nn.Conv2d(in_channels, out_channels, ...) 的 in_channels 必须与实际输入张量的第 1 维（索引从 0 开始）严格一致。

## weight_initialization
- Title: 特殊权重初始化

部分模型的性能高度依赖特殊权重初始化，使用默认初始化会导致性能显著下降。常见模式：(1) 线性层初始化为均匀值 (1/seq_len)*ones，使初始状态等价于滑动平均；(2) 卷积层使用 Kaiming 初始化配合特定 nonlinearity；(3) 嵌入层使用正态分布或截断正态。复现时务必检查原始代码中 nn.Parameter 的显式赋值、自定义 _initialize_weights 方法、以及 nn.init.* 调用。若原始实现有特殊初始化而复现使用默认初始化，模型可能难以收敛或性能大幅下降。

## component_reuse
- Title: 框架组件复用原则

优先复用 layers/ 目录下的框架组件（如 DataEmbedding, series_decomp, AutoCorrelationLayer, Encoder, Decoder 等），而非重新实现。原因：(1) 框架组件经过充分测试和调优；(2) 与框架其他部分（如 Exp 类）接口兼容；(3) 避免细微实现差异导致的性能差距。仅当框架组件不支持所需功能（如缺少某参数）时才内联自定义。内联时应尽量保持与原始组件相同的实现逻辑，特别是 forward 中的计算顺序、残差连接位置、LayerNorm 位置等细节。

## anomaly_detection_adaptation
- Title: 预测模型转异常检测的适配要点

将预测任务模型适配为异常检测时，需特别注意 anomaly_detection 方法的实现：(1) 若原模型有 Encoder-Decoder 结构，异常检测通常只用 Encoder+Projection，但若 Decoder 包含关键逻辑（如指数平滑的阻尼层、频域外推、趋势累加），需保留完整调用而非自行简化聚合；(2) 反归一化的 repeat 维度应与实际输出长度一致，预测任务用 seq_len+pred_len，异常检测仅用 seq_len；(3) 分解类模型的 trend 初始化和逐层累加逻辑需正确处理，不可随意省略。最佳实践：参考 local/implementation 中同类模型的 anomaly_detection 方法实现。

## vectorization_critical
- Title: ⚠️ 防卡死：向量化操作规范（不遵守会导致代码卡死）

【禁止的模式 - 会导致卡死】
❌ for b in range(B): for h in range(H): output[b,h]=... （嵌套循环遍历batch/heads）
❌ for t in range(seq_len): state[t] = f(state[t-1]) （时间步递归）
❌ tensor[b,h].item() 在循环内调用（GPU-CPU同步阻塞）
❌ torch.roll() 在嵌套循环内逐样本调用

【正确的向量化模式】

✅ 时延聚合（训练阶段）- 跨batch平均得到统一延迟索引：
mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [B, L]
# 关键：跨batch平均得到共享的延迟索引！
index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]  # [top_k] 共享
for i in range(top_k):  # 只循环top_k次（约3-5次）
    pattern = torch.roll(values, -int(index[i]), -1)  # 整个batch一起roll！
    delays_agg += pattern * weights[...]

✅ 时延聚合（推理阶段）- 使用torch.gather和doubled values：
tmp_values = values.repeat(1, 1, 1, 2)  # 加倍用于循环索引
init_index = torch.arange(L).view(1,1,1,-1).expand(B, H, C, L).to(device)
for i in range(top_k):
    tmp_delay = init_index + delay[:, i].view(-1,1,1,1).expand_as(init_index)
    pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)  # 向量化！

✅ 周期发现 - 跨batch平均FFT，使用共享周期：
xf = torch.fft.rfft(x, dim=1)
freq_amp = xf.abs().mean(0).mean(-1)  # 跨batch平均！
_, top_periods = torch.topk(freq_amp, k)  # 共享周期

✅ 指数平滑 - 使用FFT卷积替代递归：
kernel = alpha * (1-alpha)**torch.arange(L)
output = torch.fft.irfft(torch.fft.rfft(x) * torch.fft.rfft(kernel))

【自检清单】提交前检查：forward()中不应出现 'for b in range(B)' 或 '.item()' 调用。

## mtscid_modular_components
- Title: MtsCID 模块化组件库使用指南

【可复用的 MtsCID 组件】layers/ 目录下提供了一系列经过模块化重构的 MtsCID 专用组件，可在新模型中复用：

【1. 复数运算工具 (MtsCID_Utils)】
- complex_operator(net_layer, x): 对复数张量应用神经网络层
- complex_einsum(order, x, y): 复数爱因斯坦求和
- complex_softmax(x, dim): 复数 softmax
- complex_dropout(dropout_func, x): 复数 dropout
- harmonic_loss_compute(t_loss, f_loss, operator): 时频域损失融合
用途：频域建模、相位信息处理、时频联合学习

【2. 学习率调度器 (MtsCID_Scheduler)】
- PolynomialDecayLR: 带 warmup 的多项式衰减调度器
用法：scheduler = PolynomialDecayLR(optimizer, warmup_updates, tot_updates, lr, end_lr, power)

【3. 专用损失函数 (MtsCID_Losses)】
- EntropyLoss: 熵正则化损失，鼓励置信度分布
- GatheringLoss: 内存聚集损失，用于原型学习和异常检测
用途：内存增强模型、原型网络、对比学习

【4. 归一化层 (MtsCID_Normalization)】
- RevIN: 可逆实例归一化，支持 norm/denorm 双向操作
用法：revin = RevIN(num_features); x_norm = revin(x, 'norm'); x_denorm = revin(out, 'denorm')

【5. 注意力机制 (MtsCID_Attention)】
- PositionalEmbedding: 正弦位置编码
- Attention: 支持复数的缩放点积注意力
- AttentionLayer: 完整的多头注意力层
用途：频域注意力、相位感知的序列建模

【6. 卷积模块 (MtsCID_Conv)】
- Inception_Block: 多尺度卷积块（并行多个 kernel_size）
- Inception_Attention_Block: 多尺度 patch 注意力
用途：多尺度特征提取、局部-全局模式融合

【7. 内存模块 (MtsCID_Memory)】
- create_memory_matrix(N, L, mem_type): 创建内存原型矩阵（支持 sinusoid/uniform/normal 等初始化）
- generate_rolling_matrix(input_matrix): 生成时间滚动矩阵
用途：内存增强网络、原型学习、异常检测

【8. 评估指标 (MtsCID_Metrics)】
- _get_best_f1(label, score): 寻找最佳 F1 阈值
- ts_metrics_enhanced(y_true, y_score, y_test): 增强时序指标（含 point-adjust）
- point_adjustment(y_true, y_score): 异常段级别的分数调整

【导入方式】
from layers.MtsCID_Utils import complex_operator, harmonic_loss_compute
from layers.MtsCID_Losses import EntropyLoss, GatheringLoss
from layers.MtsCID_Memory import create_memory_matrix
# ... 其他组件类似

【使用场景】
✅ 适合复用：频域建模、内存增强、多尺度特征提取、复数运算
⚠️ 需要定制：模型核心架构（Encoder/Decoder）、任务特定的前向逻辑
❌ 不建议复用：与特定模型紧密耦合的组件（如 TransformerVar 内部的 branch 结构）

【最佳实践】
1. 优先检查 layers/MtsCID_*.py 是否有所需功能
2. 复用时保持原有接口和参数名称
3. 若需修改组件行为，继承后重写而非直接修改源码
4. 新模型中避免重复实现已有的通用功能

## multihead_attention_constraint
- Title: 多头注意力维度约束

使用 nn.MultiheadAttention 时，embed_dim (通常是 d_model) 必须能被 num_heads 整除。【常见有效组合】d_model=64, n_heads=8 (每头8维)；d_model=128, n_heads=8 (每头16维)；d_model=256, n_heads=8 (每头32维)。【错误示例】d_model=55, n_heads=8 会导致 AssertionError。【解决方法】(1) 调整 d_model 为 n_heads 的倍数；(2) 调整 n_heads 为 d_model 的因子；(3) 在代码中添加验证：assert d_model % n_heads == 0, f'd_model={d_model} must be divisible by n_heads={n_heads}'。

## frequency_domain_attention_constraint
- Title: 频域注意力的维度约束

在频域进行注意力计算时，频率分量数量 f = seq_len // 2 + 1 必须能被 num_heads 整除。【约束公式】(seq_len // 2 + 1) % n_heads == 0。【常见有效组合】seq_len=94, n_heads=8 (f=48)；seq_len=110, n_heads=8 (f=56)；seq_len=126, n_heads=8 (f=64)；seq_len=100, n_heads=3 (f=51)。【解决方法】(1) 调整 seq_len 使 f 能被 n_heads 整除；(2) 在代码中动态调整 n_heads 为 f 的因子；(3) 使用 padding 将 f 补齐到 n_heads 的倍数。【错误示例】seq_len=100, n_heads=8 会导致 f=51，51 % 8 ≠ 0，触发 AssertionError。

## exp_anomaly_detection_interface
- Title: exp_anomaly_detection.py 要求的模型接口

【关键接口要求】exp_anomaly_detection.py 要求异常检测模型必须实现以下三个方法：

【1. forward() - 标准前向传播】
签名：forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
返回：重构输出 (batch_size, seq_len, c_out)
用途：标准 PyTorch 接口，由 task_name 分发到具体任务方法

【2. forward_with_loss() - 训练时使用 ⚠️ 必须实现】
签名：forward_with_loss(self, x_enc)
返回：tuple (output, entropy_loss)
  - output: 重构输出 (batch_size, seq_len, c_out)
  - entropy_loss: 标量张量，用于正则化（如注意力熵、原型熵等）
调用位置：exp_anomaly_detection.py 的 train() 方法中
实现要点：
  - 执行完整的前向传播得到重构输出
  - 计算辅助损失（如 EntropyLoss、对比损失等）
  - 返回两者的元组
示例实现：

def forward_with_loss(self, x_enc):
    output_dict = self.model(x_enc)  # 模型前向
    output = output_dict['reconstruction']  # 重构输出
    attn = output_dict.get('attention_weights', None)
    # 计算熵损失（如果有注意力权重）
    if attn is not None:
        entropy_loss = self.entropy_loss(attn)
    else:
        entropy_loss = torch.tensor(0.0, device=output.device)
    return output, entropy_loss
```

【3. compute_anomaly_score() - 推理时使用 ⚠️ 必须实现】
签名：compute_anomaly_score(self, x_enc)
返回：anomaly_scores (batch_size, seq_len)
调用位置：exp_anomaly_detection.py 的 vali()/test() 方法中
实现要点：
  - 计算重构误差（通常是 MSE 或 MAE）
  - 可选：结合其他异常指标（如内存距离、注意力异常等）
  - 返回每个时间步的异常分数（越高越异常）
示例实现：

def compute_anomaly_score(self, x_enc):
    output_dict = self.model(x_enc)
    output = output_dict['reconstruction']
    # 计算重构误差
    rec_loss = F.mse_loss(x_enc, output, reduction='none')  # (B, L, C)
    # 可选：结合其他异常指标
    if 'memory_distance' in output_dict:
        mem_dist = output_dict['memory_distance']  # (B, L)
        # 融合两种分数
        rec_score = rec_loss.mean(dim=-1)  # (B, L)
        score = rec_score * torch.softmax(mem_dist, dim=-1)
    else:
        # 仅使用重构误差
        score = rec_loss.mean(dim=-1)  # (B, L)
    return score
```

【常见错误】
❌ 只实现 forward()，缺少 forward_with_loss() 和 compute_anomaly_score()
❌ forward_with_loss() 只返回 output，忘记返回 entropy_loss
❌ compute_anomaly_score() 返回形状错误（应该是 (B, L) 而非 (B, L, C)）
❌ entropy_loss 不是标量张量（应该是 0 维张量，不是 Python float）

【检查清单】提交前确认：
✅ Model 类包含 forward_with_loss(self, x_enc) 方法
✅ Model 类包含 compute_anomaly_score(self, x_enc) 方法
✅ forward_with_loss 返回 (output, entropy_loss) 元组
✅ compute_anomaly_score 返回 (batch_size, seq_len) 形状的分数
✅ entropy_loss 是 torch.Tensor 类型（即使为 0 也用 torch.tensor(0.0)）

## timemixer_channel_attention_implementation
- Title: TimeMixer++ Channel Attention 正确实现方式

【关键问题】TimeMixer++ 的 Channel Attention 必须在**通道维度（C）**上进行自注意力计算，而不是时间维度。实现时必须确保线性层的输入维度与通道数匹配。

【错误实现 ❌】

class ChannelAttention(nn.Module):
    def __init__(self, dim, heads=8, ...):
        self.to_qkv = nn.Linear(dim, inner_dim * 3)  # dim 应该是 enc_in
    
    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T] ❌ 错误！
        qkv = self.to_qkv(x)  # 期望输入 [..., dim]，但实际是 [..., T]
        # 这会导致维度不匹配错误！
```
错误原因：转置后最后一维变成了 T（序列长度），但 to_qkv 期望的是 C（通道数）。

【正确实现方式1：直接在通道维度计算 ✅】

class ChannelAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head if dim_head is not None else max(dim // heads, 1)
        inner_dim = self.dim_head * heads
        self.scale = self.dim_head ** -0.5
        
        # dim 必须是 configs.enc_in（通道数）
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        x: [B, T, C] - 输入在最粗尺度
        返回: [B, T, C]
        """
        B, T, C = x.shape
        
        # 方法1：reshape 为 [B*T, C]，在 C 维度做注意力
        x_flat = x.reshape(B * T, C)  # [B*T, C]
        
        # 计算 Q, K, V
        qkv = self.to_qkv(x_flat)  # [B*T, inner_dim * 3]
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B * T, self.heads, self.dim_head).transpose(0, 1), qkv)
        # q, k, v: [heads, B*T, dim_head]
        
        # 注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # [heads, B*T, dim_head]
        out = out.transpose(0, 1).reshape(B * T, -1)  # [B*T, inner_dim]
        out = self.to_out(out)  # [B*T, C]
        
        # Reshape 回 [B, T, C]
        out = out.reshape(B, T, C)
        return out
```

【正确实现方式2：在时间步内做通道注意力 ✅】

def forward(self, x):
    """
    x: [B, T, C]
    返回: [B, T, C]
    """
    B, T, C = x.shape
    
    # 对每个时间步独立做通道注意力
    # 保持 [B, T, C] 格式，在最后一维（C）上计算
    qkv = self.to_qkv(x)  # [B, T, inner_dim * 3]
    qkv = qkv.chunk(3, dim=-1)
    q, k, v = map(lambda t: t.reshape(B, T, self.heads, self.dim_head).permute(0, 2, 1, 3), qkv)
    # q, k, v: [B, heads, T, dim_head]
    
    # 在通道维度做注意力（每个时间步独立）
    attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, T, T]
    attn = F.softmax(attn, dim=-1)
    
    out = torch.matmul(attn, v)  # [B, heads, T, dim_head]
    out = out.permute(0, 2, 1, 3).reshape(B, T, -1)  # [B, T, inner_dim]
    out = self.to_out(out)  # [B, T, C]
    return out
```

【初始化要点】

# 在 Model.__init__ 中
self.channel_attention = ChannelAttention(
    dim=configs.enc_in,  # ✅ 必须是通道数
    heads=min(8, configs.enc_in),  # 确保 heads 不超过通道数
    dim_head=max(configs.enc_in // min(8, configs.enc_in), 1),
    dropout=configs.dropout
)
```

【使用方式】

def anomaly_detection(self, x_enc):
    # x_enc: [B, seq_len, enc_in]
    
    # 多尺度下采样
    multi_scale_x = self.downsampling(x_enc)  # 返回列表
    
    # 在最粗尺度应用 channel attention
    x_M = multi_scale_x[-1]  # [B, T_coarse, C]
    x_M = self.channel_attention(x_M)  # [B, T_coarse, C]
    multi_scale_x[-1] = x_M
    
    # 继续后续处理...
```

【关键检查清单】
✅ ChannelAttention.__init__ 的 dim 参数 = configs.enc_in
✅ forward 方法中，to_qkv 的输入最后一维必须是 C（通道数）
✅ 不要在转置后对 T 维度调用线性层
✅ 输入输出形状保持 [B, T, C]
✅ 确保 enc_in % heads == 0 或动态调整 heads

【常见错误】
❌ x.transpose(1, 2) 后直接调用 self.to_qkv(x)
❌ 在 [B, C, T] 格式上对最后一维（T）做线性变换
❌ 混淆了通道注意力和时间注意力的实现方式
