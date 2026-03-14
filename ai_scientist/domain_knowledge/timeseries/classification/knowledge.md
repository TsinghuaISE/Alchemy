# Domain Knowledge

- Domain: TimeSeries
- Task: classification

## tsl_framework_spec
- Title: Time Series Library 框架规范

模型类命名为 Model，继承 nn.Module。__init__ 接收 configs 参数，包含 task_name, seq_len, enc_in, num_class, d_model, n_heads, e_layers, d_ff, dropout 等配置。必须实现 classification(x_enc, x_mark_enc) 方法返回分类 logits，以及 forward() 方法调用 classification。输入 x_enc 形状为 [B, seq_len, enc_in]，x_mark_enc 为 padding 掩码（有效位为1），输出形状为 [B, num_class]。分类使用 CrossEntropyLoss。若模型需要特殊组件而框架现有组件不支持所需参数，应在模型文件中内联定义该组件。

## config_params
- Title: 支持的超参数列表

run.py 支持的参数：【基础】task_name, is_training, model_id, model, data, root_path, data_path, features, target, freq, checkpoints；【序列】seq_len, label_len, pred_len；【模型】enc_in, dec_in, c_out, d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, factor, dropout, embed, activation, top_k, num_kernels；【训练】train_epochs, batch_size, patience, learning_rate, loss, lradj。布尔参数（distil, inverse等）只能设 true 或不写。未列出参数会报错，代码中用 getattr(configs, 'param', default) 设置默认值。

## task_adaptation_semantics
- Title: 任务适配时的语义变化

预测模型适配到分类任务时，输入输出语义发生根本变化：【输入变化】x_mark_enc 从时间特征 [B,L,d_temporal] 变为 padding mask [B,L]；Embedding 层不应对 padding mask 做时间编码，应传 None。【输出变化】从序列预测 [B,pred_len,C] 变为类别 logits [B,num_class]；移除 Decoder 及其依赖（如外推、阻尼等预测专用组件）。【组件复用原则】Encoder 的特征提取逻辑（注意力、FFN、归一化）通常可直接复用；Decoder、Growth Damping、Seasonal Extrapolation 等预测专用组件应移除或简化。

## classification_head_pattern
- Title: 分类头实现模式

分类任务采用 Encoder-only 架构，输出通过展平+线性投影得到类别 logits。【标准流程】enc_out → 激活(GELU) → Dropout → 掩码屏蔽 → 展平 → 线性投影。掩码操作：output = output * x_mark_enc.unsqueeze(-1) 将 padding 位置置零。展平：output.reshape(B, -1)。【投影维度】取决于编码器输出形状：若输出 [B, L, D] 则投影 L*D → num_class；若输出 [B, C, D] 则投影 C*D → num_class。简化版本可省略激活和掩码，直接 flatten+projection。

## sequence_representation
- Title: 序列表示策略

将变长时序转为固定维度表示的三种范式：(1) 时间步token化：每个时间步投影为d_model维向量，输出[B,L,D]，展平得 L*D 维特征；(2) 变量token化：每个变量的完整序列投影为d_model维，输出[B,C,D]，展平得 C*D 维特征；(3) Patch token化：序列切分为固定长度patch后投影，输出[B,patch_num,D]。选择依据：时间步token化适合捕获时序动态，变量token化适合多变量关联建模，Patch token化适合捕获局部模式。

## temporal_modeling_paradigms
- Title: 时序建模范式

时序特征提取的三大范式：(1) 时域建模：直接在时间维度上操作（注意力/卷积/线性层）；(2) 频域建模：FFT变换→频域操作→IFFT重建，天然捕获周期性；(3) 分解建模：移动平均分离 trend/seasonal 分量分别处理。这三种范式可独立使用或组合。分类任务通常只需编码器提取特征，无需预测未来序列。

## normalization_practice
- Title: 归一化实践

时序分类常用的归一化：(1) 实例归一化（在 classification 方法内）：对每个样本沿时间维度归一化 x = (x - x.mean(1,keepdim=True)) / x.std(1,keepdim=True)，需 .detach() 防止梯度回传；(2) LayerNorm：用于 Transformer 层输出归一化；(3) BatchNorm1d：偶用于卷积层后。实例归一化可消除样本间分布差异，是时序任务的常见预处理。

## vectorization_principle
- Title: 向量化计算原则

【禁止 Python 循环遍历张量维度】在 forward 中使用 for 循环遍历 batch/时间步/特征维度会导致训练卡死或极慢（无法 GPU 并行）。【典型反例】for d in range(D): output[:,d] = func(x[:,d]) 或 for t in range(L): x[t] = f(x[t-1])。【正确做法】使用 PyTorch 批量操作：einsum、广播、gather/scatter、矩阵乘法。【FFT 卷积】若需对每个维度做相同操作，应 transpose 后在最后一维批量 FFT，而非循环。【.item() 禁用】循环中调用 tensor.item() 强制 GPU→CPU 同步，单次调用耗时可达毫秒级。【顺序依赖】若算法有时间步顺序依赖（如 RNN），使用 PyTorch 内置 RNN/LSTM 或 cumsum/cumprod 近似。

## padding_mask_mandatory
- Title: 分类任务必须使用 Padding Mask

【强制规则】在 classification 方法中，展平前必须执行：output = output * x_mark_enc.unsqueeze(-1)。【原因】UEA数据集是变长序列，padding位置若不置零会引入噪声，导致性能下降（PEMS-SF从96%降到89%）。【检查方法】搜索代码中是否有 'x_mark_enc.unsqueeze(-1)'，若缺失则为严重bug。【标准流程】enc_out → act → dropout → **mask** → reshape → projection。

## multiscale_use_first_only
- Title: 多尺度模型分类时只用第一尺度

【关键原则】若模型生成多尺度特征 enc_out_list = [scale_0, scale_1, ...]，分类时使用 enc_out = enc_out_list[0]，**禁止** torch.cat(enc_out_list)。【原因】(1) 分类需要细节信息，粗尺度会丢失细节；(2) 拼接所有尺度导致投影层输入维度过大（如从9K到16K），小样本数据集易过拟合。【参考】TimeMixer原版：enc_out = enc_out_list[0]。

## classification_head_pattern
- Title: 分类头标准实现（五步流程）

【标准流程】(1) 取编码器输出：若多尺度用 enc_out = enc_out_list[0]；(2) 激活：output = self.act(enc_out)；(3) Dropout：output = self.dropout(output)；(4) **Mask（必须）**：output = output * x_mark_enc.unsqueeze(-1)；(5) 展平+投影：output = output.reshape(B, -1); output = self.projection(output)。【投影层】self.projection = nn.Linear(seq_len * d_model, num_class)。【参考】TimeMixer/TimesNet的classification方法。

## hyperparameter_generation_rules
- Title: 超参数生成核心规则

【类型约束】(1) choices 参数：值必须在允许列表中，如 beta_schedule 只能是 ['constant', 'cosine', 'linear'] 之一，不能传数值；(2) action='store_true' 布尔参数：YAML 中只能写 true 或省略，不能写 false；(3) nargs='+' 列表参数（如 depths）：使用嵌套列表格式 depths: [[2,2,2,1]]。【参数命名】严格使用 run.py 中定义的参数名，注意单复数（num_shapelet 不是 num_shapelets）。【禁止行为】不要创造不存在的参数（如 InterpGN 没有 beta、use_hybrid、eta_threshold 参数）；不要混淆相似参数（beta vs beta_schedule, gating_value vs eta_threshold）。【验证方法】生成后检查：choices 参数的值是否为字符串且在允许列表中？参数名是否在 run.py 中存在？

## model_specific_hyperparameters
- Title: 模型特定超参数定义

【InterpGN】dnn_type: choices=['FCN','Transformer','TimesNet','PatchTST','ResNet']; num_shapelet: int; lambda_reg/lambda_div/epsilon: float; beta_schedule: choices=['constant','cosine','linear']（注意：不存在单独的 beta 参数）; gating_value: float 或 null; distance_func: choices=['euclidean','cosine','pearson']; sbm_cls: choices=['linear','bilinear','attention']; pos_weight/memory_efficient: bool。【SGN】num_groups/period/block_num/kernel_size/mlp_ratio: int; depths: list (嵌套格式 [[2,2,2,1]]); sgn_embed_loss_weight: float。【TimeMixer】down_sampling_layers/down_sampling_window: int; down_sampling_method: choices=['avg','max']; channel_independence: int; decomp_method: str; moving_avg: int。【常见错误】InterpGN 使用 beta/eta_threshold/use_hybrid（不存在）；SGN 的 depths 未使用嵌套列表；参数名拼写错误（单复数、下划线位置）。

## interpgn_forward_return_format
- Title: InterpGN forward 方法返回格式要求

【关键规则】InterpGN 模型的 forward 方法必须返回 2 个值：(output, model_info)。【ModelInfo 结构】需要定义 ModelInfo 数据类，包含字段：shapelet_preds (SBM 预测), dnn_preds (DNN 预测), preds (最终预测), eta (门控权重), loss (正则化损失，包含 lambda_div * diversity_loss + lambda_reg * l1_loss)。【训练代码期望】exp_classification.py 第 256 行：outputs, model_info = self.model(...)，然后使用 model_info.loss 和 model_info.shapelet_preds 计算额外损失。【错误模式】如果 forward 只返回 logits 会导致 'too many values to unpack' 错误。【参考实现】参考 paperbench_pro/benchmark/TimeSeries/Time_Series_Library/models/InterpGN.py 的标准实现。
