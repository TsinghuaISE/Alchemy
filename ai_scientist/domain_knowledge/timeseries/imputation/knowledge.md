# Domain Knowledge

- Domain: TimeSeries
- Task: imputation

## framework_interface
- Title: 框架接口规范

模型类命名 Model，继承 nn.Module。__init__(self, configs) 提取 task_name、seq_len、enc_in、c_out、d_model 等属性。必须实现 imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask) 方法，forward 调用该方法。输入 x_enc 形状 [B, seq_len, enc_in]，mask 形状相同（1=观测，0=缺失），输出 [B, seq_len, c_out]。【核心约束】输出长度必须等于输入长度，禁止截取。【组件内联】若需特殊组件（如带 moving_avg 参数的 EncoderLayer），应在模型文件中内联定义，避免框架组件 API 签名不匹配。

## config_params
- Title: 超参数规范

run.py 支持的参数：【基础】task_name, is_training, model_id, model, data, root_path, data_path, features, target, freq, checkpoints；【序列】seq_len, label_len, pred_len；【任务特定】mask_rate；【模型】enc_in, dec_in, c_out, d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, factor, dropout, embed, activation, top_k, num_kernels；【训练】train_epochs, batch_size, patience, learning_rate, loss, lradj。【布尔标志】distil, inverse, use_amp, use_multi_gpu, individual 等只能设 true 或不写，禁止 0/false。【禁止参数】output_attention, embed_type, stride 等未列出参数会报错，模型需要的非标准参数用 getattr(configs, 'param', default) 设置默认值。

## normalization_strategy
- Title: 归一化策略

【标准实例归一化】means = x_enc.mean(1, keepdim=True).detach()，stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()，归一化 x = (x - means) / stdev，反归一化 out = out * stdev + means。detach() 阻止梯度回传。【掩码感知归一化】（更精确）只在 mask==1 位置计算统计量：means = sum(x) / sum(mask==1)，归一化后用 masked_fill(mask==0, 0) 置零缺失位。两种方式均需在输出后反归一化。

## tensor_contiguous
- Title: 张量连续性

【view vs reshape】view 要求张量在内存中连续，permute/transpose/FFT 后的张量不连续会报错 'view size is not compatible'。解决方案：(1) 用 reshape 代替 view，会自动处理不连续；(2) 先调用 contiguous() 再 view。【安全模式】涉及 FFT、注意力、多尺度操作后，统一使用 reshape。

## architecture_patterns
- Title: 架构模式

【纯编码器】主流模式：Embedding → Encoder → Linear(d_model, c_out)，输出等长，无需扩展时序。【分解后合成】decomp 分离 trend/seasonal → 分别处理 → 相加，有效分离信号与噪声。【倒置处理】permute(0,2,1) 将变量维作为 token 维度，在变量间做注意力捕捉跨通道依赖。【多尺度】下采样→编码→上采样，在不同分辨率上建模后融合。【MLP 混合】ResBlock 结构交替做时间/通道线性变换加残差，轻量高效。投影层统一用 Linear(hidden_dim, c_out)。

## decomposition_technique
- Title: 分解技术

【时域分解】移动平均 AvgPool1d(kernel_size, stride=1) 提取趋势，残差为季节成分。kernel_size 需为奇数，两端 padding 保持长度。【多尺度分解】用多个 kernel_size 分别分解后取平均，对周期不确定的序列更鲁棒。【频域分解】torch.fft.rfft 变换到频域，筛选 top-k 频率或做频谱线性变换，irfft 恢复。分解后分别建模再相加，对缺失数据情况下的插补质量提升显著。

## dimension_consistency
- Title: 维度一致性

【输出等长】插补任务输出必须为 [B, seq_len, c_out]，不做截取，区别于预测任务的 [B, pred_len, c_out]。【分段整除】使用 seg_len/patch_len 时须满足 seq_len % seg_len == 0 或显式填充：pad = ceil(seq_len/seg_len)*seg_len - seq_len，用 ReplicationPad1d 或 zeros 补齐。【反归一化对齐】stdev/means 形状为 [B,1,C]，输出 [B,L,C] 时通过 unsqueeze 和 repeat 对齐，或用广播机制。【reshape 安全】任何 [B,T,C] → [B,T//k,k,C] 的 reshape 需确保 T 能被 k 整除。

## attention_variants
- Title: 注意力变体

【自注意力】标准 Transformer 在时间维度做全局注意力，O(L²) 复杂度，适合短序列。【稀疏注意力】采样 top-k query 或使用 LSH 哈希分桶，降低复杂度到 O(L log L)。【频域注意力】在频谱上做线性变换，等价于循环卷积，天然捕捉周期依赖。【自相关】频域计算 Q·K* 得相关性，时延聚合捕捉周期模式。【双阶段】先在时间维做注意力，再在变量维做路由聚合，分离时序与跨通道依赖。

## dataset_features
- Title: 数据集约束

禁止硬编码 enc_in、dec_in、c_out，由框架根据数据集自动设置。常用数据集特征数：ETTh1/ETTh2/ETTm1/ETTm2=7，Weather=21，ECL=321，Traffic=862，Exchange=8。插补任务通过 mask_rate 参数控制训练时随机掩码比例（如 0.25 表示 25% 缺失），测试时使用相同掩码模式评估重建质量。
