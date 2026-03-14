# Domain Knowledge

- Domain: TimeSeries
- Task: long_term_forecast

## framework_interface
- Title: 框架接口规范

模型类必须命名为 Model，继承 nn.Module。__init__(self, configs) 从 configs 提取 task_name、seq_len、pred_len、enc_in、d_model 等属性。forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) 返回 dec_out[:, -self.pred_len:, :]。x_enc 形状 [B, seq_len, enc_in]，x_mark_enc 为时间戳 [B, seq_len, time_dim] 或 None，输出 [B, pred_len, c_out]。纯编码器模型可忽略 x_dec/x_mark_dec。

## config_params
- Title: 支持的超参数完整列表

run.py 支持的全部参数（只能使用这些参数，其他参数会报错）：【基础】task_name, is_training, model_id, model, data, root_path, data_path, features, target, freq, checkpoints；【序列长度】seq_len, label_len, pred_len, seasonal_patterns；【任务特定】mask_rate, anomaly_ratio；【模型结构】expand, d_conv, top_k, num_kernels, enc_in, dec_in, c_out, d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, factor, dropout, embed, activation；【高级特性】channel_independence, decomp_method, use_norm, down_sampling_layers, down_sampling_window, down_sampling_method, seg_len；【训练】num_workers, itr, train_epochs, batch_size, patience, learning_rate, des, loss, lradj；【GPU】use_gpu, gpu, gpu_type, devices；【其他】p_hidden_dims, p_hidden_layers, use_dtw, augmentation_ratio, seed, patch_len, node_dim, gcn_depth, gcn_dropout, propalpha, conv_channel, skip_channel, alpha, top_p, pos, extra_tag。【布尔标志参数（只能设 true 或不写，不能设 0/false）】distil, inverse, use_amp, use_multi_gpu, individual, jitter, scaling, permutation, randompermutation, magwarp, timewarp, windowslice, windowwarp, rotation, spawner, dtwwarp, shapedtwwarp, wdba, discdtw, discsdtw。这些参数启用时写 true（如 distil: true），不需要时不要写在配置中。【禁止使用】num_scales, stride, output_attention, embed_type, detail_freq 等未列出的参数会导致运行失败。代码中需要的非标准参数必须用 getattr(configs, 'param', default) 设置默认值。

## dataset_features
- Title: 数据集与特征数对应关系

超参数配置中禁止硬编码 enc_in、dec_in、c_out，这些由数据集自动决定。常用数据集特征数：ETTh1/ETTh2/ETTm1/ETTm2=7，Weather=21，ECL(Electricity)=321，Traffic=862，Exchange=8，ILI=7，PEMS(03/04/07/08)=170/307/883/170。可通过 data 参数指定数据集名称（如 data: ETTh1），框架会自动设置对应的 enc_in/dec_in/c_out。

## instance_norm
- Title: 可逆实例归一化

处理非平稳性的标准做法，几乎所有现代模型都采用：means = x_enc.mean(1, keepdim=True).detach()，stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()，归一化 x_enc = (x_enc - means) / stdev，反归一化 dec_out = dec_out * stdev + means。detach() 阻止梯度回传到统计量。

## decomposition
- Title: 序列分解技术

移动平均分解：series_decomp(kernel_size)，seasonal, trend = self.decomp(x)，内部用 AvgPool1d 平滑得趋势，残差为季节。频域分解：torch.fft.rfft(x, dim=1) 得频谱，筛选 top-k 频率或掩码分离高频（时变）/低频（时不变）成分，torch.fft.irfft 恢复。分解后分别建模再相加是提升性能的有效手段。

## input_processing
- Title: 输入处理策略

Patching：x.unfold(dim=-1, size=patch_len, step=stride) 切块后 Linear(patch_len, d_model) 嵌入，降低注意力复杂度。通道处理：channel_independence=True 时 reshape [B,T,N] 为 [B*N,T,1] 独立处理；False 时特征共享嵌入。通道独立更稳定，通道混合适合强相关变量。FlattenHead 将编码输出展平投影到 pred_len。

## architecture_patterns
- Title: 主流架构模式

纯编码器：DataEmbedding → Encoder → Linear(d_model, pred_len/c_out)，简洁高效。MLP混合器：ResBlock 交替做时间 Linear(seq_len, d_model) 和通道 Linear(enc_in, d_model) 变换加残差。多尺度：Pool/Conv 下采样构建多分辨率表示，季节↑趋势↓方向混合后融合。倒置注意力：permute(0,2,1) 后在变量维度做注意力捕捉变量依赖。

## linear_techniques
- Title: 线性映射技术

直接映射 Linear(seq_len, pred_len) 是强基线。结合分解：分别对 trend/seasonal 用独立 Linear 预测后相加。残差旁路：self.ar = Linear(seq_len, pred_len)，dec_out = model_out + ar(x.permute(0,2,1)).permute(0,2,1)，帮助梯度流动。权重可初始化为 (1/seq_len) * ones([pred_len, seq_len])。

## multivariate_handling
- Title: 多变量与外生变量

倒置嵌入：DataEmbedding_inverted 用 Linear(seq_len, d_model) 将每个变量时间序列嵌入为 token，在变量维度做注意力。外生变量融合：(1) concat 后共同嵌入；(2) 独立编码后交叉注意力；(3) global token 聚合。x_mark_dec 提供未来时间戳，可用于生成预测位置编码。

## time_feature_encoding
- Title: 时间特征编码规范与数据集频率差异

时间序列数据集的频率特性对模型设计至关重要。常见数据集频率：ETTh1/ETTh2为小时级(hourly, freq='h')，ETTm1/ETTm2为15分钟级(15-minute, freq='15min')，Weather为10分钟级。当timeenc=1时，时间特征被归一化到[-0.5, 0.5]范围，特征顺序为[HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]。对于小时级数据，HourOfDay在索引0位置，范围[0,23]映射到[-0.5,0.5]；对于15分钟级数据，时间特征可能包含更细粒度的信息。模型设计时必须考虑：(1)不同频率数据的时间特征维度和顺序可能不同；(2)归一化方式影响embedding索引计算；(3)周期性模式的粒度需要与数据频率匹配(如embed_size=24适合小时级，embed_size=96适合15分钟级)。

## frequency_domain_decomposition
- Title: 频域分解中的静态-动态组件建模

频域分解是时间序列预测的重要技术。核心思想：将时间序列分解为时不变(time-invariant)和时变(time-varying)组件。时不变组件代表长期稳定模式(如日周期、周周期)，时变组件代表局部波动。实现要点：(1)使用FFT将时域信号转换到频域：X_freq = torch.fft.rfft(x, dim=1)，频率分量数为F=L//2+1；(2)通过embedding bank存储时不变模式，索引应基于时间戳信息(如小时、星期)；(3)动态组件通过频域滤波处理：X_dynamic = X_freq - X_static；(4)可学习的复数滤波器ω对动态组件进行频谱调制；(5)IFFT恢复时域信号。关键注意：embedding索引的提取必须与数据集的时间特征编码方式一致，错误的索引会导致检索到错误的周期模式，严重影响性能。

## dataset_specific_patterns
- Title: 数据集特定的周期性模式与超参数适配

不同时间序列数据集具有不同的周期性特征，模型超参数需要相应调整。ETTh1(小时级电力负载)：具有明显的日周期(24小时)和周周期(7天)，适合embed_size=24或168；ETTm1(15分钟级电力负载)：具有更细粒度的日内模式，适合embed_size=96(每天96个15分钟)；Weather(气象数据)：周期性较弱，可能需要更大的embed_size或不同的分解策略；Traffic(交通流量)：工作日/周末模式差异大，需要考虑星期信息。模型在不同数据集上的性能差异往往源于：(1)周期粒度不匹配；(2)时间索引提取错误；(3)归一化策略不当。建议：针对数据集频率调整embed_size，验证时间特征提取逻辑，在多个数据集上测试以发现潜在问题。

## pslstm_architecture
- Title: P-sLSTM完整架构要求

P-sLSTM不是简单的sLSTM，而是完整的xLSTM架构。必须包含：(1)LinearHeadwiseExpand多头投影层，每个头独立处理；(2)CausalConv1d因果卷积(kernel_size可配)，处理局部时间依赖；(3)MultiHeadLayerNorm多头归一化；(4)GatedFeedForward门控FFN(proj_factor=1.3)；(5)sLSTMBlock包含残差连接xlstm_norm+xlstm+ffn_norm+ffn；(6)xLSTMBlockStack可堆叠多块。核心forward流程：patching→embedding→conv1d+激活→四门投影(i,f,z,o)→sLSTMCell→dropout→group_norm→残差→FFN→投影到pred_len。缺少任何组件都会显著降低性能。

## data_frequency_adaptation
- Title: 数据频率自适应的patch设计

不同频率数据需要不同patch_size以匹配时间粒度。小时级(ETTh1/ETTh2)：patch_size=12，stride=12，每天24点分2个patch；15分钟级(ETTm1/ETTm2)：patch_size=6，stride=6，每天96点分16个patch；10分钟级(Weather)：patch_size=56，stride=56。规律：高频数据用小patch捕捉细粒度模式，低频数据用大patch捕捉长期趋势。patch_num=(seq_len-patch_size)//stride+1，影响最终投影维度。错误的patch_size会导致高频数据(如ETTm1)性能大幅下降(可能损失10%+)，而低频数据相对稳定。

## pred_len_specific_tuning
- Title: 预测长度特定的超参数调优

同一模型在不同pred_len下需要不同超参数。P-sLSTM示例：pred_len=96时用num_heads=2,conv1d_kernel_size=32,dropout=0.1(需要更多正则化)；pred_len=192/336时用num_heads=4,conv1d_kernel_size=4,dropout=0.0(更大容量)；pred_len=720时用num_heads=4,conv1d_kernel_size=2,dropout=0.0(最小卷积核)。长预测需要更大感受野(更多头)，短预测需要更强正则化。使用_pred_len_overrides字典存储这些配置。固定超参数会导致某些pred_len性能不佳。

## conv1d_importance
- Title: 因果卷积在时序建模中的关键作用

CausalConv1d是RNN/LSTM架构的重要增强。作用：(1)捕捉局部时间依赖，补充LSTM的全局记忆；(2)增加感受野，kernel_size控制局部窗口大小；(3)提供归纳偏置，强制时间因果性。实现要点：padding=kernel_size-1保证因果性，groups=feature_dim实现depthwise卷积(通道独立)，SiLU激活函数。P-sLSTM中conv1d_kernel_size从2到32变化，小kernel适合长预测(捕捉趋势)，大kernel适合短预测(捕捉细节)。缺少卷积层会使模型在高频数据上性能下降10-15%。

## architecture_simplification_risk
- Title: 架构简化的性能风险评估

简化SOTA模型架构会导致不均匀的性能下降。风险因素：(1)数据频率：高频数据(15min级)对架构完整性更敏感，简化可能损失10-20%性能；低频数据(小时级)相对稳定，损失2-5%；(2)关键组件：多头机制、卷积层、FFN缺一不可，缺少任何一个都会显著影响性能；(3)超参数适配：固定超参数无法适应不同数据集/pred_len，导致某些场景性能崩溃。建议：复现时保持架构完整性，使用数据集特定的超参数配置，在多个数据集上验证以发现潜在问题。
