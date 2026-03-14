# Domain Knowledge

- Domain: TimeSeries
- Task: short_term_forecast

## framework_interface
- Title: 框架接口规范

模型类命名 Model，继承 nn.Module。__init__(self, configs) 提取 task_name、seq_len、label_len、pred_len、enc_in、d_model 等属性。forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) 返回 dec_out[:, -self.pred_len:, :]。输入 x_enc 形状 [B, seq_len, enc_in]，x_mark_enc 为时间戳特征或 None，输出 [B, pred_len, c_out]。短期预测 pred_len 通常为 12、24、48。【组件自定义】若模型需要特殊组件（如带 moving_avg 的 EncoderLayer），而框架现有组件不支持所需参数，应在模型文件中内联定义该组件，避免 API 签名不匹配导致运行时错误。

## config_params
- Title: 支持的超参数列表

run.py 支持的参数（只能使用这些）：【基础】task_name, is_training, model_id, model, data, root_path, data_path, features, target, freq, checkpoints；【序列】seq_len, label_len, pred_len, seasonal_patterns；【模型】enc_in, dec_in, c_out, d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, factor, dropout, embed, activation；【高级】channel_independence, decomp_method, use_norm, down_sampling_layers, down_sampling_window, down_sampling_method, top_k, num_kernels；【训练】train_epochs, batch_size, patience, learning_rate, loss, lradj。【短期预测损失函数】loss 只能是 MAPE/MASE/SMAPE，禁止 MSE/MAE（接口不兼容会报错）。【布尔标志参数（只能设 true 或不写，禁止设 0/false）】distil, inverse, use_amp, use_multi_gpu, individual, jitter, scaling, permutation 等。启用时写 true（如 distil: true），不需要时不要写在配置中。【禁止】output_attention, embed_type, detail_freq, stride, num_scales 等未列出参数会报错。

## m4_dataset_constraint
- Title: 短期预测专用 M4 数据集

【重要】短期预测任务专为 M4 数据集设计，与长期预测不同。必须设置 data: m4，并通过 seasonal_patterns 指定频率：Yearly/Quarterly/Monthly/Weekly/Daily/Hourly。框架根据 seasonal_patterns 自动设置 pred_len（Yearly=6, Quarterly=8, Monthly=18, Weekly=13, Daily=14, Hourly=48）、seq_len=2*pred_len、label_len=pred_len。【损失函数】短期预测只支持 MAPE/MASE/SMAPE 损失函数，禁止使用 MSE/MAE，因为 exp_short_term_forecasting.py 使用特殊损失接口 criterion(batch_x, frequency_map, outputs, batch_y, batch_y_mark)。推荐 loss: SMAPE。

## dataset_features
- Title: 数据集特征数

禁止在超参数中硬编码 enc_in、dec_in、c_out，由框架根据数据集自动设置。M4 数据集为单变量预测（enc_in=1, dec_in=1, c_out=1），每条序列长度不固定，通过 seasonal_patterns 决定预测长度。

## instance_norm
- Title: 可逆实例归一化

处理非平稳性的标准做法：means = x_enc.mean(1, keepdim=True).detach()，stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()，归一化 x_enc = (x_enc - means) / stdev，预测后反归一化 dec_out = dec_out * stdev + means。detach() 防止梯度回传到统计量。短期预测因序列更短，归一化更关键。

## decomposition
- Title: 序列分解

移动平均：series_decomp(moving_avg) 用 AvgPool1d 分离趋势和季节成分。频域分解：torch.fft.rfft 得频谱，筛选 top-k 频率分离周期性成分。分解后分别建模再相加，对短期预测尤其有效，因局部模式更显著。

## encoder_decoder
- Title: 编码器-解码器结构

Encoder-Decoder：x_dec 由 label_len 段历史加 pred_len 段零/均值初始化，通过交叉注意力融合编码器输出。Encoder-Only：纯编码器直接 Linear(d_model, pred_len) 投影，更简洁。短期预测因预测长度小，两种结构性能相近，纯编码器更高效。

## attention_variants
- Title: 注意力变体

稀疏注意力 ProbAttention：采样 top-k query 降低复杂度到 O(L log L)。自相关 AutoCorrelation：频域计算周期依赖，通过时延聚合捕捉周期模式。频域注意力 FourierBlock：在频谱上做线性变换，天然捕捉周期性。倒置注意力：permute(0,2,1) 后在变量维度做注意力，捕捉变量间依赖。

## dimension_consistency
- Title: 维度一致性与时序扩展

【时序扩展】部分纯编码器模型的内部模块假设输入长度为 seq_len + pred_len，须在 embedding 后用 predict_linear = Linear(seq_len, seq_len + pred_len) 扩展：enc_out = predict_linear(enc_out.permute(0,2,1)).permute(0,2,1)。【分段整除】使用 seg_len/patch_len 分割序列时，必须满足 seq_len % seg_len == 0 或显式填充。【reshape 安全】[B,T,C]->[B,T//k,k,C] 需确保 T 能被 k 整除。【反归一化对齐】若输出长度变为 seq_len + pred_len，stdev/means 需 repeat 到该长度。
