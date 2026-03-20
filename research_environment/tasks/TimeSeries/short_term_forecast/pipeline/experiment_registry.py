# -*- coding: utf-8 -*-
TASK_DATA_MAP = {
    'anomaly_detection': {
        'MSL', 'PSM'
    },
    'classification': {
        'Heartbeat', 'PEMS-SF'
    },
    'imputation': [
        'ETTh1', 'Weather'
    ],
    'long_term_forecast': [
        'ETTh1', 'ETTm1'
    ],
    'short_term_forecast': [
        'M4'
    ]
}
# TASK_DATA_MAP = {
#     'anomaly_detection': {
#         'MSL', 'PSM', 'SMAP', 'SMD', 'SWaT'
#     },
#     'classification': {
#         'EthanolConcentration', 'FaceDetection', 'Handwriting', 'Heartbeat', 'JapaneseVowels', 'PEMS-SF', 'SelfRegulationSCP1',
#         'SelfRegulationSCP2', 'SpokenArabicDigits', 'UWaveGestureLibrary'
#     },
#     'imputation': [
#         'ECL', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather'
#     ],
#     'long_term_forecast': [
#         'Exchange', 'ECL', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ILI', 'Traffic', 'Weather'
#     ],
#     'short_term_forecast': [
#         'M4'
#     ]
# }

# DATASET_REGISTRY = {
#     "MSL": {
#         "root_path": "./dataset/MSL",
#         "data": "MSL",
#         "features": "M",
#         "enc_in": 55,
#         "c_out": 55
#     },
#     "PSM": {
#         "root_path": "./dataset/PSM",
#         "data": "PSM",
#         "features": "M",
#         "enc_in": 25,
#         "c_out": 25
#     },
#     "SMAP": {
#         "root_path": "./dataset/SMAP",
#         "data": "SMAP",
#         "features": "M",
#         "enc_in": 25,
#         "c_out": 25
#     },
#     "SMD": {
#         "root_path": "./dataset/SMD",
#         "data": "SMD",
#         "features": "M",
#         "enc_in": 38,
#         "c_out": 38
#     },
#     "SWaT": {
#         "root_path": "./dataset/SWaT",
#         "data": "SWaT",
#         "features": "M",
#         "enc_in": 51,
#         "c_out": 51
#     },
#     "EthanolConcentration": {
#         "root_path": "./dataset/EthanolConcentration",
#         "data": "UEA",
#         "model_id": "EthanolConcentration"
#     },
#     "FaceDetection": {
#         "root_path": "./dataset/FaceDetection",
#         "data": "UEA",
#         "model_id": "FaceDetection"
#     },
#     "Handwriting": {
#         "root_path": "./dataset/Handwriting",
#         "data": "UEA",
#         "model_id": "Handwriting"
#     },
#     "Heartbeat": {
#         "root_path": "./dataset/Heartbeat",
#         "data": "UEA",
#         "model_id": "Heartbeat"
#     },
#     "JapaneseVowels": {
#         "root_path": "./dataset/JapaneseVowels",
#         "data": "UEA",
#         "model_id": "JapaneseVowels"
#     },
#     "PEMS-SF": {
#         "root_path": "./dataset/PEMS-SF",
#         "data": "UEA",
#         "model_id": "PEMS-SF"
#     },
#     "SelfRegulationSCP1": {
#         "root_path": "./dataset/SelfRegulationSCP1",
#         "data": "UEA",
#         "model_id": "SelfRegulationSCP1"
#     },
#     "SelfRegulationSCP2": {
#         "root_path": "./dataset/SelfRegulationSCP2",
#         "data": "UEA",
#         "model_id": "SelfRegulationSCP2"
#     },
#     "SpokenArabicDigits": {
#         "root_path": "./dataset/SpokenArabicDigits",
#         "data": "UEA",
#     },
#     "UWaveGestureLibrary": {
#         "root_path": "./dataset/UWaveGestureLibrary",
#         "data": "UEA",
#         "model_id": "UWaveGestureLibrary"
#     },
#     "ECL": {
#         "root_path": "./dataset/electricity",
#         "data_path": "electricity.csv",
#         "data": "custom",
#         "features": "M",
#         "enc_in": 321,
#         "dec_in": 321,
#         "c_out": 321
#     },
#     "EPF_NP": {
#         "root_path": "./dataset/EPF",
#         "data_path": "NP.csv",
#         "data": "custom",
#         "features": "MS",
#         "enc_in": 3,
#         "dec_in": 3,
#         "c_out": 1
#     },
#     "EPF_PJM": {
#         "root_path": "./dataset/EPF",
#         "data_path": "PJM.csv",
#         "data": "custom",
#         "features": "MS",
#         "enc_in": 3,
#         "dec_in": 3,
#         "c_out": 1
#     },
#     "EPF_BE": {
#         "root_path": "./dataset/EPF",
#         "data_path": "BE.csv",
#         "data": "custom",
#         "features": "MS",
#         "enc_in": 3,
#         "dec_in": 3,
#         "c_out": 1
#     },
#     "EPF_FR": {
#         "root_path": "./dataset/EPF",
#         "data_path": "FR.csv",
#         "data": "custom",
#         "features": "MS",
#         "enc_in": 3,
#         "dec_in": 3,
#         "c_out": 1
#     },
#     "EPF_DE": {
#         "root_path": "./dataset/EPF",
#         "data_path": "DE.csv",
#         "data": "custom",
#         "features": "MS",
#         "enc_in": 3,
#         "dec_in": 3,
#         "c_out": 1
#     },
#     "ETTh1": {
#         "root_path": "./dataset/ETT-small",
#         "data_path": "ETTh1.csv",
#         "data": "ETTh1",
#         "features": "M",
#         "enc_in": 7,
#         "dec_in": 7,
#         "c_out": 7
#     },
#     "ETTh2": {
#         "root_path": "./dataset/ETT-small",
#         "data_path": "ETTh2.csv",
#         "data": "ETTh2",
#         "features": "M",
#         "enc_in": 7,
#         "dec_in": 7,
#         "c_out": 7
#     },
#     "ETTm1": {
#         "root_path": "./dataset/ETT-small",
#         "data_path": "ETTm1.csv",
#         "data": "ETTm1",
#         "features": "M",
#         "enc_in": 7,
#         "dec_in": 7,
#         "c_out": 7
#     },
#     "ETTm2": {
#         "root_path": "./dataset/ETT-small",
#         "data_path": "ETTm2.csv",
#         "data": "ETTm2",
#         "features": "M",
#         "enc_in": 7,
#         "dec_in": 7,
#         "c_out": 7
#     },
#     "Traffic": {
#         "root_path": "./dataset/traffic",
#         "data_path": "traffic.csv",
#         "data": "custom",
#         "features": "M",
#         "enc_in": 862,
#         "dec_in": 862,
#         "c_out": 862
#     },
#     "Weather": {
#         "root_path": "./dataset/weather",
#         "data_path": "weather.csv",
#         "data": "custom",
#         "features": "M",
#         "enc_in": 21,
#         "dec_in": 21,
#         "c_out": 21
#     },
#     "Exchange": {
#         "root_path": "./dataset/exchange_rate",
#         "data_path": "exchange_rate.csv",
#         "data": "custom",
#         "features": "M",
#         "enc_in": 8,
#         "dec_in": 8,
#         "c_out": 8
#     },
#     "ILI": {
#         "root_path": "./dataset/illness",
#         "data_path": "national_illness.csv",
#         "data": "custom",
#         "features": "M",
#         "enc_in": 7,
#         "dec_in": 7,
#         "c_out": 7
#     },
#     "m4": {
#         "root_path": "./dataset/m4",
#         "data": "m4",
#         "features": "M",
#         "enc_in": 1,
#         "dec_in": 1,
#         "c_out": 1,
#         "loss": "SMAPE"
#     },
#     # M4 数据集的大写别名（修复 KeyError: 'M4' 问题）
#     "M4": {
#         "root_path": "./dataset/m4",
#         "data": "m4",  # 注意：data 字段必须是小写 m4
#         "features": "M",
#         "enc_in": 1,
#         "dec_in": 1,
#         "c_out": 1,
#         "loss": "SMAPE"
#     }
# }

# ==================== 模型-数据集 特定参数覆盖表 ====================
MODEL_PARAM_OVERRIDES = {
    "MICN": {
        "EthanolConcentration": {"c_out": 3}
    },
    "FiLM": {
        "EthanolConcentration": {"seq_len": 1751, "pred_len": 1751}, 
    },
    "TimesNet": {
        "FaceDetection": {"num_kernels": 4},
    },
    "iTransformer": {
        "EthanolConcentration": {"d_model": 2048},
        "FaceDetection": {"d_model": 128},
        "Handwriting": {"d_model": 128},
        "Heartbeat": {"d_model": 128},
        "JapaneseVowels": {"d_model": 128},
        "PEMS-SF": {"d_model": 128},
        "SelfRegulationSCP1": {"d_model": 128},
        "SelfRegulationSCP2": {"d_model": 128},
        "SpokenArabicDigits": {"d_model": 128},
        "UWaveGestureLibrary": {"d_model": 128}
    },
    "TimesNet": {
        "EthanolConcentration": {"d_model": 16, "d_ff": 32, "e_layers": 2, "train_epochs": 30, "top_k": 3},
        "FaceDetection": {"d_model": 64, "d_ff": 256, "e_layers": 2, "train_epochs": 30, "top_k": 3},
        "Handwriting": {"d_model": 32, "d_ff": 64, "e_layers": 2, "train_epochs": 30, "top_k": 3},
        "Heartbeat": {"d_model": 16, "d_ff": 32, "e_layers": 3, "train_epochs": 30, "top_k": 1},
        "JapaneseVowels": {"d_model": 16, "d_ff": 32, "e_layers": 2, "train_epochs": 60, "top_k": 3},
        "PEMS-SF": {"d_model": 64, "d_ff": 64, "e_layers": 6, "train_epochs": 30, "top_k": 3},
        "SelfRegulationSCP1": {"d_model": 16, "d_ff": 32, "e_layers": 3, "train_epochs": 30, "top_k": 3},
        "SelfRegulationSCP2": {"d_model": 32, "d_ff": 32, "e_layers": 3, "train_epochs": 30, "top_k": 3},
        "SpokenArabicDigits": {"d_model": 32, "d_ff": 32, "e_layers": 2, "train_epochs": 30, "top_k": 2},
        "UWaveGestureLibrary": {"d_model": 32, "d_ff": 64, "e_layers": 2, "train_epochs": 30, "top_k": 3}
    },
    "PSWI": {
        # 所有数据集的通用PSWI参数
        "_default": {
            "pswi_n_pairs": 1,
            "pswi_noise": 0.01,
            "pswi_reg_sk": 1.0,
            "pswi_numItermax": 1000,
            "pswi_stopThr": 1.0e-09,
            "pswi_normalize": 1,  # 修改为1以匹配原版默认值
            "mask_rate": [0.1, 0.3, 0.5, 0.7],
        },
        "ETTh1": {
            "pswi_lr": 0.01,
            "pswi_n_epochs": 200,
            "pswi_batch_size": 256,
            "mask_rate": [0.1, 0.3, 0.5, 0.7],
        },
        "ETTh2": {
            "pswi_lr": 0.01,
            "pswi_n_epochs": 200,
            "pswi_batch_size": 256
        },
        "ETTm1": {
            "pswi_lr": 0.01,
            "pswi_n_epochs": 200,
            "pswi_batch_size": 256
        },
        "ETTm2": {
            "pswi_lr": 0.01,
            "pswi_n_epochs": 200,
            "pswi_batch_size": 256
        },
        "ECL": {
            "pswi_lr": 0.05,
            "pswi_n_epochs": 200,
            "pswi_batch_size": 256
        },
        "Traffic": {
            "pswi_lr": 0.1,
            "pswi_n_epochs": 300,
            "pswi_batch_size": 256
        },
        "Weather": {
            "pswi_lr": 0.01,
            "pswi_n_epochs": 200,
            "pswi_batch_size": 256,
            "mask_rate": [0.1, 0.3, 0.5, 0.7],
        }
    },
    # ------------------------------------------------------------------
    # P-sLSTM overrides (faithful to original scripts, with pred_len-specific settings)
    # ------------------------------------------------------------------
    "P_sLSTM": {
        # Weather (21 channels)
        "Weather": {
            "pslstm_patch_size": 56,
            "pslstm_stride": 56,
            "pslstm_embedding_dim": 100,
            "pslstm_conv1d_kernel_size": 8,
            "dropout": 0.1,
            "_pred_len_overrides": {
                96: {"pslstm_num_heads": 2, "pslstm_num_blocks": 2},
                192: {"pslstm_num_heads": 4, "pslstm_num_blocks": 2},
                336: {"pslstm_num_heads": 2, "pslstm_num_blocks": 1},
                720: {"pslstm_num_heads": 2, "pslstm_num_blocks": 2},
            },
        },
        # ECL / Electricity (321 channels)
        "ECL": {
            "pslstm_embedding_dim": 600,
            "pslstm_num_heads": 3,
            "pslstm_num_blocks": 1,
            "dropout": 0.1,
            "learning_rate": 0.0005,
            "_pred_len_overrides": {
                96: {"pslstm_patch_size": 56, "pslstm_stride": 56, "pslstm_conv1d_kernel_size": 8},
                192: {"pslstm_patch_size": 16, "pslstm_stride": 16, "pslstm_conv1d_kernel_size": 32},
                336: {"pslstm_patch_size": 16, "pslstm_stride": 16, "pslstm_conv1d_kernel_size": 32},
                720: {"pslstm_patch_size": 16, "pslstm_stride": 16, "pslstm_conv1d_kernel_size": 32},
            },
        },
        # ETTm1 (7 channels)
        "ETTm1": {
            "pslstm_patch_size": 6,
            "pslstm_stride": 6,
            "pslstm_embedding_dim": 100,
            "pslstm_num_blocks": 1,
            "_pred_len_overrides": {
                96: {"pslstm_num_heads": 2, "pslstm_conv1d_kernel_size": 32, "dropout": 0.1},
                192: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 4, "dropout": 0.0},
                336: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 4, "dropout": 0.0},
                720: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 2, "dropout": 0.0},
            },
        },
        # ETTm2 (7 channels) - mirror ETTm1
        "ETTm2": {
            "pslstm_patch_size": 6,
            "pslstm_stride": 6,
            "pslstm_embedding_dim": 100,
            "pslstm_num_blocks": 1,
            "_pred_len_overrides": {
                96: {"pslstm_num_heads": 2, "pslstm_conv1d_kernel_size": 32, "dropout": 0.1},
                192: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 4, "dropout": 0.0},
                336: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 4, "dropout": 0.0},
                720: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 2, "dropout": 0.0},
            },
        },
        # ETTh1 (7 channels) - hour-level, larger patch
        "ETTh1": {
            "pslstm_patch_size": 12,
            "pslstm_stride": 12,
            "pslstm_embedding_dim": 100,
            "pslstm_num_blocks": 1,
            "_pred_len_overrides": {
                96: {"pslstm_num_heads": 2, "pslstm_conv1d_kernel_size": 32, "dropout": 0.1},
                192: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 4, "dropout": 0.0},
                336: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 4, "dropout": 0.0},
                720: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 2, "dropout": 0.0},
            },
        },
        # ETTh2 (7 channels) - hour-level
        "ETTh2": {
            "pslstm_patch_size": 12,
            "pslstm_stride": 12,
            "pslstm_embedding_dim": 100,
            "pslstm_num_blocks": 1,
            "_pred_len_overrides": {
                96: {"pslstm_num_heads": 2, "pslstm_conv1d_kernel_size": 32, "dropout": 0.1},
                192: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 4, "dropout": 0.0},
                336: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 4, "dropout": 0.0},
                720: {"pslstm_num_heads": 4, "pslstm_conv1d_kernel_size": 2, "dropout": 0.0},
            },
        },
        # Traffic (862 channels) - aligned with ECL settings
        "Traffic": {
            "pslstm_embedding_dim": 600,
            "pslstm_num_heads": 3,
            "pslstm_num_blocks": 1,
            "dropout": 0.1,
            "learning_rate": 0.0005,
            "_pred_len_overrides": {
                96: {"pslstm_patch_size": 56, "pslstm_stride": 56, "pslstm_conv1d_kernel_size": 8},
                192: {"pslstm_patch_size": 16, "pslstm_stride": 16, "pslstm_conv1d_kernel_size": 32},
                336: {"pslstm_patch_size": 16, "pslstm_stride": 16, "pslstm_conv1d_kernel_size": 32},
                720: {"pslstm_patch_size": 16, "pslstm_stride": 16, "pslstm_conv1d_kernel_size": 32},
            },
        },
        # Exchange (8 channels)
        "Exchange": {
            "pslstm_patch_size": 12,
            "pslstm_stride": 12,
            "pslstm_embedding_dim": 100,
            "pslstm_num_heads": 2,
            "pslstm_num_blocks": 1,
            "pslstm_conv1d_kernel_size": 4,
            "dropout": 0.1,
        },
        # ILI (7 channels) - short sequence
        "ILI": {
            "pslstm_patch_size": 4,
            "pslstm_stride": 4,
            "pslstm_embedding_dim": 100,
            "pslstm_num_heads": 2,
            "pslstm_num_blocks": 1,
            "pslstm_conv1d_kernel_size": 2,
            "dropout": 0.1,
        },
        # M4 (short-term forecasting)
        "M4": {
            "pslstm_patch_size": 8,
            "pslstm_stride": 4,
            "pslstm_embedding_dim": 64,
            "pslstm_num_heads": 2,
            "pslstm_num_blocks": 2,
            "pslstm_conv1d_kernel_size": 4,
            "dropout": 0.05,
        },
    },
    "CATCH": {
        "_default": {
            "catch_patch_size": 16,
            "catch_patch_stride": 8,
            "catch_cf_dim": 64,
            "catch_head_dim": 64,
            "catch_head_dropout": 0.1,
            "catch_individual": 0,
            "catch_affine": 0,
            "catch_subtract_last": 0,
            
            "catch_regular_lambda": 0.5,
            "catch_temperature": 0.07,
            "catch_auxi_loss": "MAE",
            "catch_auxi_type": "complex",
            "catch_auxi_mode": "fft",
            "catch_module_first": 1,
            "catch_dc_lambda": 0.005,
            "catch_auxi_lambda": 0.005,
            
            "catch_score_lambda": 0.05,
            "catch_inference_patch_size": 32,
            "catch_inference_patch_stride": 1,
            "catch_mask": 0,
            
            "catch_lr": 0.0001,
            "catch_mask_lr": 1.0e-05,
            "catch_pct_start": 0.3,
            
            "d_model": 128,
            "d_ff": 256,
            "e_layers": 3,
            "n_heads": 2,
            "dropout": 0.2,
        },
    },
    "CrossAD": {
        "_default": {
            # CrossAD 核心架构参数
            "crossad_patch_len": 6,
            "crossad_ms_kernels": [16, 8, 4, 2],
            "crossad_ms_method": "average_pooling",
            "crossad_topk": 10,
            "crossad_n_query": 5,
            "crossad_query_len": 5,
            "crossad_bank_size": 32,
            "crossad_decay": 0.95,
            "crossad_epsilon": 1e-5,
            "crossad_m_layers": 2,
            
            # CrossAD Dropout 参数
            "crossad_attn_dropout": 0.1,
            "crossad_proj_dropout": 0.1,
            "crossad_ff_dropout": 0.1,
            
            # CrossAD 其他参数
            "crossad_norm": "layernorm",
            "crossad_lradj": "type1",
            
            # 通用架构参数
            "d_model": 128,
            "d_ff": None,
            "e_layers": 2,
            "d_layers": 2,
            "n_heads": 4,
            "activation": "gelu",
        },
        "MSL": {
            "enc_in": 55,
            "c_out": 55,
        },
        "PSM": {
            "enc_in": 25,
            "c_out": 25,
        },
        "SMAP": {
            "enc_in": 25,
            "c_out": 25,
        },
        "SMD": {
            "enc_in": 38,
            "c_out": 38,
        },
        "SWaT": {
            "enc_in": 51,
            "c_out": 51,
        },
    },
    "InterpGN": {
        "_default": {
            "dnn_type": "FCN",
            "num_shapelet": 10,
            "lambda_reg": 0.1,
            "lambda_div": 0.1,
            "epsilon": 1.0,
            "beta_schedule": "constant",
            "gating_value": 1.0,
            "distance_func": "euclidean",
            "sbm_cls": "linear",
            "pos_weight": False,
            "memory_efficient": False,
            "train_epochs": 500,
            "patience": 50,
        },
    },
    "SGN": {
        "_default": {
            "num_groups": 4,
            "period": 32,
            "depths": [6, 6, 6, 6, 6, 6],
            "block_num": 1,
            "kernel_size": 3,
            "mlp_ratio": 2,
            "num_kernels": 7,
            "sgn_embed_loss_weight": 0.1,

        },
    },
    "TimeMixer": {
        "_default": {
            # TimeMixer 核心架构参数
            "down_sampling_layers": 3,
            "down_sampling_window": 2,
            "down_sampling_method": "avg",
            "channel_independence": 0,
            "decomp_method": "moving_avg",
            "moving_avg": 25,
        },
        "PEMS-SF": {
            "d_model": 64,
            "d_ff": 64,
            "e_layers": 6, 
            "train_epochs": 50, 
            "patience": 20,
            "top_k": 3
        },
        "MSL": {
            "d_layers": 1,
            "dropout": 0.1,
            "channel_independence": 1,
            "decomp_method": "moving_avg",
            "moving_avg": 25,
            "use_norm": 1,
            "down_sampling_layers": 3,
            "down_sampling_window": 2,
            "down_sampling_method": "avg",
        },
        "PSM": {
            "d_layers": 1,
            "dropout": 0.1,
            "channel_independence": 1,
            "decomp_method": "moving_avg",
            "moving_avg": 25,
            "use_norm": 1,
            "down_sampling_layers": 3,
            "down_sampling_window": 2,
            "down_sampling_method": "avg",
        },
    },
    "MtsCID": {
        "_default": {
            "mtscid_temperature": 0.1,
            "mtscid_alpha": 1.0,
            "mtscid_aggregation": "normal_mean",
            "mtscid_peak_lr": 0.002,
            "mtscid_end_lr": 5.0e-05,
            "mtscid_weight_decay": 5.0e-05,
            "mtscid_warmup_epoch": 0,
            "mtscid_multiscale_patch_size": [10, 20],
            "mtscid_multiscale_kernel_size": [5],
            "mtscid_branch1_networks": ["fc_linear", "intra_fc_transformer", "multiscale_ts_attention"],
            "mtscid_branch2_networks": ["multiscale_conv1d", "inter_fc_transformer"],
            "mtscid_branch1_match_dimension": "first",
            "mtscid_branch2_match_dimension": "first",
            "mtscid_decoder_networks": ["linear"],
            "mtscid_decoder_group_embedding": "False",
            "mtscid_branches_group_embedding": "False_False",
            "mtscid_memory_guided": "sinusoid",
            "mtscid_embedding_init": "normal",
        },
        "MSL": {
            "d_model": 55,
        },
        "PSM": {
            "d_model": 25,
        },
        "SMAP": {
            "d_model": 25,
        },
        "SMD": {
            "d_model": 38,
        },
        "SWaT": {
            "d_model": 51,
        },
    },
    "TimeEmb": {
        "_default": {
            "d_model": 512,
            "use_revin": 1,
            "use_hour_index": 1,
            "use_day_index": 0,
            "hour_length": 24,
            "day_length": 7,
            "train_epochs": 30,
            "patience": 5,
            "batch_size": 256,
            "learning_rate": 0.005,
            "random_seed": 2024,
            "rec_lambda": 0.0,
            "auxi_lambda": 1.0,
            "auxi_loss": "MAE",
            "auxi_mode": "fft",
            "auxi_type": "complex",
            "module_first": 1,
            "pred_len": [96],
        },
        "ETTh1": {
            "_pred_len_overrides": {
                96: {"rec_lambda": 0.25, "auxi_lambda": 0.75},
                192: {"rec_lambda": 0.25, "auxi_lambda": 0.75},
                336: {"rec_lambda": 0.0, "auxi_lambda": 1.0},
                720: {"rec_lambda": 1.0, "auxi_lambda": 0.0},
            },
        },
        "ETTh2": {
            "_pred_len_overrides": {
                96: {"rec_lambda": 0.25, "auxi_lambda": 0.75},
                192: {"rec_lambda": 0.5, "auxi_lambda": 0.5},
                336: {"rec_lambda": 0.0, "auxi_lambda": 1.0},
                720: {"rec_lambda": 0.25, "auxi_lambda": 0.75},
            },
        },
        "ETTm2": {
            "_pred_len_overrides": {
                96: {"rec_lambda": 0.25, "auxi_lambda": 0.75},
                192: {"rec_lambda": 0.25, "auxi_lambda": 0.75},
                336: {"rec_lambda": 0.0, "auxi_lambda": 1.0},
            },
        }
    }
}
