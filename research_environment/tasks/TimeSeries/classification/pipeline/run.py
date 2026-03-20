import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # TimeEmb optimization objective
    parser.add_argument('--rec_lambda', type=float, default=0., help='weight of reconstruction function')
    parser.add_argument('--auxi_lambda', type=float, default=1., help='weight of auxiliary function')
    parser.add_argument('--auxi_loss', type=str, default='MAE', help='loss function for auxi')
    parser.add_argument('--auxi_mode', type=str, default='fft', help='auxi loss mode, options: [fft, rfft, rfft-D, rfft-2D, legendre, chebyshev, hermite, laguerre]')
    parser.add_argument('--auxi_type', type=str, default='complex', help='auxi loss type, options: [complex, mag, phase, mag-phase, complex-phase, complex-mag-phase]')
    parser.add_argument('--module_first', type=int, default=1, help='calculate module first then mean')
    parser.add_argument('--leg_degree', type=int, default=2, help='degree for legendre/chebyshev/hermite/laguerre polynomials')

    # CATCH
    parser.add_argument('--catch_patch_size', type=int, default=16, help='CATCH patch size')
    parser.add_argument('--catch_patch_stride', type=int, default=8, help='CATCH patch stride')
    parser.add_argument('--catch_cf_dim', type=int, default=64, help='CATCH channel fusion dim')
    parser.add_argument('--catch_head_dim', type=int, default=64, help='CATCH attention head dim')
    parser.add_argument('--catch_head_dropout', type=float, default=0.1, help='CATCH head dropout')
    parser.add_argument('--catch_individual', type=int, default=0, help='CATCH individual head flag')
    parser.add_argument('--catch_affine', type=int, default=0, help='CATCH RevIN affine')
    parser.add_argument('--catch_subtract_last', type=int, default=0, help='CATCH RevIN subtract last')
    parser.add_argument('--catch_regular_lambda', type=float, default=0.5, help='CATCH contrastive regularization')
    parser.add_argument('--catch_temperature', type=float, default=0.07, help='CATCH contrastive temperature')
    parser.add_argument('--catch_auxi_loss', type=str, default='MAE', help='CATCH auxiliary loss type')
    parser.add_argument('--catch_auxi_type', type=str, default='complex', help='CATCH auxiliary loss signal type')
    parser.add_argument('--catch_auxi_mode', type=str, default='fft', help='CATCH auxiliary fft mode')
    parser.add_argument('--catch_module_first', type=int, default=1, help='CATCH module first flag')
    parser.add_argument('--catch_dc_lambda', type=float, default=0.005, help='CATCH dynamical contrastive loss weight')
    parser.add_argument('--catch_auxi_lambda', type=float, default=0.005, help='CATCH auxiliary loss weight')
    parser.add_argument('--catch_score_lambda', type=float, default=0.05, help='CATCH frequency score weight for testing')
    parser.add_argument('--catch_inference_patch_size', type=int, default=32, help='CATCH inference patch size')
    parser.add_argument('--catch_inference_patch_stride', type=int, default=1, help='CATCH inference patch stride')
    parser.add_argument('--catch_mask', type=int, default=0, help='CATCH frequency mask flag (unused by default)')
    parser.add_argument('--catch_lr', type=float, default=0.0001, help='CATCH main learning rate')
    parser.add_argument('--catch_mask_lr', type=float, default=0.00001, help='CATCH mask generator learning rate')
    parser.add_argument('--catch_pct_start', type=float, default=0.3, help='CATCH OneCycleLR pct_start')

    # MtsCID
    parser.add_argument('--mtscid_temperature', type=float, default=0.1, help='MtsCID temperature for softmax')
    parser.add_argument('--mtscid_alpha', type=float, default=1.0, help='MtsCID entropy loss weight')
    parser.add_argument('--mtscid_aggregation', type=str, default='normal_mean',
                        choices=['normal_mean', 'mean', 'max', 'harmonic_mean', 'harmonic_max'],
                        help='MtsCID anomaly score aggregation method')
    parser.add_argument('--mtscid_peak_lr', type=float, default=0.002, help='MtsCID peak learning rate')
    parser.add_argument('--mtscid_end_lr', type=float, default=0.00005, help='MtsCID end learning rate')
    parser.add_argument('--mtscid_weight_decay', type=float, default=0.00005, help='MtsCID weight decay')
    parser.add_argument('--mtscid_warmup_epoch', type=int, default=0, help='MtsCID warmup epochs')
    parser.add_argument('--mtscid_multiscale_patch_size', nargs="+", type=int, default=[10, 20],
                        help='MtsCID multiscale patch sizes')
    parser.add_argument('--mtscid_multiscale_kernel_size', nargs="+", type=int, default=[5],
                        help='MtsCID multiscale kernel sizes')
    parser.add_argument('--mtscid_branch1_networks', nargs="+", type=str,
                        default=['fc_linear', 'intra_fc_transformer', 'multiscale_ts_attention'],
                        help='MtsCID branch1 network layers')
    parser.add_argument('--mtscid_branch2_networks', nargs="+", type=str,
                        default=['multiscale_conv1d', 'inter_fc_transformer'],
                        help='MtsCID branch2 network layers')
    parser.add_argument('--mtscid_branch1_match_dimension', type=str, default='first',
                        choices=['none', 'first', 'middle', 'last'])
    parser.add_argument('--mtscid_branch2_match_dimension', type=str, default='first',
                        choices=['none', 'first', 'middle', 'last'])
    parser.add_argument('--mtscid_decoder_networks', nargs="+", type=str, default=['linear'])
    parser.add_argument('--mtscid_decoder_group_embedding', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--mtscid_branches_group_embedding', type=str, default='False_False')
    parser.add_argument('--mtscid_memory_guided', type=str, default='sinusoid')
    parser.add_argument('--mtscid_embedding_init', type=str, default='normal')

    # CrossAD
    parser.add_argument('--crossad_patch_len', type=int, default=6, help='CrossAD patch length')
    parser.add_argument('--crossad_ms_kernels', nargs="+", type=int, default=[16, 8, 4, 2],
                        help='CrossAD multi-scale kernels')
    parser.add_argument('--crossad_ms_method', type=str, default='average_pooling',
                        help='CrossAD multi-scale method')
    parser.add_argument('--crossad_topk', type=int, default=10, help='CrossAD router top-k')
    parser.add_argument('--crossad_n_query', type=int, default=5, help='CrossAD number of queries')
    parser.add_argument('--crossad_query_len', type=int, default=5, help='CrossAD query length')
    parser.add_argument('--crossad_bank_size', type=int, default=32, help='CrossAD context bank size')
    parser.add_argument('--crossad_decay', type=float, default=0.95, help='CrossAD EMA decay')
    parser.add_argument('--crossad_epsilon', type=float, default=1e-5, help='CrossAD epsilon')
    parser.add_argument('--crossad_m_layers', type=int, default=2, help='CrossAD extractor layers')
    parser.add_argument('--crossad_attn_dropout', type=float, default=0.1, help='CrossAD attention dropout')
    parser.add_argument('--crossad_proj_dropout', type=float, default=0.1, help='CrossAD projection dropout')
    parser.add_argument('--crossad_ff_dropout', type=float, default=0.1, help='CrossAD FFN dropout')
    parser.add_argument('--crossad_norm', type=str, default='layernorm', help='CrossAD norm type')
    parser.add_argument('--crossad_lradj', type=str, default='type1', help='CrossAD lr schedule type')

    # Noise / filtering (for TimeEmb auxi loss masking)
    parser.add_argument('--add_noise', action='store_true', help='add noise before training')
    parser.add_argument('--noise_amp', type=float, default=1., help='noise amplitude')
    parser.add_argument('--noise_freq_percentage', type=float, default=0.05, help='noise frequency percentage')
    parser.add_argument('--noise_seed', type=int, default=2023, help='noise seed')
    parser.add_argument('--noise_type', type=str, default='sin', help='noise type, options: [sin, normal]')
    parser.add_argument('--cutoff_freq_percentage', type=float, default=0.06, help='cutoff frequency for masking')
    parser.add_argument('--data_percentage', type=float, default=1., help='percentage of training data')

    # Random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # InterpGN specific
    parser.add_argument('--dnn_type', type=str, default='FCN',
                        choices=['FCN', 'Transformer', 'TimesNet', 'PatchTST', 'ResNet'],
                        help='DNN type for InterpGN hybrid model')
    parser.add_argument('--num_shapelet', type=int, default=10, help='Number of shapelets per scale for InterpGN')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='L1 regularization weight for InterpGN')
    parser.add_argument('--lambda_div', type=float, default=0.1, help='Diversity loss weight for InterpGN')
    parser.add_argument('--epsilon', type=float, default=1.0, help='RBF epsilon for shapelet distance')
    parser.add_argument('--beta_schedule', type=str, default='constant',
                        choices=['constant', 'cosine', 'linear'],
                        help='Beta scheduler for shapelet auxiliary loss')
    parser.add_argument('--gating_value', type=float, default=None,
                        help='Gating threshold (None for soft gating)')
    parser.add_argument('--distance_func', type=str, default='euclidean',
                        choices=['euclidean', 'cosine', 'pearson'],
                        help='Distance function for shapelet matching')
    parser.add_argument('--sbm_cls', type=str, default='linear',
                        choices=['linear', 'bilinear', 'attention'],
                        help='Classifier type for the shapelet bottleneck')
    parser.add_argument('--pos_weight', action='store_true', default=False,
                        help='Clamp classifier weights to non-negative')
    parser.add_argument('--memory_efficient', action='store_true', default=False,
                        help='Use memory efficient shapelet distance')

    # P-sLSTM
    parser.add_argument('--pslstm_patch_size', type=int, default=16, help='P-sLSTM patch size')
    parser.add_argument('--pslstm_stride', type=int, default=8, help='P-sLSTM stride for patching')
    parser.add_argument('--pslstm_embedding_dim', type=int, default=128, help='P-sLSTM embedding dimension')
    parser.add_argument('--pslstm_num_heads', type=int, default=4, help='P-sLSTM number of heads')
    parser.add_argument('--pslstm_num_blocks', type=int, default=2, help='P-sLSTM number of xLSTM blocks')
    parser.add_argument('--pslstm_conv1d_kernel_size', type=int, default=4, help='P-sLSTM conv1d kernel size')

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # TimeEmb
    parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')
    parser.add_argument('--use_hour_index', type=int, default=1, help='1: use hour_index or 0: no use')
    parser.add_argument('--use_day_index', type=int, default=0, help='1: use day_index or 0: no use')
    parser.add_argument('--hour_length', type=int, default=24, help='embedding length of hour index')
    parser.add_argument('--day_length', type=int, default=7, help='embedding length of day index')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    # PSW-I
    parser.add_argument('--pswi_lr', type=float, default=0.01, help='PSW-I learning rate')
    parser.add_argument('--pswi_n_epochs', type=int, default=500, help='PSW-I optimization epochs')
    parser.add_argument('--pswi_batch_size', type=int, default=512, help='PSW-I OT batch size')
    parser.add_argument('--pswi_n_pairs', type=int, default=1, help='PSW-I number of batch pairs per epoch')
    parser.add_argument('--pswi_noise', type=float, default=0.01, help='PSW-I initialization noise')
    parser.add_argument('--pswi_reg_sk', type=float, default=1.0, help='PSW-I Sinkhorn regularization')
    parser.add_argument('--pswi_numItermax', type=int, default=1000, help='PSW-I Sinkhorn max iterations')
    parser.add_argument('--pswi_stopThr', type=float, default=1e-9, help='PSW-I Sinkhorn stop threshold')
    parser.add_argument('--pswi_normalize', type=int, default=0, help='PSW-I normalize OT cost matrix')

    # SGN specific
    parser.add_argument('--num_groups', type=int, default=4, help='SGN number of channel groups')
    parser.add_argument('--period', type=int, default=32, help='SGN period window size')
    parser.add_argument('--depths', nargs='+', type=int, default=[2, 2, 2, 1], help='SGN depths per layer')
    parser.add_argument('--block_num', type=int, default=1, help='SGN dilated conv block number')
    parser.add_argument('--kernel_size', type=int, default=3, help='SGN kernel size')
    parser.add_argument('--mlp_ratio', type=int, default=2, help='SGN FFN expansion ratio')
    parser.add_argument('--sgn_embed_loss_weight', type=float, default=0.1, help='SGN embedding loss weight')

    args = parser.parse_args()

    # seed
    fix_seed = args.random_seed if hasattr(args, "random_seed") else 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
