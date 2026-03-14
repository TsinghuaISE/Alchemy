

import os
import argparse
import yaml
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'

import numpy as np
# 临时修复旧代码兼容性
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument(
        '--gpu-id',
        '--gpu',
        dest='gpu_id',
        type=str,
        help='CUDA device id(s), e.g. 0 or "1,2"; defaults to respecting CUDA_VISIBLE_DEVICES or overall.yaml.',
    )
    parser.add_argument(
        '--model-config',
        dest='model_config',
        type=str,
        default=None,
        help='Optional model config YAML path (overrides configs/model/{model}.yaml).',
    )

    config_dict = {}

    args, _ = parser.parse_known_args()
    if args.gpu_id is not None:
        config_dict['gpu_id'] = args.gpu_id
    if args.model_config:
        config_dict['model_config_path'] = args.model_config

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
