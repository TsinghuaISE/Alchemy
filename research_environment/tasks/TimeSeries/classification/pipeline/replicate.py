#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimeSeries replicate.py - 简化版（单次执行）

配置加载优先级（从高到低）：
1. MODEL_PARAM_OVERRIDES (experiment_registry.py) - 最高优先级
2. Model Config (config/{task}/{model}.yaml)
3. Dataset Config (config/dataset/{dataset}.yaml)
4. Task Config (config/{task}/overall.yaml)
5. Global Config (config/overall.yaml) - 最低优先级

前置条件（由 executor 挂载处理）：
  - algorithm.py 已挂载到 models/{model}.py
  - hyperparameter.yaml 已挂载到 config/{task}/{model}.yaml
  - 数据集已挂载到 dataset/

输入：
  --model Autoformer    模型类名
  --dataset ETTh1       数据集名
  --task long_term_forecast  任务类型（可选）
  --gpu 0               GPU ID（可选）
  --log-dir /logs       日志输出目录（可选）

输出：
  直接透传 run.py 的输出（包含 mse, mae 等指标）
  外层 collector 解析指标
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import yaml
from copy import deepcopy

# 禁用 Python 输出缓冲
os.environ['PYTHONUNBUFFERED'] = '1'

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="TimeSeries replicate - single execution mode"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model name (e.g., Autoformer, TimesNet)"
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Dataset name (e.g., ETTh1, Weather)"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="long_term_forecast",
        help="Task type (long_term_forecast, short_term_forecast, imputation, anomaly_detection, classification)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU ID (e.g., 0)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Log output directory"
    )
    return parser.parse_args()


def load_yaml_file(yaml_path: Path) -> dict:
    """加载 YAML 文件"""
    if not yaml_path.exists():
        return {}
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}


def deep_merge(base: dict, override: dict) -> dict:
    """
    深度合并两个字典，override 中的值会覆盖 base 中的值
    
    Args:
        base: 基础配置字典
        override: 覆盖配置字典
        
    Returns:
        合并后的配置字典（新字典，不修改原字典）
    """
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = deep_merge(result[key], value)
        else:
            # 直接覆盖
            result[key] = deepcopy(value)
    
    return result


def load_hierarchical_config(model: str, dataset: str, task: str) -> dict:
    """
    加载分层配置
    
    优先级（从高到低）：
    1. MODEL_PARAM_OVERRIDES (experiment_registry.py) - 最高优先级
    2. Model Config (config/{task}/{model}.yaml)
    3. Dataset Config (config/dataset/{dataset}.yaml)
    4. DATASET_REGISTRY (experiment_registry.py) - 数据集路径配置
    5. Global Config (config/overall.yaml) - 最低优先级
    
    Args:
        model: 模型名称
        dataset: 数据集名称
        task: 任务类型
        
    Returns:
        合并后的配置字典
    """
    print("=" * 60, flush=True)
    print(f"[replicate] 开始加载分层配置", flush=True)
    print(f"  任务: {task}", flush=True)
    print(f"  模型: {model}", flush=True)
    print(f"  数据集: {dataset}", flush=True)
    print("=" * 60, flush=True)
    
    # Layer 1: 全局配置（最低优先级）
    overall_path = ROOT / "config" / "overall.yaml"
    config = load_yaml_file(overall_path)
    if config:
        print(f"[replicate] ✓ 加载全局配置: {overall_path}", flush=True)
    else:
        print(f"[replicate] ⚠ 全局配置不存在，跳过: {overall_path}", flush=True)
    
    # Layer 1.2: 任务级配置
    task_normalized = task.replace("-", "_")
    task_overall_path = ROOT / "config" / task_normalized / "overall.yaml"
    task_overall_config = load_yaml_file(task_overall_path)
    if task_overall_config:
        print(f"[replicate] ✓ 加载任务级配置: {task_overall_path}", flush=True)
        config = deep_merge(config, task_overall_config)
    else:
        print(f"[replicate] ⚠ 任务级配置不存在，跳过: {task_overall_path}", flush=True)
    
    # Layer 1.5: DATASET_REGISTRY（数据集路径配置）
    try:
        from experiment_registry import DATASET_REGISTRY
        
        if dataset in DATASET_REGISTRY:
            dataset_registry_config = DATASET_REGISTRY[dataset]
            print(f"[replicate] ✓ 应用 DATASET_REGISTRY: {dataset}", flush=True)
            print(f"[replicate]   - root_path: {dataset_registry_config.get('root_path', 'N/A')}", flush=True)
            print(f"[replicate]   - data_path: {dataset_registry_config.get('data_path', 'N/A')}", flush=True)
            config = deep_merge(config, dataset_registry_config)
        else:
            print(f"[replicate] ⚠ DATASET_REGISTRY 中无 {dataset} 的配置", flush=True)
    except ImportError:
        print("[replicate] ⚠ 无法导入 DATASET_REGISTRY", flush=True)
    
    # Layer 2: 数据集配置
    dataset_path = ROOT / "config" / "dataset" / f"{dataset}.yaml"
    dataset_config = load_yaml_file(dataset_path)
    if dataset_config:
        print(f"[replicate] ✓ 加载数据集配置: {dataset_path}", flush=True)
        config = deep_merge(config, dataset_config)
    else:
        print(f"[replicate] ⚠ 数据集配置不存在，跳过: {dataset_path}", flush=True)
    
    # Layer 3: 模型配置
    model_path = None
    for ext in ['.yaml', '.yml']:
        candidate = ROOT / "config" / task_normalized / f"{model}{ext}"
        if candidate.exists():
            model_path = candidate
            break
    
    if model_path:
        model_config = load_yaml_file(model_path)
        print(f"[replicate] ✓ 加载模型配置: {model_path}", flush=True)
        config = deep_merge(config, model_config)
    else:
        print(f"[replicate] ✗ 模型配置不存在: config/{task_normalized}/{model}.yaml", flush=True)
        raise FileNotFoundError(f"Model config not found for {model} in task {task}")
    
    # Layer 4: MODEL_PARAM_OVERRIDES（最高优先级）
    try:
        from experiment_registry import MODEL_PARAM_OVERRIDES
        
        if model in MODEL_PARAM_OVERRIDES:
            # 首先应用 _default 配置（如果存在）
            if "_default" in MODEL_PARAM_OVERRIDES[model]:
                default_overrides = deepcopy(MODEL_PARAM_OVERRIDES[model]["_default"])
                print(f"[replicate] ✓ 应用 MODEL_PARAM_OVERRIDES._default: {len(default_overrides)} 个参数", flush=True)
                config = deep_merge(config, default_overrides)
            
            # 然后应用数据集特定配置（会覆盖 _default）
            if dataset in MODEL_PARAM_OVERRIDES[model]:
                overrides = deepcopy(MODEL_PARAM_OVERRIDES[model][dataset])
                print(f"[replicate] ✓ 应用 MODEL_PARAM_OVERRIDES.{dataset}: {len(overrides)} 个参数", flush=True)
                
                # 处理特殊的 _pred_len_overrides
                pred_len_overrides = overrides.pop('_pred_len_overrides', None)
                
                # 应用常规覆盖
                config = deep_merge(config, overrides)
                
                # 如果有 pred_len_overrides，保存到配置中供后续使用
                if pred_len_overrides:
                    config['_pred_len_overrides'] = pred_len_overrides
                    print(f"[replicate]   - 包含 pred_len 特定覆盖: {list(pred_len_overrides.keys())}", flush=True)
            else:
                print(f"[replicate] ℹ MODEL_PARAM_OVERRIDES 中无 {dataset} 的数据集特定覆盖", flush=True)
        else:
            print(f"[replicate] ℹ MODEL_PARAM_OVERRIDES 中无 {model} 的覆盖", flush=True)
    except ImportError:
        print("[replicate] ⚠ 无法导入 MODEL_PARAM_OVERRIDES", flush=True)
    
    print("=" * 60, flush=True)
    print(f"[replicate] 配置加载完成", flush=True)
    print("=" * 60, flush=True)
    
    return config


def find_model_file(model: str) -> Path:
    """
    查找模型文件
    
    位置：models/{model}.py
    """
    model_path = ROOT / "models" / f"{model}.py"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path


def build_run_args_from_config(config: dict, dataset: str, task: str, model: str) -> list:
    """
    从配置字典构建命令行参数列表
    
    支持 YAML 中的列表参数展开：
    - 如果有列表参数（如 pred_len: [96, 192, 336, 720]），会生成多组参数
    - 多个列表参数会按索引对齐（zip 方式）
    
    特殊处理的多值参数（nargs='+'）：
    - depths: 需要将整个列表展开为多个命令行参数值
    
    Returns:
        list of list: 多组参数列表，每组对应一次执行
    """
    # 定义需要展开的多值参数（nargs='+'）
    # 这些参数的列表值会被展开为多个命令行参数，而不是触发多次运行
    MULTI_VALUE_PARAMS = {'depths'}
    
    # 提取特殊字段
    pred_len_overrides = config.pop('_pred_len_overrides', None)
    
    # 第一步：识别列表参数
    list_params = {}  # {key: [values]} - 用于多次运行
    multi_value_params = {}  # {key: [values]} - 用于单次运行的多值参数
    scalar_params = {}  # {key: value}
    
    for key, value in config.items():
        # 跳过元数据字段和特殊字段
        if key.startswith('_') or key in ['runs', 'model', 'task_name', 'env', 'sweep']:
            continue
        
        if isinstance(value, (list, tuple)) and len(value) > 0:
            # 判断是多值参数还是多次运行参数
            if key in MULTI_VALUE_PARAMS:
                multi_value_params[key] = list(value)
            else:
                list_params[key] = list(value)
        else:
            scalar_params[key] = value
    
    # 第二步：确定需要执行的次数（所有列表参数应该长度相同）
    if list_params:
        list_lengths = [len(v) for v in list_params.values()]
        if len(set(list_lengths)) > 1:
            print(f"[replicate] Warning: List parameters have different lengths: {dict(zip(list_params.keys(), list_lengths))}", flush=True)
            print(f"[replicate] Will use the maximum length and repeat shorter lists", flush=True)
        num_runs = max(list_lengths)
    else:
        num_runs = 1
    
    # 第三步：生成多组参数
    all_run_args = []
    
    for run_idx in range(num_runs):
        args = []
        current_pred_len = None
        
        # 添加标量参数
        for key, value in scalar_params.items():
            if value is True:
                if key in ["is_training", "use_gpu", "use_multi_gpu"]:
                    args.extend([f"--{key}", "1"])
                else:
                    args.append(f"--{key}")
            elif value is False or value is None:
                continue
            else:
                args.extend([f"--{key}", str(value)])
        
        # 添加多值参数（展开整个列表为多个命令行参数）
        for key, values in multi_value_params.items():
            args.append(f"--{key}")
            for v in values:
                args.append(str(v))
        
        # 添加列表参数（取当前索引的值）
        for key, values in list_params.items():
            # 如果索引超出范围，使用最后一个值
            idx = run_idx if run_idx < len(values) else len(values) - 1
            value = values[idx]
            
            # 记录当前的 pred_len（用于后续的 pred_len_overrides）
            if key == 'pred_len':
                current_pred_len = value
            
            if value is True:
                if key in ["is_training", "use_gpu", "use_multi_gpu"]:
                    args.extend([f"--{key}", "1"])
                else:
                    args.append(f"--{key}")
            elif value is False or value is None:
                continue
            else:
                args.extend([f"--{key}", str(value)])
        
        # 应用 pred_len_overrides（如果存在）
        if pred_len_overrides and current_pred_len is not None:
            if current_pred_len in pred_len_overrides:
                print(f"[replicate] 应用 pred_len={current_pred_len} 的特定覆盖", flush=True)
                for key, value in pred_len_overrides[current_pred_len].items():
                    # 移除已存在的该参数
                    new_args = []
                    skip_next = False
                    for i, arg in enumerate(args):
                        if skip_next:
                            skip_next = False
                            continue
                        if arg == f"--{key}":
                            skip_next = True
                            continue
                        new_args.append(arg)
                    args = new_args
                    
                    # 添加新值
                    if value is True:
                        if key in ["is_training", "use_gpu", "use_multi_gpu"]:
                            args.extend([f"--{key}", "1"])
                        else:
                            args.append(f"--{key}")
                    elif value is False or value is None:
                        continue
                    else:
                        args.extend([f"--{key}", str(value)])
        
        # 过滤掉可能重复的必需参数
        filtered_args = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg in ["--task_name", "--model", "--data", "--dataset", "--model_id"]:
                skip_next = True
                continue
            filtered_args.append(arg)
        
        # 生成 model_id
        model_id = config.get('model_id', dataset)
        
        # 添加必需参数
        final_args = [
            "--task_name", task,
            "--model", model,
            "--model_id", model_id,
            "--data", config.get('data', dataset),
        ]
        final_args.extend(filtered_args)
        
        all_run_args.append(final_args)
    
    return all_run_args


def main():
    args = parse_args()
    
    # 验证模型文件存在
    try:
        model_path = find_model_file(args.model)
    except FileNotFoundError as e:
        print(f"[replicate] Error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    print(f"[replicate] Model: {args.model}", flush=True)
    print(f"[replicate] Dataset: {args.dataset}", flush=True)
    print(f"[replicate] Task: {args.task}", flush=True)
    print(f"[replicate] Model file: {model_path}", flush=True)
    
    # 设置环境
    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    
    # 加载分层配置
    try:
        config = load_hierarchical_config(args.model, args.dataset, args.task)
    except Exception as e:
        print(f"[replicate] Error loading config: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 从配置构建参数（可能返回多组）
    try:
        all_run_args = build_run_args_from_config(config, args.dataset, args.task, args.model)
    except Exception as e:
        print(f"[replicate] Error building args: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"[replicate] Total runs: {len(all_run_args)}", flush=True)
    print(f"[replicate] GPU: {args.gpu or 'default'}", flush=True)
    print(f"[replicate] CWD: {ROOT}", flush=True)
    
    # 执行所有运行
    all_success = True
    for run_idx, run_args in enumerate(all_run_args, 1):
        print("=" * 60, flush=True)
        print(f"[replicate] Run {run_idx}/{len(all_run_args)}", flush=True)
        
        # 构建完整命令
        # 使用容器内 tsl 环境的 Python，而不是宿主机的 Python
        cmd = ["/opt/conda/envs/tsl/bin/python", "-u", "run.py"] + run_args
        
        print(f"[replicate] Running: {' '.join(cmd)}", flush=True)
        print("=" * 60, flush=True)
        
        # 直接执行，透传输出
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # 同时输出到 stdout 和日志文件
        log_file = None
        if args.log_dir:
            log_dir = Path(args.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{args.model}_{args.dataset}_{args.task}_run{run_idx}.log"
            log_file = log_path.open("w", encoding="utf-8")
            print(f"[replicate] Log file: {log_path}", flush=True)
        
        try:
            for line in proc.stdout:
                # 输出到 stdout（外层 executor 捕获）
                sys.stdout.write(line)
                sys.stdout.flush()
                
                # 同时写入日志文件
                if log_file:
                    log_file.write(line)
                    log_file.flush()
        finally:
            if log_file:
                log_file.close()
        
        # 等待进程结束
        ret = proc.wait()
        
        print("=" * 60, flush=True)
        print(f"[replicate] Run {run_idx} exit code: {ret}", flush=True)
        
        if ret != 0:
            all_success = False
            print(f"[replicate] Run {run_idx} failed!", flush=True)
    
    # 最终退出码
    final_exit_code = 0 if all_success else 1
    print("=" * 60, flush=True)
    print(f"[replicate] All runs completed. Success: {all_success}", flush=True)
    print(f"[replicate] Final exit code: {final_exit_code}", flush=True)
    
    sys.exit(final_exit_code)

if __name__ == "__main__":
    main()
