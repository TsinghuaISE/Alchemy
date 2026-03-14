#!/usr/bin/env python3
"""
MMRec replicate.py - 简化版（单次执行）

前置条件（由 executor 挂载处理）：
  - algorithm.py 已挂载到 src/models/{model.lower()}.py
  - hyperparameter.yaml 已挂载到 src/configs/model/{model}.yaml
  - 数据集已挂载到 data/{dataset}/

输入：
  --model BM3       模型类名
  --dataset baby    数据集名
  --gpu 0           GPU ID（可选）
  --log-dir /logs   日志输出目录（挂载到外层 logs/）

输出：
  直接透传 MMRec main.py 的输出（包含 best metric）
  外层 collector 解析 █████████████ BEST 区域
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# 禁用 Python 输出缓冲
os.environ['PYTHONUNBUFFERED'] = '1'

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMRec replicate - single execution mode"
    )
    parser.add_argument(
        "--model", "-m", 
        required=True, 
        help="Model class name (e.g., BM3)"
    )
    parser.add_argument(
        "--dataset", "-d", 
        required=True, 
        help="Dataset name (e.g., baby)"
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
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Optional model config YAML path (overrides default)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置环境
    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    
    # 构建命令
    cmd = [
        sys.executable, "-u", "main.py",
        "-m", args.model,
        "-d", args.dataset,
    ]
    
    # 如果指定了 model-config，添加参数
    if args.model_config:
        cmd.extend(["--model-config", args.model_config])
    
    print(f"[replicate] Running: {' '.join(cmd)}", flush=True)
    print(f"[replicate] GPU: {args.gpu or 'default'}", flush=True)
    print(f"[replicate] CWD: {SRC_DIR}", flush=True)
    print("=" * 60, flush=True)
    
    # 直接执行，透传输出
    proc = subprocess.Popen(
        cmd,
        cwd=SRC_DIR,
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
        log_path = log_dir / f"{args.model}_{args.dataset}.log"
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
    print(f"[replicate] Exit code: {ret}", flush=True)
    
    sys.exit(ret)


if __name__ == "__main__":
    main()
