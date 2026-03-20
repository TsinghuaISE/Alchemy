#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证TimeEmb配置是否与原版一致
"""
import yaml
from pathlib import Path
from experiment_registry import MODEL_PARAM_OVERRIDES, DATASET_REGISTRY

def verify_timeemb_config():
    """验证TimeEmb配置"""
    print("=" * 80)
    print("TimeEmb 配置验证")
    print("=" * 80)
    
    # 1. 检查 experiment_registry.py 中的配置
    print("\n[1] 检查 MODEL_PARAM_OVERRIDES['TimeEmb']:")
    if "TimeEmb" in MODEL_PARAM_OVERRIDES:
        timeemb_config = MODEL_PARAM_OVERRIDES["TimeEmb"]
        
        # 检查默认配置
        if "_default" in timeemb_config:
            print("\n  ✓ _default 配置:")
            for key, value in timeemb_config["_default"].items():
                print(f"    - {key}: {value}")
        
        # 检查数据集特定配置
        for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
            if dataset in timeemb_config:
                print(f"\n  ✓ {dataset} 特定配置:")
                for key, value in timeemb_config[dataset].items():
                    print(f"    - {key}: {value}")
            else:
                print(f"\n  ℹ {dataset}: 使用 _default 配置")
    else:
        print("  ✗ 未找到 TimeEmb 配置")
        return False
    
    # 2. 检查 YAML 配置文件
    print("\n[2] 检查 config/long_term_forecast/TimeEmb.yaml:")
    yaml_path = Path(__file__).parent / "config" / "long_term_forecast" / "TimeEmb.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        print("  ✓ YAML 配置文件存在")
        print("\n  关键参数:")
        for key in ["seq_len", "pred_len", "batch_size", "learning_rate", "d_model", 
                    "use_revin", "use_hour_index", "rec_lambda", "auxi_lambda"]:
            if key in yaml_config:
                print(f"    - {key}: {yaml_config[key]}")
    else:
        print("  ✗ YAML 配置文件不存在")
        return False
    
    # 3. 验证关键参数
    print("\n[3] 验证关键参数与原版一致性:")
    checks = []
    
    # 检查默认损失权重
    default_config = timeemb_config.get("_default", {})
    checks.append(("rec_lambda (default)", default_config.get("rec_lambda"), 0.0))
    checks.append(("auxi_lambda (default)", default_config.get("auxi_lambda"), 1.0))
    checks.append(("auxi_loss", default_config.get("auxi_loss"), "MAE"))
    checks.append(("auxi_mode", default_config.get("auxi_mode"), "fft"))
    checks.append(("use_revin", default_config.get("use_revin"), 1))
    checks.append(("use_hour_index", default_config.get("use_hour_index"), 1))
    
    # 检查ETTh1的pred_len特定配置
    if "ETTh1" in timeemb_config and "_pred_len_overrides" in timeemb_config["ETTh1"]:
        pred_len_overrides = timeemb_config["ETTh1"]["_pred_len_overrides"]
        expected = {
            96: {"rec_lambda": 0.25, "auxi_lambda": 0.75},
            192: {"rec_lambda": 0.25, "auxi_lambda": 0.75},
            336: {"rec_lambda": 0.0, "auxi_lambda": 1.0},
            720: {"rec_lambda": 1.0, "auxi_lambda": 0.0}
        }
        for pred_len, expected_vals in expected.items():
            if pred_len in pred_len_overrides:
                actual_vals = pred_len_overrides[pred_len]
                checks.append((f"ETTh1 pred_len={pred_len} rec_lambda", 
                             actual_vals.get("rec_lambda"), expected_vals["rec_lambda"]))
                checks.append((f"ETTh1 pred_len={pred_len} auxi_lambda", 
                             actual_vals.get("auxi_lambda"), expected_vals["auxi_lambda"]))
    
    # 打印检查结果
    all_passed = True
    for name, actual, expected in checks:
        if actual == expected:
            print(f"  ✓ {name}: {actual}")
        else:
            print(f"  ✗ {name}: {actual} (期望: {expected})")
            all_passed = False
    
    # 4. 总结
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有配置检查通过！TimeEmb 已正确配置为与原版一致。")
        print("\n建议的测试命令:")
        print("\n# ETTm1 测试 (seq_len=96, pred_len=96)")
        print("python run.py \\")
        print("  --task_name long_term_forecast \\")
        print("  --model TimeEmb \\")
        print("  --data ETTm1 \\")
        print("  --seq_len 96 \\")
        print("  --pred_len 96 \\")
        print("  --batch_size 256 \\")
        print("  --learning_rate 0.005 \\")
        print("  --train_epochs 30 \\")
        print("  --patience 5 \\")
        print("  --is_training 1")
        print("\n# ETTh1 测试 (seq_len=96, pred_len=96)")
        print("python run.py \\")
        print("  --task_name long_term_forecast \\")
        print("  --model TimeEmb \\")
        print("  --data ETTh1 \\")
        print("  --seq_len 96 \\")
        print("  --pred_len 96 \\")
        print("  --batch_size 256 \\")
        print("  --learning_rate 0.005 \\")
        print("  --train_epochs 30 \\")
        print("  --patience 5 \\")
        print("  --is_training 1")
    else:
        print("✗ 部分配置检查失败，请检查上述错误。")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    verify_timeemb_config()

