import os
import re
import yaml
import sys

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts', 'classification')
CONFIG_DIR = os.path.join(BASE_DIR, 'config', 'classification')

# 定义需要排除的参数 (这些参数允许在不同任务中不一致)
EXCLUDED_PARAMS = {
    'root_path', 
    'data',
    'model_id',
    'c_out',
    'seq_len',
    'pred_len',
    'num_kernels'
}
# ===========================================

def parse_value(value):
    """转换字符串为 python 类型"""
    value = value.strip("'").strip('"')
    if value.isdigit(): return int(value)
    try: return float(value)
    except ValueError: pass
    if value.lower() == 'true': return True
    if value.lower() == 'false': return False
    return value

def extract_shell_variables(content):
    """提取 Shell 变量定义"""
    variables = {}
    pattern = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)', re.MULTILINE)
    matches = pattern.findall(content)
    for key, val in matches:
        variables[key] = val.strip()
    return variables

def parse_command_block(command_str, shell_vars):
    """解析单个命令块的参数"""
    params = {}
    pattern = re.compile(r'--([\w-]+)\s+([^\s]+)')
    matches = pattern.findall(command_str)
    
    for key, value in matches:
        if key not in EXCLUDED_PARAMS:
            # 变量替换
            if value.startswith('$'):
                var_name = value[1:]
                real_value = shell_vars.get(var_name, value)
                params[key] = parse_value(real_value)
            else:
                params[key] = parse_value(value)
    return params

def check_consistency_and_extract(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    shell_vars = extract_shell_variables(content)
    # 预处理：合并换行
    content_merged = content.replace('\\\n', ' ').replace('\\', ' ')
    
    # 按照 python -u run.py 切分
    # 使用 split 而不是 find，这样可以拿到所有的块
    parts = content_merged.split('python -u run.py')
    
    # parts[0] 是文件头部的 export 等内容，跳过
    if len(parts) < 2:
        return None, "No python commands found"

    # 提取所有块的参数
    all_configs = []
    for i, part in enumerate(parts[1:]): # 从第一个 python 命令开始
        # 有时候 split 会导致后面跟着一些 && 或者 ;，简单的正则提取参数通常能忽略掉这些噪音
        config = parse_command_block(part, shell_vars)
        if config: # 确保提取到了参数
            all_configs.append(config)

    if not all_configs:
        return None, "No valid parameters parsed"

    # === 核心逻辑：一致性检查 ===
    baseline_config = all_configs[0]
    
    for idx, current_config in enumerate(all_configs[1:]):
        # 比较 current_config 和 baseline_config
        # 1. 检查键是否一致
        if set(baseline_config.keys()) != set(current_config.keys()):
            diff = set(baseline_config.keys()) ^ set(current_config.keys())
            return None, f"Parameter keys mismatch in block {idx+2}: {diff}"
        
        # 2. 检查值是否一致
        for key, val in baseline_config.items():
            if current_config[key] != val:
                return None, (
                    f"CONFLICT DETECTED in block {idx+2} (Dataset index):\n"
                    f"    Parameter: --{key}\n"
                    f"    Baseline value: {val}\n"
                    f"    Current value:  {current_config[key]}\n"
                    f"    Action: Skipped generation to avoid overwriting special settings."
                )

    # 如果全部通过，返回基准配置
    return baseline_config, None

def main():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    sh_files = [f for f in os.listdir(SCRIPTS_DIR) if f.endswith('.sh')]
    print(f"Scanning {len(sh_files)} scripts with Consistency Check...\n")

    success_count = 0
    fail_count = 0

    for file_name in sh_files:
        file_path = os.path.join(SCRIPTS_DIR, file_name)
        model_name = os.path.splitext(file_name)[0]
        
        try:
            config, error_msg = check_consistency_and_extract(file_path)
            
            if config:
                yaml_path = os.path.join(CONFIG_DIR, f"{model_name}.yaml")
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"✔ [OK] {file_name}")
                success_count += 1
            else:
                # 打印具体的错误信息（通常是冲突）
                print(f"✘ [SKIP] {file_name}")
                print(f"   Reason: {error_msg}")
                print("-" * 40)
                fail_count += 1
                
        except Exception as e:
            print(f"✘ [ERROR] {file_name}: {e}")
            fail_count += 1

    print(f"\nSummary: {success_count} converted, {fail_count} skipped/failed.")

if __name__ == "__main__":
    main()