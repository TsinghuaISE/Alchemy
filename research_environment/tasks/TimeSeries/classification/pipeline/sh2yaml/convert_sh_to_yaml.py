import os
import re
import yaml

# 定义路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts', 'anomaly_detection')
CONFIG_DIR = os.path.join(BASE_DIR, 'config', 'anomaly_detection')

# 定义需要排除的参数 (数据集相关 或 运行时环境相关)
# 注意：我也把 'model_id' 加入了排除列表，因为它通常包含数据集名称 (如 MSL)，不具备通用性
EXCLUDED_PARAMS = {
    'root_path', 
    'data', 
    'enc_in', 
    'c_out'
}

def parse_value(value):
    """
    将字符串值转换为 Python 的 int, float 或保留为 string
    """
    # 尝试转换为 int
    if value.isdigit():
        return int(value)
    # 尝试转换为 float
    try:
        return float(value)
    except ValueError:
        pass
    # 处理布尔值习惯 (虽参数通常是 0/1)
    if value.lower() == 'true': return True
    if value.lower() == 'false': return False
    return value

def parse_sh_file(file_path):
    """
    解析 .sh 文件，提取参数
    """
    params = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 预处理：将反斜杠换行符替换为空格，合并为一行方便正则匹配
    content = content.replace('\\\n', ' ').replace('\\', ' ')
    
    # 2. 正则匹配：查找 --key value 格式
    # Pattern 解释: -- 后面跟非空字符(键) + 空格 + 非空字符(值)
    pattern = re.compile(r'--([\w-]+)\s+([^\s]+)')
    matches = pattern.findall(content)
    
    for key, value in matches:
        if key not in EXCLUDED_PARAMS:
            params[key] = parse_value(value)
            
    return params

def main():
    # 确保输出目录存在
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        print(f"Created directory: {CONFIG_DIR}")

    # 遍历 scripts/anomaly_detection 下的所有子文件夹
    if not os.path.exists(SCRIPTS_DIR):
        print(f"Error: Directory {SCRIPTS_DIR} not found.")
        return

    generated_models = set()

    # os.walk 会遍历所有子目录 (MSL, PSM, etc.)
    for root, dirs, files in os.walk(SCRIPTS_DIR):
        for file in files:
            if file.endswith('.sh'):
                # 获取模型名称 (假设文件名就是模型名，例如 Autoformer.sh)
                model_name = os.path.splitext(file)[0]
                
                # 如果这个模型的 YAML 还没生成过，则进行处理
                # (假设不同数据集下的同名模型脚本，其模型核心参数是一样的，只取第一个遇到的即可)
                if model_name not in generated_models:
                    file_path = os.path.join(root, file)
                    print(f"Processing {model_name} from {file_path}...")
                    
                    config_data = parse_sh_file(file_path)
                    
                    # 生成 YAML 路径
                    yaml_path = os.path.join(CONFIG_DIR, f"{model_name}.yaml")
                    
                    # 写入 YAML
                    with open(yaml_path, 'w', encoding='utf-8') as f:
                        # default_flow_style=False 保持块状格式，sort_keys=False 保持参数顺序
                        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                    
                    print(f"--> Saved to {yaml_path}")
                    generated_models.add(model_name)

    print("\nConversion complete!")

if __name__ == "__main__":
    main()