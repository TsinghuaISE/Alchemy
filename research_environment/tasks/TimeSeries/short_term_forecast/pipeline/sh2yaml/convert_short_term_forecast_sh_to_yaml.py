#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 TimeSeries/time_series/scripts/short_term_forecast 下的 .sh 脚本
转换为 TimeSeries/time_series/config/short_term_forecast 下的 <Model>.yaml。

设计目标：
- 只生成"按任务类型"的模型 YAML：不区分数据集，不写入 root_path/data_path/model_id 等数据集相关字段
- seasonal_patterns 用 [] 格式记录，运行时遍历执行每个值
- 其他参数如果都相同就保持标量，如果不同就转换为列表（与 seasonal_patterns 对齐）
- 参考 config/imputation 的格式（扁平结构，列表用 []）
- 把 $model_name 替换为真实值

用法：
  python convert_short_term_forecast_sh_to_yaml.py
  python convert_short_term_forecast_sh_to_yaml.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_SCRIPTS_DIR = os.path.join(
    "TimeSeries", "time_series", "scripts", "short_term_forecast"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    "TimeSeries", "time_series", "config", "short_term_forecast"
)

# 需要删除的字段（数据集相关）
DROP_KEYS = {"root_path", "data_path", "model_id", "model"}

# 必须用列表格式的字段（与 seasonal_patterns 对齐）
LIST_ALIGNED_KEYS = {"seasonal_patterns"}


def _strip_quotes(s: str) -> str:
    """去除字符串两端的引号"""
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1]
    return s


def _try_number(s: str) -> Any:
    """尝试将字符串转换为数字"""
    s = str(s).strip()
    if not s:
        return s
    # int
    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except Exception:
            return s
    # float
    if re.fullmatch(r"-?\d+\.\d+", s):
        try:
            return float(s)
        except Exception:
            return s
    return s


def parse_bash_vars(text: str) -> Dict[str, str]:
    """
    解析 bash 变量赋值（标量）
    """
    scalars: Dict[str, str] = {}
    
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        
        # 标量：name=value
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if m:
            name, rhs = m.group(1), m.group(2).strip()
            scalars[name] = _strip_quotes(rhs)
    
    return scalars


def extract_runpy_commands(text: str) -> List[str]:
    """
    从 shell 文本中抽取完整的 `python ... run.py ...` 命令字符串（把 \ 换行拼接）
    """
    lines = text.splitlines()
    cmds: List[str] = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        if "python" in line and "run.py" in line:
            buf = [line.rstrip()]
            i += 1
            # 继续收集后续参数行（典型每行以 "\" 结尾）
            while i < len(lines):
                nxt = lines[i].rstrip()
                if not nxt.strip():
                    break
                if nxt.strip().startswith("python") and "run.py" in nxt:
                    break
                # 常见参数行以 -- 开头；也可能是缩进的 --xx
                if nxt.lstrip().startswith("--") or nxt.strip().endswith("\\"):
                    buf.append(nxt)
                    i += 1
                    continue
                # 遇到 done/fi 等，结束
                if nxt.strip() in {"done", "fi"}:
                    break
                # 其他非参数行，停止
                if nxt.strip().startswith("#"):
                    i += 1
                    continue
                break
            cmd = "\n".join(buf)
            cmds.append(cmd)
        else:
            i += 1
    
    return cmds


def expand_command_template(cmd: str, scalars: Dict[str, str]) -> str:
    """
    对命令中的 $var、${var} 做展开
    """
    s = cmd
    # 1) 展开 ${var}
    for k, v in scalars.items():
        s = re.sub(rf"\$\{{{re.escape(k)}\}}", str(v), s)
    # 2) 展开 $var（避免误伤 $i 等特殊变量）
    for k, v in scalars.items():
        # 使用单词边界，但排除 $var_xxx 的情况
        s = re.sub(rf"\${re.escape(k)}(?![A-Za-z0-9_])", str(v), s)
    
    return s


def parse_runpy_args(cmd: str) -> Dict[str, Any]:
    """
    将一条已展开的 `python -u run.py --k v --flag` 解析为参数字典
    """
    # 去掉反斜杠换行
    one_line = " ".join([ln.rstrip().rstrip("\\").strip() for ln in cmd.splitlines()])
    try:
        tokens = shlex.split(one_line, posix=True)
    except Exception:
        tokens = one_line.split()
    
    # 找到 run.py 之后的参数 token
    try:
        run_idx = tokens.index("run.py")
    except ValueError:
        # 有时写成 python -u ./run.py
        run_idx = None
        for j, t in enumerate(tokens):
            if t.endswith("/run.py") or t.endswith("\\run.py"):
                run_idx = j
                break
        if run_idx is None:
            raise ValueError(f"无法定位 run.py: {tokens[:10]} ...")
    
    args_tokens = tokens[run_idx + 1 :]
    out: Dict[str, Any] = {}
    i = 0
    while i < len(args_tokens):
        t = args_tokens[i]
        if not t.startswith("--"):
            i += 1
            continue
        key = t[2:]
        # 布尔 flag：--use_amp
        if i + 1 >= len(args_tokens) or args_tokens[i + 1].startswith("--"):
            out[key] = True
            i += 1
            continue
        val = args_tokens[i + 1]
        out[key] = _try_number(_strip_quotes(val))
        i += 2
    
    return out


def parse_sh_file(text: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    解析 sh 文件，返回 (model_name, runs)
    """
    scalars = parse_bash_vars(text)
    
    # 提取 model_name
    model_name = None
    m = re.search(r"^\s*model_name\s*=\s*([A-Za-z0-9_\-]+)\s*$", text, flags=re.M)
    if m:
        model_name = m.group(1)
        scalars["model_name"] = model_name
    
    runs: List[Dict[str, Any]] = []
    
    # 提取所有 python 命令
    cmds = extract_runpy_commands(text)
    for cmd in cmds:
        # 展开变量
        expanded_cmd = expand_command_template(cmd, scalars)
        args = parse_runpy_args(expanded_cmd)
        
        # 只处理 short_term_forecast 任务
        if args.get("task_name") == "short_term_forecast":
            # 替换 $model_name
            if args.get("model") == "$model_name" and model_name:
                args["model"] = model_name
            runs.append(args)
    
    # 最终兜底：如果没找到 model_name，就从 runs 的 --model 推断
    if not model_name and runs:
        mn = runs[0].get("model")
        if isinstance(mn, str) and mn:
            model_name = mn
    
    return model_name, runs


def build_yaml_config_from_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    从多个 runs 构建 yaml 配置
    """
    if not runs:
        raise ValueError("空 runs")
    
    # 按 seasonal_patterns 排序（保持一致的顺序）
    seasonal_order = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
    if all("seasonal_patterns" in r for r in runs):
        def sort_key(r):
            sp = r.get("seasonal_patterns", "")
            try:
                return seasonal_order.index(sp)
            except ValueError:
                return 999
        runs_sorted = sorted(runs, key=sort_key)
    else:
        runs_sorted = list(runs)
    
    n = len(runs_sorted)
    out: Dict[str, Any] = {}
    
    # task_name
    task_name = None
    for r in runs_sorted:
        if "task_name" in r:
            task_name = r["task_name"]
            break
    out["task_name"] = task_name or "short_term_forecast"
    
    # 收集所有 key，排除 DROP_KEYS
    keys = set.intersection(*[set(r.keys()) for r in runs_sorted])
    keys = {k for k in keys if k not in DROP_KEYS and k != "task_name"}
    
    # seasonal_patterns 一定用 list
    if "seasonal_patterns" in keys:
        out["seasonal_patterns"] = [r["seasonal_patterns"] for r in runs_sorted]
        keys.remove("seasonal_patterns")
    elif "seasonal_patterns" in runs_sorted[0]:
        out["seasonal_patterns"] = [r.get("seasonal_patterns") for r in runs_sorted]
    
    seasonal_list = out.get("seasonal_patterns", [])
    if not seasonal_list:
        raise ValueError("无法构造 seasonal_patterns list（脚本缺少 seasonal_patterns）")
    
    # 其他 key：常量 -> 标量；变化 -> list（与 seasonal_patterns 顺序对齐）
    for k in sorted(keys):
        if k in LIST_ALIGNED_KEYS:
            continue
        vals = [r.get(k) for r in runs_sorted]
        if all(v == vals[0] for v in vals):
            out[k] = vals[0]
        else:
            # 如果长度不匹配，用第一个值填充
            if len(vals) != len(seasonal_list):
                out[k] = vals[0] if vals else None
            else:
                out[k] = vals
    
    return out


def _yaml_scalar(v: Any) -> str:
    """将值转换为 yaml 标量字符串"""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    # 尽量保持简洁：无空格/特殊符号则不加引号
    if re.match(r"^[A-Za-z0-9_./\-]+$", s):
        return s
    # 其他情况用双引号
    s = s.replace('"', '\\"')
    return f'"{s}"'


def _yaml_list(seq: Sequence[Any]) -> str:
    """将序列转换为 yaml 列表字符串（flow style）"""
    return "[" + ", ".join(_yaml_scalar(x) for x in seq) + "]"


def dump_simple_yaml(mapping: Dict[str, Any]) -> str:
    """
    生成类似现有 config/imputation 的"扁平 key: value + flow list"风格 YAML
    """
    lines: List[str] = []
    # task_name 放最前面，随后按 key 排序（seasonal_patterns 靠前）
    preferred = ["task_name", "is_training", "seasonal_patterns"]
    keys = list(mapping.keys())
    ordered: List[str] = []
    for k in preferred:
        if k in mapping:
            ordered.append(k)
    for k in sorted(keys):
        if k not in ordered:
            ordered.append(k)
    
    for k in ordered:
        v = mapping[k]
        if isinstance(v, (list, tuple)):
            lines.append(f"{k}: {_yaml_list(v)}")
        else:
            lines.append(f"{k}: {_yaml_scalar(v)}")
    lines.append("")  # 文件末尾空行
    return "\n".join(lines)


@dataclass
class ScriptInfo:
    path: str
    model: str


def discover_scripts(scripts_dir: str) -> List[str]:
    """发现所有 .sh 文件"""
    sh_files: List[str] = []
    for root, _, files in os.walk(scripts_dir):
        for fn in files:
            if fn.endswith(".sh"):
                sh_files.append(os.path.join(root, fn))
    return sorted(sh_files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts_dir", type=str, default=DEFAULT_SCRIPTS_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    scripts_dir = args.scripts_dir
    output_dir = args.output_dir
    
    if not os.path.isdir(scripts_dir):
        raise SystemExit(f"scripts_dir 不存在: {scripts_dir}")
    
    sh_files = discover_scripts(scripts_dir)
    if not sh_files:
        raise SystemExit(f"未找到 .sh: {scripts_dir}")
    
    # 先解析出每个脚本的 model_name，并按模型分组
    model_to_scripts: Dict[str, List[ScriptInfo]] = {}
    skipped = 0
    
    for p in sh_files:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        try:
            model_name, runs = parse_sh_file(text)
        except Exception as e:
            if args.verbose:
                print(f"[WARN] 解析失败 {p}: {e}")
            skipped += 1
            continue
        
        if not model_name or not runs:
            skipped += 1
            continue
        
        # 从文件名提取模型名（去掉 _M4 后缀）
        base_name = os.path.splitext(os.path.basename(p))[0]
        model_from_file = base_name.replace("_M4", "")
        
        # 优先使用脚本中的 model_name，否则使用文件名
        final_model_name = model_name or model_from_file
        
        model_to_scripts.setdefault(final_model_name, []).append(
            ScriptInfo(path=p, model=final_model_name)
        )
    
    if args.verbose:
        print(
            f"[*] found scripts: {len(sh_files)}, runnable: {sum(len(v) for v in model_to_scripts.values())}, skipped: {skipped}"
        )
    
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)
    
    generated = 0
    for model, infos in sorted(model_to_scripts.items(), key=lambda kv: kv[0].lower()):
        # 选择第一个脚本（通常只有一个）
        chosen = infos[0]
        with open(chosen.path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        try:
            model_name, runs = parse_sh_file(text)
        except Exception as e:
            print(f"[ERROR] 解析失败 {chosen.path}: {e}")
            continue
        
        if not runs:
            print(f"[WARN] 未找到有效 runs: {chosen.path}")
            continue
        
        try:
            cfg = build_yaml_config_from_runs(runs)
        except Exception as e:
            print(f"[ERROR] 构建配置失败 {chosen.path}: {e}")
            continue
        
        yaml_text = dump_simple_yaml(cfg)
        out_path = os.path.join(output_dir, f"{model_name}.yaml")
        
        if args.dry_run:
            print(
                f"[DRY] {model_name}: {os.path.relpath(chosen.path)} -> {os.path.relpath(out_path)} (seasonal_patterns={cfg.get('seasonal_patterns')})"
            )
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(yaml_text)
            generated += 1
            if args.verbose:
                print(f"[OK] {model_name}: {os.path.relpath(chosen.path)} -> {os.path.relpath(out_path)}")
    
    if args.dry_run:
        print(f"[*] dry-run done. models={len(model_to_scripts)}")
    else:
        print(f"[+] generated yaml files: {generated} -> {output_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

