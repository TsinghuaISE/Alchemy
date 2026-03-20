#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 TimeSeries/time_series/scripts/long_term_forecast 下的 .sh 脚本（跳过 AugmentSample/）
转换为 TimeSeries/time_series/config/long_term_forecast 下的 <Model>.yaml。

设计目标：
- 只生成"按任务类型"的模型 YAML：不区分数据集，不写入 root_path/data_path/model_id 等数据集相关字段
- pred_len 用 [] 格式记录，运行时遍历执行每个值
- data, features, enc_in, dec_in, c_out 也用 [] 格式记录，顺序和 pred_len 顺序对应
- 其他参数如果都相同就保持标量，如果不同就转换为列表（与 pred_len 对齐）
- 参考 config/imputation 的格式（扁平结构，列表用 []）

用法：
  python convert_long_term_forecast_sh_to_yaml.py
  python convert_long_term_forecast_sh_to_yaml.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_SCRIPTS_DIR = os.path.join(
    "TimeSeries", "time_series", "scripts", "long_term_forecast"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    "TimeSeries", "time_series", "config", "long_term_forecast"
)

# 需要删除的字段（数据集相关）
DROP_KEYS = {"root_path", "data_path", "model_id", "model"}

# 必须用列表格式的字段（与 pred_len 对齐）
LIST_ALIGNED_KEYS = {"data", "features", "enc_in", "dec_in", "c_out", "pred_len"}

# 数据集目录优先级（用于选择同一模型的多个脚本时）
DATASET_DIR_RANK = {
    "ECL_script": 0,
    "Traffic_script": 1,
    "Weather_script": 2,
    "Exchange_script": 3,
    "ETT_script": 4,
    "ILI_script": 5,
}


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


def parse_bash_vars(text: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    解析 bash 变量赋值：
    - 标量：name=value
    - 数组：name=(a b c)
    """
    scalars: Dict[str, str] = {}
    arrays: Dict[str, List[str]] = {}
    
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        
        # 数组：name=(a b c)
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\((.*)\)\s*$", line)
        if m:
            name = m.group(1)
            inner = m.group(2).strip()
            try:
                items = shlex.split(inner, posix=True)
                arrays[name] = [_strip_quotes(x) for x in items if x.strip()]
            except Exception:
                arrays[name] = [x.strip() for x in inner.split() if x.strip()]
            continue
        
        # 标量：name=value
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if m:
            name, rhs = m.group(1), m.group(2).strip()
            scalars[name] = _strip_quotes(rhs)
    
    return scalars, arrays


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


def expand_command_template(
    cmd: str, scalars: Dict[str, str], arrays: Dict[str, List[str]]
) -> List[str]:
    """
    对命令中的 $var、${var}、${arr[$i]} 做有限展开
    如果命令中没有数组引用，直接返回展开后的单个命令
    """
    s = cmd
    # 1) 展开 ${var}
    for k, v in scalars.items():
        s = re.sub(rf"\$\{{{re.escape(k)}\}}", str(v), s)
    # 2) 展开 $var（避免误伤 $i 等特殊变量）
    # 先处理 ${var} 形式，再处理 $var 形式
    for k, v in scalars.items():
        # 使用单词边界，但排除 $var_xxx 的情况
        s = re.sub(rf"\${re.escape(k)}(?![A-Za-z0-9_])", str(v), s)
    
    # 如果命令中有数组引用，需要展开
    arr_refs = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\[\$?(\w+)\]\}", s)
    if arr_refs:
        # 找到所有被引用的数组
        arr_names = set()
        for arr_name, idx_var in arr_refs:
            if arr_name in arrays:
                arr_names.add(arr_name)
        
        if arr_names:
            # 所有数组应该有相同长度
            lens = [len(arrays[a]) for a in arr_names]
            if lens and len(set(lens)) == 1:
                n = lens[0]
                expanded = []
                for idx in range(n):
                    s_copy = s
                    # 展开所有数组引用
                    for arr_name, idx_var in arr_refs:
                        if arr_name in arrays and 0 <= idx < len(arrays[arr_name]):
                            pattern = rf"\$\{{{re.escape(arr_name)}\[\$?{re.escape(idx_var)}\]\}}"
                            s_copy = re.sub(pattern, str(arrays[arr_name][idx]), s_copy)
                    expanded.append(s_copy)
                return expanded
    
    return [s]


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


def parse_sh_file_with_loops(text: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    解析 sh 文件，支持 for 循环和数组索引
    返回 (model_name, runs)
    """
    lines = text.splitlines()
    scalars, arrays = parse_bash_vars(text)
    
    # 提取 model_name
    model_name = None
    m = re.search(r"^\s*model_name\s*=\s*([A-Za-z0-9_\-]+)\s*$", text, flags=re.M)
    if m:
        model_name = m.group(1)
        scalars["model_name"] = model_name
    
    runs: List[Dict[str, Any]] = []
    
    # 处理 for 循环
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 识别 for i in "${!pred_lens[@]}"; do
        m_idx = re.match(r'^for\s+(\w+)\s+in\s+"?\$\{!([A-Za-z0-9_]+)\[@\]\}"?;?\s*do\s*$', line)
        if m_idx:
            loop_var = m_idx.group(1)
            arr_name = m_idx.group(2)
            if arr_name not in arrays:
                i += 1
                continue
            
            # 捕获 block 到 done
            i += 1
            block: List[str] = []
            depth = 1
            while i < len(lines):
                s2 = lines[i].strip()
                if s2.startswith("for "):
                    depth += 1
                if s2 == "done":
                    depth -= 1
                    if depth == 0:
                        break
                block.append(lines[i])
                i += 1
            
            # 展开循环
            for idx in range(len(arrays[arr_name])):
                local_vars = dict(scalars)
                local_vars[loop_var] = str(idx)
                local_vars["i"] = str(idx)
                local_vars[f"${loop_var}"] = str(idx)
                
                # 提取 block 中的 python 命令
                block_text = "\n".join(block)
                cmds = extract_runpy_commands(block_text)
                for cmd in cmds:
                    # 先手动替换数组索引
                    expanded_cmd = cmd
                    for arr_key, arr_val in arrays.items():
                        if idx < len(arr_val):
                            # 替换 ${arr[$i]} 和 ${arr[i]}
                            expanded_cmd = re.sub(
                                rf"\$\{{{re.escape(arr_key)}\[\$?{re.escape(loop_var)}\]\}}",
                                str(arr_val[idx]),
                                expanded_cmd
                            )
                            expanded_cmd = re.sub(
                                rf"\$\{{{re.escape(arr_key)}\[\$?i\]\}}",
                                str(arr_val[idx]),
                                expanded_cmd
                            )
                    
                    expanded_cmds = expand_command_template(expanded_cmd, local_vars, arrays)
                    for exp_cmd in expanded_cmds:
                        args = parse_runpy_args(exp_cmd)
                        if args.get("task_name") == "long_term_forecast":
                            if args.get("model") == "$model_name" and model_name:
                                args["model"] = model_name
                            runs.append(args)
            
            i += 1  # skip done
            continue
        
        # 识别 for pred_len in ...
        m = re.match(r"^for\s+(\w+)\s+in\s+(.+)$", line)
        if m:
            loop_var = m.group(1)
            rest = m.group(2)
            # 去掉注释
            rest = rest.split("#", 1)[0].strip()
            vals = [v for v in rest.split() if v and v != "do"]
            # 寻找 do
            while i < len(lines) and "do" not in lines[i]:
                i += 1
            i += 1  # skip line containing do
            block: List[str] = []
            depth = 1
            while i < len(lines):
                s2 = lines[i].strip()
                if s2.startswith("for "):
                    depth += 1
                if s2 == "done":
                    depth -= 1
                    if depth == 0:
                        break
                block.append(lines[i])
                i += 1
            
            # 展开循环
            for v in vals:
                local_vars = dict(scalars)
                local_vars[loop_var] = str(v)
                
                block_text = "\n".join(block)
                cmds = extract_runpy_commands(block_text)
                for cmd in cmds:
                    # 先手动替换循环变量（处理 $var 和 ${var} 形式）
                    expanded_cmd = cmd
                    # 替换 ${var} 形式
                    expanded_cmd = re.sub(rf"\$\{{{re.escape(loop_var)}\}}", str(v), expanded_cmd)
                    # 替换 $var 形式（使用单词边界，避免误替换）
                    expanded_cmd = re.sub(rf"\${re.escape(loop_var)}(?![A-Za-z0-9_])", str(v), expanded_cmd)
                    
                    expanded_cmds = expand_command_template(expanded_cmd, local_vars, arrays)
                    for exp_cmd in expanded_cmds:
                        args = parse_runpy_args(exp_cmd)
                        if args.get("task_name") == "long_term_forecast":
                            if args.get("model") == "$model_name" and model_name:
                                args["model"] = model_name
                            runs.append(args)
            
            i += 1  # skip done
            continue
        
        i += 1
    
    # 处理非循环的 python 命令（只处理不在任何循环块中的命令）
    # 如果已经处理了循环，就不再处理非循环命令，避免重复
    # 这里我们通过检查是否在循环块中来避免重复
    # 简单方法：如果 runs 为空，说明没有循环，才处理非循环命令
    if not runs:
        cmds = extract_runpy_commands(text)
        for cmd in cmds:
            expanded_cmds = expand_command_template(cmd, scalars, arrays)
            for exp_cmd in expanded_cmds:
                args = parse_runpy_args(exp_cmd)
                if args.get("task_name") == "long_term_forecast":
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
    
    # 按 pred_len 排序
    if all("pred_len" in r for r in runs):
        runs_sorted = sorted(runs, key=lambda x: int(x["pred_len"]))
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
    out["task_name"] = task_name or "long_term_forecast"

    # 收集所有 key，排除 DROP_KEYS
    keys = set.intersection(*[set(r.keys()) for r in runs_sorted])
    keys = {k for k in keys if k not in DROP_KEYS and k != "task_name"}

    # pred_len 一定用 list
    if "pred_len" in keys:
        pred_vals = []
        for r in runs_sorted:
            val = r["pred_len"]
            # 如果仍然是字符串且包含 $，说明变量没有被正确替换
            if isinstance(val, str) and "$" in val:
                raise ValueError(f"pred_len 变量未正确展开: {val}")
            pred_vals.append(int(val))
        out["pred_len"] = pred_vals
        keys.remove("pred_len")
    elif "pred_len" in runs_sorted[0]:
        pred_vals = []
        for r in runs_sorted:
            val = r.get("pred_len", 0)
            if isinstance(val, str) and "$" in val:
                raise ValueError(f"pred_len 变量未正确展开: {val}")
            pred_vals.append(int(val))
        out["pred_len"] = pred_vals
    
    pred_list = out.get("pred_len", [])
    if not pred_list:
        raise ValueError("无法构造 pred_len list（脚本缺少 pred_len）")
    
    # 强制写入 5 个占位字段（与 pred_len 等长）
    for k in ["data", "features", "enc_in", "dec_in", "c_out"]:
        if any(k in r for r in runs_sorted):
            out[k] = [r.get(k, None) for r in runs_sorted]
        else:
            # 如果脚本中没有，用 None 占位
            out[k] = [None for _ in range(len(pred_list))]

    # 其他 key：常量 -> 标量；变化 -> list（与 pred_len 顺序对齐）
    for k in sorted(keys):
        if k in LIST_ALIGNED_KEYS:
            continue
        vals = [r.get(k) for r in runs_sorted]
        if all(v == vals[0] for v in vals):
            out[k] = vals[0]
        else:
            # 如果长度不匹配，用第一个值填充
            if len(vals) != len(pred_list):
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
    # task_name 放最前面，随后按 key 排序（pred_len 靠前）
    preferred = ["task_name", "is_training", "seq_len", "label_len", "pred_len"]
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
    rank: int


def discover_scripts(scripts_dir: str) -> List[str]:
    """发现所有 .sh 文件，跳过 AugmentSample"""
    sh_files: List[str] = []
    for root, _, files in os.walk(scripts_dir):
        if "AugmentSample" in root:
            continue
        for fn in files:
            if fn.endswith(".sh"):
                sh_files.append(os.path.join(root, fn))
    return sorted(sh_files)


def choose_preferred_script(per_model: List[ScriptInfo]) -> ScriptInfo:
    """选择同一模型的多个脚本中最合适的一个"""
    # rank 小优先；其次文件名短优先；最后路径字典序
    return sorted(per_model, key=lambda x: (x.rank, len(os.path.basename(x.path)), x.path))[0]


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
            model_name, runs = parse_sh_file_with_loops(text)
        except Exception as e:
            if args.verbose:
                print(f"[WARN] 解析失败 {p}: {e}")
            skipped += 1
            continue

        if not model_name or not runs:
            skipped += 1
            continue

        # rank by dataset directory name
        rank = 99
        for part in p.split(os.sep):
            if part in DATASET_DIR_RANK:
                rank = DATASET_DIR_RANK[part]
                break
        
        model_to_scripts.setdefault(model_name, []).append(
            ScriptInfo(path=p, model=model_name, rank=rank)
        )

    if args.verbose:
        print(
            f"[*] found scripts: {len(sh_files)}, runnable: {sum(len(v) for v in model_to_scripts.values())}, skipped: {skipped}"
        )

    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

    generated = 0
    for model, infos in sorted(model_to_scripts.items(), key=lambda kv: kv[0].lower()):
        chosen = choose_preferred_script(infos)
        with open(chosen.path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        try:
            model_name, runs = parse_sh_file_with_loops(text)
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
                f"[DRY] {model_name}: {os.path.relpath(chosen.path)} -> {os.path.relpath(out_path)} (pred_len={cfg.get('pred_len')})"
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
