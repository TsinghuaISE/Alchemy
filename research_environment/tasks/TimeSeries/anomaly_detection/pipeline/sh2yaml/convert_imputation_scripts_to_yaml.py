#!/usr/bin/env python3
"""
将 time_series/scripts/imputation/**.sh 中的命令，聚合转换为
time_series/config/imputation/<Model>.yaml。

目标：
- 不区分数据集：删除随数据集变化的参数（root_path/data_path/data/enc_in/... 等），
  运行时由 experiment_registry.py 的 DATASET_REGISTRY 自动补齐。
- 支持同一 .sh 里多条命令仅 mask_rate 不同：聚合为 `mask_rate: [0.125, 0.25, ...]`。

用法：
  python convert_imputation_scripts_to_yaml.py
  python convert_imputation_scripts_to_yaml.py --dry-run
  python convert_imputation_scripts_to_yaml.py --overwrite
"""

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ASSIGNMENT_RE = re.compile(r"^(export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
VAR_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


IMPUTATION_DATASET_SUFFIXES = {"ECL", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"}

# 这些参数会随数据集变化（或者无通用意义），应当从 YAML 中剔除
EXCLUDED_ARGS = {
    "task_name",
    "model",
    "model_id",
    "mask_rate",
    "root_path",
    "data_path",
    "data",
    "enc_in",
    "dec_in",
    "c_out",
    "gpu_id",
}


def merge_lines(lines: Iterable[str]) -> List[str]:
    """把反斜杠续行的 shell 命令合并为单行。"""
    merged: List[str] = []
    buf: List[str] = []
    for raw in lines:
        stripped = raw.rstrip()
        if not stripped:
            if buf:
                merged.append(" ".join(buf).strip())
                buf = []
            continue
        if stripped.endswith("\\"):
            buf.append(stripped[:-1].rstrip())
        else:
            buf.append(stripped)
            merged.append(" ".join(buf).strip())
            buf = []
    if buf:
        merged.append(" ".join(buf).strip())
    return merged


def parse_assignment(line: str) -> Tuple[str, str, bool]:
    match = ASSIGNMENT_RE.match(line.strip())
    if not match:
        return "", "", False
    exported = bool(match.group(1))
    key = match.group(2)
    value = match.group(3).strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        value = value[1:-1]
    return key, value, exported


def substitute_vars(text: str, variables: Dict[str, str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return str(variables.get(name, match.group(0)))

    return VAR_RE.sub(_replace, text)


def convert_token(token: str) -> Any:
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def normalize_values(values: List[str]) -> Any:
    if not values:
        return True
    if len(values) == 1:
        return convert_token(values[0])
    return [convert_token(v) for v in values]


def parse_python_command(command: str, variables: Dict[str, str], source: Path) -> Optional[Dict[str, Any]]:
    substituted = substitute_vars(command, variables)
    try:
        tokens = shlex.split(substituted)
    except ValueError:
        return None
    if not tokens:
        return None

    try:
        run_idx = next(idx for idx, t in enumerate(tokens) if t.endswith("run.py"))
    except StopIteration:
        return None

    args_tokens = tokens[run_idx + 1 :]
    parsed_args: Dict[str, Any] = {}
    i = 0
    while i < len(args_tokens):
        token = args_tokens[i]
        if token.startswith("--"):
            key = token[2:]
            i += 1
            values: List[str] = []
            while i < len(args_tokens) and not args_tokens[i].startswith("--"):
                values.append(args_tokens[i])
                i += 1
            parsed_args[key] = normalize_values(values)
        else:
            i += 1

    if not parsed_args:
        return None
    return {"source": str(source.as_posix()), "command": substituted, "args": parsed_args}


def parse_script(script_path: Path) -> List[Dict[str, Any]]:
    merged_lines = merge_lines(script_path.read_text(encoding="utf-8").splitlines())
    variables: Dict[str, str] = {}
    env_vars: Dict[str, str] = {}
    commands: List[Dict[str, Any]] = []

    for line in merged_lines:
        if not line or line.lstrip().startswith("#"):
            continue

        python_idx = line.find("python")
        if python_idx != -1:
            prefix = line[:python_idx].strip()
            cmd_part = line[python_idx:].strip()

            # 捕获 `export X=...` 或 `X=... python ...` 前缀
            if prefix:
                for tok in prefix.split():
                    if tok == "export":
                        continue
                    key, value, exported = parse_assignment(tok)
                    if key:
                        variables[key] = value
                        if exported or key.isupper():
                            env_vars[key] = value

            parsed = parse_python_command(cmd_part, {**env_vars, **variables}, script_path)
            if parsed:
                parsed["env"] = dict(env_vars) if env_vars else {}
                commands.append(parsed)
            continue

        key, value, exported = parse_assignment(line)
        if key:
            variables[key] = value
            if exported or key.isupper():
                env_vars[key] = value

    return commands


def infer_model_name_from_script(script_path: Path) -> str:
    stem = script_path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1] in IMPUTATION_DATASET_SUFFIXES:
        return "_".join(parts[:-1])
    return stem


def render_yaml(data: Dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore

        class FlowList(list):
            """用于强制 YAML 以 flow style 输出的 list（例如：[0.1, 0.2]）。"""

        class CustomDumper(yaml.SafeDumper):  # type: ignore
            pass

        def _represent_flow_list(dumper, value):  # type: ignore
            return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)

        CustomDumper.add_representer(FlowList, _represent_flow_list)

        # 仅对 mask_rate 施加 flow style（不影响其他 list 的可读性）
        if "mask_rate" in data and isinstance(data["mask_rate"], list):
            data["mask_rate"] = FlowList(data["mask_rate"])

        return yaml.dump(data, Dumper=CustomDumper, allow_unicode=True, sort_keys=False)
    except Exception:
        return json.dumps(data, indent=2, ensure_ascii=False)


def dict_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    keys = set(a) | set(b)
    diff: Dict[str, Tuple[Any, Any]] = {}
    for k in sorted(keys):
        if a.get(k) != b.get(k):
            diff[k] = (a.get(k), b.get(k))
    return diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert scripts/imputation/*.sh into config/imputation/*.yaml.")
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path("scripts") / "imputation",
        help="imputation 脚本目录（默认 scripts/imputation）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("config") / "imputation",
        help="输出目录（默认 config/imputation）",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 YAML")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    args = parser.parse_args()

    scripts_dir: Path = args.scripts_dir
    output_dir: Path = args.output_dir

    if not scripts_dir.exists():
        raise SystemExit(f"Scripts directory not found: {scripts_dir}")

    model_payloads: Dict[str, Dict[str, Any]] = {}

    for script_path in sorted(scripts_dir.rglob("*.sh")):
        commands = parse_script(script_path)
        if not commands:
            continue

        model_name = infer_model_name_from_script(script_path)
        for cmd in commands:
            cmd_args = cmd.get("args") or {}
            if str(cmd_args.get("task_name")) != "imputation":
                continue

            cleaned_args = {k: v for k, v in cmd_args.items() if k not in EXCLUDED_ARGS}
            mask_rate = cmd_args.get("mask_rate", None)

            if model_name not in model_payloads:
                model_payloads[model_name] = {
                    "task_name": "imputation",
                    "env": {},  # 默认不写 CUDA_VISIBLE_DEVICES，GPU 由 replicate.py 统一调度
                    "mask_rate": [],
                    "_base_args": cleaned_args,
                    "_sources": set(),
                }

            payload = model_payloads[model_name]
            base_args = payload["_base_args"]
            diff = dict_diff(base_args, cleaned_args)
            if diff:
                # 保守策略：以第一次出现的 base_args 为准，其余差异打印警告
                print(f"[!] Warning: {model_name} args mismatch between scripts.")
                print(f"    - script: {script_path}")
                for k, (v0, v1) in list(diff.items())[:30]:
                    print(f"      * {k}: {v0} != {v1}")

            if mask_rate is not None:
                if isinstance(mask_rate, list):
                    payload["mask_rate"].extend(mask_rate)
                else:
                    payload["mask_rate"].append(mask_rate)

            payload["_sources"].add(str(script_path))

    # finalize & write
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for model_name, payload in sorted(model_payloads.items()):
        base_args = payload.pop("_base_args")
        sources = sorted(payload.pop("_sources"))
        # uniq + sort (float)
        if isinstance(payload.get("mask_rate"), list):
            payload["mask_rate"] = sorted({float(x) for x in payload["mask_rate"]})

        # 把 base_args 平铺到顶层（replicate.py 会把顶层当作 args）
        out_data: Dict[str, Any] = {}
        out_data.update(base_args)
        out_data["task_name"] = "imputation"
        if payload.get("mask_rate"):
            out_data["mask_rate"] = payload["mask_rate"]
        if args.dry_run:
            print(f"- {model_name}: mask_rate={out_data.get('mask_rate')} (sources={len(sources)})")
            continue

        yaml_path = output_dir / f"{model_name}.yaml"
        if yaml_path.exists() and not args.overwrite:
            print(f"Skip existing {yaml_path} (use --overwrite to replace)")
            continue
        yaml_path.write_text(render_yaml(out_data), encoding="utf-8")
        written += 1

    if args.dry_run:
        print(f"Dry-run done. Parsed {len(model_payloads)} models from {scripts_dir}")
    else:
        print(f"Done. Wrote {written} YAML files to {output_dir}")


if __name__ == "__main__":
    main()


