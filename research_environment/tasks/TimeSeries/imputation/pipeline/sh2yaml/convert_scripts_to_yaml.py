#!/usr/bin/env python3
"""
Convert python invocations inside scripts/*.sh into YAML configs.

The script walks scripts/, extracts every `python ... run.py ...` command,
groups them by (model, task_name), and writes YAML files to configs/.
Each output file is named <model>_<task_name>.yaml and contains the runs
plus the original command and any CUDA env defined in the shell script.

Example:
    python convert_scripts_to_yaml.py
    python convert_scripts_to_yaml.py --dry-run
    python convert_scripts_to_yaml.py --output-dir configs/generated --overwrite
"""

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ASSIGNMENT_RE = re.compile(r"^(export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
VAR_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


def merge_lines(lines: Iterable[str]) -> List[str]:
    """Merge backslash-continued lines into complete commands."""
    merged: List[str] = []
    buffer: List[str] = []
    for raw in lines:
        stripped = raw.rstrip()
        if not stripped:
            if buffer:
                merged.append(" ".join(buffer).strip())
                buffer = []
            continue
        if stripped.endswith("\\"):
            buffer.append(stripped[:-1].rstrip())
        else:
            buffer.append(stripped)
            merged.append(" ".join(buffer).strip())
            buffer = []
    if buffer:
        merged.append(" ".join(buffer).strip())
    return merged


def parse_assignment(line: str) -> Tuple[str, str, bool]:
    """
    Parse environment/variable assignment lines.
    Returns (key, value, exported). Empty key means no assignment was found.
    """
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
    """Replace $var or ${var} using the variables dict."""
    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return str(variables.get(name, match.group(0)))

    return VAR_RE.sub(_replace, text)


def convert_token(token: str) -> Any:
    """Convert tokens to Python primitives when reasonable."""
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


def parse_python_command(
    command: str, variables: Dict[str, str], env_vars: Dict[str, str], source: Path
) -> Optional[Dict[str, Any]]:
    substituted = substitute_vars(command, variables)
    try:
        tokens = shlex.split(substituted)
    except ValueError:
        return None
    if not tokens:
        return None

    try:
        run_idx = next(
            idx
            for idx, token in enumerate(tokens)
            if token.endswith("run.py") or token.endswith("run.py'")
        )
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

    command_entry: Dict[str, Any] = {
        "name": parsed_args.get("model_id") or source.stem,
        "source": str(source.as_posix()),
        "command": substituted,
        "args": parsed_args,
    }
    if env_vars:
        command_entry["env"] = dict(env_vars)
    return command_entry


def update_env_prefix(prefix: str, variables: Dict[str, str], env_vars: Dict[str, str]) -> None:
    """Capture inline environment like `CUDA_VISIBLE_DEVICES=0` before python."""
    if not prefix:
        return
    tokens = prefix.split()
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "export":
            idx += 1
            continue
        key, value, exported = parse_assignment(token)
        if key:
            variables[key] = value
            if exported or key.isupper():
                env_vars[key] = value
        idx += 1


def parse_script(script_path: Path) -> List[Dict[str, Any]]:
    merged_lines = merge_lines(script_path.read_text().splitlines())
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
            update_env_prefix(prefix, variables, env_vars)
            parsed = parse_python_command(cmd_part, {**env_vars, **variables}, env_vars, script_path)
            if parsed:
                commands.append(parsed)
            continue

        key, value, exported = parse_assignment(line)
        if key:
            variables[key] = value
            if exported or key.isupper():
                env_vars[key] = value
            continue

    return commands


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def render_yaml(data: Dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    except Exception:
        return json.dumps(data, indent=2, ensure_ascii=False)


def convert_scripts_to_yaml(scripts_dir: Path) -> Dict[str, Dict[str, Any]]:
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    script_paths = sorted(scripts_dir.rglob("*.sh"))

    for script_path in script_paths:
        commands = parse_script(script_path)
        for cmd in commands:
            model = cmd["args"].get("model")
            task = cmd["args"].get("task_name")
            if not model or not task:
                continue
            key = (str(model), str(task))
            if key not in groups:
                groups[key] = {"model": model, "task_name": task, "runs": []}
            groups[key]["runs"].append(cmd)

    for group in groups.values():
        group["runs"].sort(key=lambda item: (item["source"], item["name"]))
    return groups


def write_groups(
    groups: Dict[Tuple[str, str], Dict[str, Any]], output_dir: Path, overwrite: bool
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for (model, task), payload in groups.items():
        filename = f"{safe_name(model)}_{safe_name(task)}.yaml"
        destination = output_dir / filename
        if destination.exists() and not overwrite:
            print(f"Skip existing {destination} (use --overwrite to replace)")
            continue
        with destination.open("w", encoding="utf-8") as fh:
            fh.write(render_yaml(payload))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate YAML configs from scripts/*.sh python invocations."
    )
    parser.add_argument("--scripts-dir", default="scripts", type=Path, help="Root folder of .sh files.")
    parser.add_argument(
        "--output-dir",
        default=Path("configs") / "generated_from_scripts",
        type=Path,
        help="Where to write the generated YAML files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing YAML files in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report without writing any files.",
    )
    args = parser.parse_args()

    scripts_dir = args.scripts_dir
    output_dir = args.output_dir

    if not scripts_dir.exists():
        raise SystemExit(f"Scripts directory not found: {scripts_dir}")

    groups = convert_scripts_to_yaml(scripts_dir)

    if args.dry_run:
        print(f"Parsed {len(groups)} model/task groups from {scripts_dir}.")
        for (model, task), payload in sorted(groups.items()):
            print(f"- {model} / {task}: {len(payload['runs'])} runs")
    else:
        write_groups(groups, output_dir, overwrite=args.overwrite)
        print(f"Wrote {len(groups)} YAML files to {output_dir}")


if __name__ == "__main__":
    main()
