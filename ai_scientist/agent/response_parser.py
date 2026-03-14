"""LLM 响应解析 - 提取代码、超参数、假设."""

import re
from pathlib import Path

import yaml
from loguru import logger


def extract_artifact(response: str, current: str | None,
                     is_yaml: bool = False, path: Path | None = None) -> str:
    """从 LLM 响应中提取代码或超参数.

    is_yaml=True 时验证格式, 无效则回退到 current, 避免浪费评测资源.
    """
    if "<tool_call>" in response and "<name>str_replace</name>" in response:
        result = _apply_str_replace(response, current) if current else _extract_code(response, is_yaml)
    elif "<<<old>>>" in response:
        result = _apply_old_new_replace(response, current) if current else _extract_code(response, is_yaml)
    else:
        result = _extract_code(response, is_yaml)

    result = result or current or ""

    if is_yaml:
        try:
            yaml.safe_load(result)
        except yaml.YAMLError:
            logger.warning("YAML 格式无效, 回退")
            result = current or ""

    if result and path:
        path.write_text(result, encoding="utf-8")

    return result


def extract_hypothesis(response: str) -> tuple[str, str, str | None]:
    hyp = re.search(r'##\s*HYPOTHESIS\s*\n(.*?)(?=##|\Z)', response, re.DOTALL)
    rat = re.search(r'##\s*RATIONALE\s*\n(.*?)(?=##|```|\Z)', response, re.DOTALL)
    key = re.search(r'##\s*KEY_CHANGES\s*\n(.*?)(?=##|```|\Z)', response, re.DOTALL)
    return (
        hyp.group(1).strip() if hyp else "(未提供假设)",
        rat.group(1).strip() if rat else "",
        key.group(1).strip() if key else None,
    )


def _apply_str_replace(response: str, content: str) -> str:
    pattern = r'<tool_call>\s*<name>str_replace</name>\s*<old_str>\s*(.*?)\s*</old_str>\s*<new_str>\s*(.*?)\s*</new_str>\s*</tool_call>'
    for old, new in re.findall(pattern, response, re.DOTALL):
        old, new = old.strip('\n'), new.strip('\n')
        if old in content:
            content = content.replace(old, new, 1)
    return content


def _apply_old_new_replace(response: str, content: str) -> str:
    pattern = r'<<<old>>>\s*\n(.*?)<<<new>>>\s*\n(.*?)(?=<<<|$)'
    for old, new in re.findall(pattern, response, re.DOTALL):
        old, new = old.strip(), new.strip()
        if old in content:
            content = content.replace(old, new, 1)
    return content


def _extract_code(response: str, is_yaml: bool = False) -> str:
    lang = "yaml" if is_yaml else "python"
    match = re.search(rf'```{lang}\s*\n(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()
