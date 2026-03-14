"""调用 research_environment 执行评测."""

import ast
from pathlib import Path

from loguru import logger

from research_environment.orchestrator import RunOrchestrator

# task 目录根路径
_TASKS_ROOT = Path(__file__).resolve().parents[2] / "research_environment" / "tasks"


def evaluate(domain: str, task: str, paper_id: str, algo_code: str,
             algo_path: Path, hp_path: Path, timeout: int = 86400) -> dict:
    """执行评测, 返回最佳指标字典."""
    model_name = _extract_model_class(algo_code) or paper_id.split("_")[0]
    logger.info(f"评测开始: {paper_id} (model={model_name})")

    task_dir = _TASKS_ROOT / domain / task
    if not task_dir.exists():
        raise FileNotFoundError(f"任务目录不存在: {task_dir}")

    orchestrator = RunOrchestrator(task_dir)
    summary = orchestrator.run(
        model=model_name,
        algorithm_path=Path(algo_path),
        hp_path=Path(hp_path),
        run_id=paper_id,
        fail_fast=True,
        streaming=False,
    )

    if summary.failed == summary.completed and summary.completed > 0:
        # 全部失败, 取最后一个错误
        last_err = summary.results[-1] if summary.results else None
        msg = last_err.get_full_error() if last_err else "Unknown error"
        raise RuntimeError(msg)

    # 提取最佳指标
    metrics: dict = {}
    for ds_result in summary.best_per_dataset.values():
        m = ds_result.get("metrics", {})
        if not metrics or _better(m, metrics, orchestrator.task_config):
            metrics = m

    if not metrics:
        raise ValueError(f"评测未产出有效指标: {paper_id}")

    logger.info(f"评测完成: {paper_id}, metrics={metrics}")
    return metrics


def _better(new: dict, old: dict, task_config: dict) -> bool:
    """判断 new 是否优于 old."""
    eval_cfg = task_config.get("eval", {})
    pm = eval_cfg.get("primary_metric", "")
    hib = eval_cfg.get("higher_is_better", True)
    nv, ov = new.get(pm, 0), old.get(pm, 0)
    return nv > ov if hib else nv < ov


def compare_metrics(metric: str, seed: dict, current: dict,
                    higher_is_better: bool, threshold: float = 0.01) -> str:
    if metric not in current:
        return "error"
    seed_val = seed.get(metric, 0)
    cur_val = current[metric]

    delta = (cur_val - seed_val) if higher_is_better else (seed_val - cur_val)

    if seed_val == 0:
        if delta > 0:
            return "improvement"
        if delta < 0:
            return "regression"
        return "neutral"

    rel = delta / abs(seed_val)
    if rel > threshold:
        return "improvement"
    if rel < -threshold:
        return "regression"
    return "neutral"


def get_best_hypothesis(improvements: list, metric: str, higher_is_better: bool):
    key = lambda h: h.metrics_after.get(metric, 0)
    return max(improvements, key=key) if higher_is_better else min(improvements, key=key)


def _extract_model_class(code: str | None) -> str | None:
    if not code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.bases:
            return node.name
    return None
