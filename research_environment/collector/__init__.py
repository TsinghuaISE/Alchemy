"""Collector 模块 - 结果追踪、聚合与持久化.

不含领域解析逻辑, 解析由 TaskPlugin.parse_output() 负责.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from research_environment.collector.result_table import (
    ResultTable, DatasetResult, AlgorithmProgress,
)


# ══════════════════════════════════════════════════════════════════════
# 结果数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TaskResult:
    """单个任务的执行结果."""
    algorithm: str = ""
    dataset: str = ""
    hp_name: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    success: bool = False
    duration: float = 0.0
    error: Optional[str] = None
    raw_stdout: Optional[str] = None
    raw_stderr: Optional[str] = None
    node: str = ""
    gpu_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.success:
            d.pop("raw_stdout", None)
            d.pop("raw_stderr", None)
        return d

    def get_full_error(self) -> str:
        parts = []
        if self.error:
            parts.append(f"Error: {self.error}")
        if self.raw_stderr:
            parts.append(f"Stderr:\n{self.raw_stderr}")
        if self.raw_stdout:
            parts.append(f"Stdout:\n{self.raw_stdout}")
        return "\n".join(parts) if parts else "Unknown error"


@dataclass
class RunSummary:
    """单算法运行摘要."""
    domain: str
    task: str
    model: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    total_tasks: int = 0
    completed: int = 0
    success: int = 0
    failed: int = 0
    results: List[TaskResult] = field(default_factory=list)
    best_per_dataset: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_result(self, result: TaskResult):
        self.results.append(result)
        self.completed += 1
        if result.success:
            self.success += 1
        else:
            self.failed += 1

    def finalize(self, best_results: Dict[str, DatasetResult]):
        self.end_time = datetime.now().isoformat()
        for ds, r in best_results.items():
            self.best_per_dataset[ds] = {"metrics": r.metrics, "hp": r.best_hp}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d


@dataclass
class BatchRunSummary:
    """多算法批量运行摘要."""
    domain: str
    task: str
    algorithms: List[str]
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    total_tasks: int = 0
    completed: int = 0
    results_by_algorithm: Dict[str, List[TaskResult]] = field(default_factory=dict)
    best_by_algorithm: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    fatal_by_algorithm: Dict[str, Optional[TaskResult]] = field(default_factory=dict)

    def add_result(self, result: TaskResult):
        algo = result.algorithm
        if algo not in self.results_by_algorithm:
            self.results_by_algorithm[algo] = []
        self.results_by_algorithm[algo].append(result)
        self.completed += 1

    def set_fatal(self, algorithm: str, result: TaskResult):
        self.fatal_by_algorithm[algorithm] = result

    def finalize(self, best_results: Dict[str, Dict[str, DatasetResult]]):
        self.end_time = datetime.now().isoformat()
        for algo, ds_results in best_results.items():
            self.best_by_algorithm[algo] = {
                ds: {"metrics": r.metrics, "hp": r.best_hp}
                for ds, r in ds_results.items()
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain, "task": self.task,
            "algorithms": self.algorithms,
            "start_time": self.start_time, "end_time": self.end_time,
            "total_tasks": self.total_tasks, "completed": self.completed,
            "results_by_algorithm": {
                a: [r.to_dict() for r in rs]
                for a, rs in self.results_by_algorithm.items()
            },
            "best_by_algorithm": self.best_by_algorithm,
            "fatal_by_algorithm": {
                a: r.to_dict() if r else None
                for a, r in self.fatal_by_algorithm.items()
            },
        }


# ══════════════════════════════════════════════════════════════════════
# 保存与报告
# ══════════════════════════════════════════════════════════════════════

def save_results(summary: RunSummary | BatchRunSummary,
                 output_dir: Path, filename: str | None = None) -> Path:
    """保存运行结果到 JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(summary, BatchRunSummary):
            names = "_".join(summary.algorithms[:3])
            if len(summary.algorithms) > 3:
                names += f"_+{len(summary.algorithms) - 3}"
            filename = f"batch_{names}_{ts}.json"
        else:
            filename = f"{summary.model}_{ts}.json"
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def generate_report(summary: RunSummary) -> str:
    """生成单算法文本报告."""
    lines = [
        "", "=" * 70,
        f"  {summary.domain}/{summary.task} - {summary.model}",
        "=" * 70,
        f"  Total: {summary.completed}/{summary.total_tasks}  |  "
        f"Success: {summary.success}  |  Failed: {summary.failed}",
        "-" * 70,
    ]
    if summary.best_per_dataset:
        lines.append("  Best Results per Dataset:")
        for ds, info in summary.best_per_dataset.items():
            ms = ", ".join(f"{k}={v:.4f}" for k, v in info["metrics"].items())
            lines.append(f"    {ds}: {ms}  [HP: {info['hp']}]")
    lines.extend(["=" * 70, ""])
    return "\n".join(lines)


def generate_batch_report(summary: BatchRunSummary) -> str:
    """生成多算法文本报告."""
    lines = [
        "", "=" * 80,
        f"  {summary.domain}/{summary.task} - Batch Run Summary",
        "=" * 80,
        f"  Algorithms: {len(summary.algorithms)}  |  "
        f"Total Tasks: {summary.completed}/{summary.total_tasks}",
        "-" * 80,
    ]
    for algo in summary.algorithms:
        results = summary.results_by_algorithm.get(algo, [])
        s = sum(1 for r in results if r.success)
        fatal = summary.fatal_by_algorithm.get(algo)
        if fatal:
            icon, color = "✗ STOPPED", "\033[31m"
        elif len(results) - s > 0:
            icon, color = "⚠ PARTIAL", "\033[33m"
        else:
            icon, color = "✓ DONE", "\033[32m"
        lines.append(f"\n  {color}[{icon}]\033[0m {algo}")
        lines.append(f"    Tasks: {len(results)} (success: {s}, failed: {len(results) - s})")
        if fatal:
            lines.append(f"    Fatal Error: {fatal.error}")
        if algo in summary.best_by_algorithm:
            best = summary.best_by_algorithm[algo]
            if best:
                lines.append("    Best Results:")
                for ds, info in best.items():
                    if info["metrics"]:
                        ms = ", ".join(f"{k}={v:.4f}" for k, v in info["metrics"].items())
                        lines.append(f"      {ds}: {ms}  [HP: {info['hp']}]")
    lines.extend(["", "=" * 80, ""])
    return "\n".join(lines)


__all__ = [
    "ResultTable", "DatasetResult", "AlgorithmProgress",
    "TaskResult", "RunSummary", "BatchRunSummary",
    "save_results", "generate_report", "generate_batch_report",
]
