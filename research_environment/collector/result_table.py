"""结果表格 - 基于 rich 的实验结果实时追踪与展示.

单算法时 algorithms 默认为 ["_"], 对外隐藏 algorithm 参数.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text


_DEFAULT_ALGO = "_"

STATUS_STYLE = {
    "pending": ("○ wait", "dim"),
    "running": ("◐ run", "cyan"),
    "done":    ("✓ done", "green"),
    "failed":  ("✗ fail", "red"),
    "stopped": ("⊘ stop", "yellow"),
}


@dataclass
class DatasetResult:
    """单个数据集的结果."""
    dataset: str
    metrics: Dict[str, float] = field(default_factory=dict)
    best_hp: str = "-"
    status: str = "pending"


@dataclass
class AlgorithmProgress:
    """算法执行进度."""
    total: int = 0
    completed: int = 0
    success: int = 0
    failed: int = 0
    status: str = "pending"
    current_round: int = 0

    @property
    def is_done(self) -> bool:
        return self.completed >= self.total


# ── 工具 ──

def _fmt(value: Optional[float], status: str) -> Text:
    if value is None or value == 0:
        return Text("running", style="cyan italic") if status == "running" else Text("...", style="dim")
    return Text(f"{value:.4f}")


def _is_better(new: float, old: float, higher_is_better: bool) -> bool:
    if higher_is_better:
        return new > old
    return (old == 0 and new > 0) or (old > 0 and new < old)


def _improvement_msg(label: str, metric: str, old: float, new: float,
                     hp: str, higher_is_better: bool) -> str:
    if old > 0:
        pct = abs(new - old) / old * 100
        sign = "+" if higher_is_better else "-"
        return f"✓ {label}: {metric} {old:.4f} → {new:.4f} ({sign}{pct:.1f}%) [{hp}]"
    return f"✓ {label}: {metric} = {new:.4f} [{hp}]"


# ══════════════════════════════════════════════════════════════════════

class ResultTable:
    """实验结果表格, 支持单算法和多算法模式.

    单算法: ResultTable(datasets=[...], metrics=[...])
    多算法: ResultTable(datasets=[...], metrics=[...], algorithms=[...])
    """

    def __init__(self, datasets: List[str], metrics: List[str],
                 total_tasks: int = 0, primary_metric: str = "recall@20",
                 higher_is_better: bool = True,
                 title: str = "Best Test Metrics",
                 algorithms: Optional[List[str]] = None):
        self.title = title
        self.datasets = list(datasets)
        self.metrics = list(metrics)
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better
        self.algorithms = list(algorithms) if algorithms else [_DEFAULT_ALGO]
        self.batch_mode = algorithms is not None

        self._lock = threading.Lock()
        self.results: Dict[str, Dict[str, DatasetResult]] = {
            a: {ds: DatasetResult(dataset=ds) for ds in datasets}
            for a in self.algorithms
        }
        self.progress: Dict[str, AlgorithmProgress] = {
            a: AlgorithmProgress(total=len(datasets)) for a in self.algorithms
        }
        self.stopped: Dict[str, bool] = {a: False for a in self.algorithms}
        self.last_improvement: Optional[str] = None
        self._console = Console()
        self._live: Optional[Live] = None

    def _algo(self, algorithm: Optional[str] = None) -> str:
        return algorithm or _DEFAULT_ALGO

    # ── 状态更新 ──

    def mark_running(self, dataset: str, algorithm: Optional[str] = None):
        a = self._algo(algorithm)
        with self._lock:
            if a in self.results and dataset in self.results[a]:
                self.results[a][dataset].status = "running"
                if self.progress[a].status == "pending":
                    self.progress[a].status = "running"

    def update(self, dataset: str, metrics: Dict[str, float],
               hp: str = "-", mark_done: bool = True,
               algorithm: Optional[str] = None) -> Optional[str]:
        a = self._algo(algorithm)
        with self._lock:
            if a not in self.results or dataset not in self.results[a]:
                return None
            r = self.results[a][dataset]
            old = r.metrics.get(self.primary_metric, 0)
            new = metrics.get(self.primary_metric, 0)
            improvement = None
            if _is_better(new, old, self.higher_is_better):
                label = f"{a}/{dataset}" if self.batch_mode else dataset
                improvement = _improvement_msg(
                    label, self.primary_metric, old, new, hp, self.higher_is_better)
                r.metrics, r.best_hp = metrics, hp
                self.last_improvement = improvement
            if mark_done:
                r.status = "done"
            return improvement

    def mark_completed(self, dataset: str, success: bool = True,
                       algorithm: Optional[str] = None):
        a = self._algo(algorithm)
        with self._lock:
            self.completed_tasks += 1
            if a not in self.progress:
                return
            p = self.progress[a]
            p.completed += 1
            if success:
                p.success += 1
            else:
                p.failed += 1
            if p.is_done:
                p.status = "done" if p.failed == 0 else "failed"

    def set_status(self, dataset: str, status: str,
                   algorithm: Optional[str] = None):
        a = self._algo(algorithm)
        with self._lock:
            if a in self.results and dataset in self.results[a]:
                self.results[a][dataset].status = status

    def mark_algorithm_stopped(self, algorithm: str):
        with self._lock:
            self.stopped[algorithm] = True
            if algorithm in self.progress:
                self.progress[algorithm].status = "stopped"

    def update_algorithm_round(self, algorithm: str, round_num: int):
        with self._lock:
            if algorithm in self.progress:
                self.progress[algorithm].current_round = round_num

    def reset_algorithm_for_new_round(self, algorithm: str):
        with self._lock:
            if algorithm not in self.results:
                return
            for ds in self.datasets:
                if ds in self.results[algorithm]:
                    r = self.results[algorithm][ds]
                    r.status, r.metrics, r.best_hp = "pending", {}, "-"
            if algorithm in self.progress:
                rd = self.progress[algorithm].current_round
                self.progress[algorithm] = AlgorithmProgress(
                    total=len(self.datasets), current_round=rd + 1)
            self.stopped[algorithm] = False

    # ── 渲染 ──

    def build_table(self) -> Table:
        with self._lock:
            table = Table(title=self.title, show_lines=True)
            if self.batch_mode:
                table.add_column("Algorithm", style="bold")
                table.add_column("Round", justify="center")
            table.add_column("Dataset", style="bold" if not self.batch_mode else None)
            for m in self.metrics[:4]:
                table.add_column(m, justify="center",
                                 style="cyan" if m == self.primary_metric else None)
            table.add_column("Status", justify="center")
            table.add_column("HP", justify="center")

            for algo in self.algorithms:
                rd = self.progress[algo].current_round
                for j, ds in enumerate(self.datasets):
                    r = self.results[algo][ds]
                    sk = "stopped" if self.stopped.get(algo) else r.status
                    icon, sty = STATUS_STYLE.get(sk, STATUS_STYLE["pending"])
                    row: list = []
                    if self.batch_mode:
                        row += [algo if j == 0 else "", str(rd) if j == 0 else ""]
                    row.append(ds)
                    for m in self.metrics[:4]:
                        row.append(_fmt(r.metrics.get(m), r.status))
                    row += [Text(icon, style=sty), r.best_hp]
                    table.add_row(*row)

            table.caption = (
                f"Progress: {self.completed_tasks}/{self.total_tasks}"
                + (f"  |  {self.last_improvement}" if self.last_improvement else ""))
            return table

    def print(self):
        self._console.print(self.build_table())

    def start_live(self) -> Live:
        self._live = Live(self.build_table(), console=self._console,
                          refresh_per_second=2)
        self._live.start()
        return self._live

    def refresh_live(self):
        if self._live:
            self._live.update(self.build_table())

    def stop_live(self):
        if self._live:
            self._live.stop()
            self._live = None

    # ── 查询 ──

    def get_best_results(self) -> dict:
        if self.batch_mode:
            return {a: dr.copy() for a, dr in self.results.items()}
        return self.results[_DEFAULT_ALGO].copy()

    def get_overall_best(self) -> Optional[dict]:
        best, best_val = None, 0.0
        for algo_results in self.results.values():
            for ds, r in algo_results.items():
                v = r.metrics.get(self.primary_metric, 0)
                if v > best_val:
                    best_val = v
                    best = {"dataset": ds, "metrics": r.metrics.copy(),
                            "hp_info": r.best_hp}
        return best

    def summary_dict(self) -> dict:
        def _ds_dict(ar: Dict[str, DatasetResult]) -> dict:
            return {ds: {"metrics": r.metrics, "best_hp": r.best_hp,
                         "status": r.status} for ds, r in ar.items()}

        with self._lock:
            if not self.batch_mode:
                return {
                    "total_tasks": self.total_tasks,
                    "completed_tasks": self.completed_tasks,
                    "datasets": _ds_dict(self.results[_DEFAULT_ALGO]),
                }
            return {
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "algorithms": {
                    a: {
                        "progress": asdict(self.progress[a]),
                        "stopped": self.stopped[a],
                        "datasets": _ds_dict(ar),
                    }
                    for a, ar in self.results.items()
                },
            }
