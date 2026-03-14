"""GPU 资源调度器 - 多节点 GPU 分配与批量任务执行."""

from __future__ import annotations

import itertools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class GPUResource:
    """一块 GPU 资源."""
    node: str
    gpu_id: int


class NodeGPUScheduler:
    """多节点 GPU 调度器, 线程安全.

    通过 round-robin 分配 GPU, 支持每 GPU 并发上限.
    """

    def __init__(self, nodes: List[str], gpu_ids: List[int],
                 max_per_gpu: int = 1):
        self._resources = [
            GPUResource(node=n, gpu_id=g)
            for n in nodes for g in gpu_ids
        ]
        if not self._resources:
            raise ValueError("No GPU resources available")

        self._max_per_gpu = max_per_gpu
        self._usage: Dict[GPUResource, int] = {r: 0 for r in self._resources}
        self._cond = threading.Condition()
        self._cycle = itertools.cycle(self._resources)
        self._cycle_pos = 0

    @property
    def total_gpus(self) -> int:
        return len(self._resources)

    @property
    def max_concurrent(self) -> int:
        return len(self._resources) * self._max_per_gpu

    def acquire(self, stop_event: Optional[threading.Event] = None) -> Optional[GPUResource]:
        """获取一块 GPU, 阻塞直到有空闲资源或 stop_event 触发."""
        with self._cond:
            while True:
                if stop_event and stop_event.is_set():
                    return None
                # round-robin 查找可用 GPU
                for _ in range(len(self._resources)):
                    r = next(self._cycle)
                    if self._usage[r] < self._max_per_gpu:
                        self._usage[r] += 1
                        return r
                # 全部满载, 等待释放
                self._cond.wait(timeout=0.5)

    def release(self, resource: GPUResource) -> None:
        """归还 GPU 资源."""
        with self._cond:
            if resource in self._usage:
                self._usage[resource] = max(0, self._usage[resource] - 1)
            self._cond.notify_all()

    def run_batch(
        self,
        tasks: List[Any],
        run_fn: Callable[[Any, GPUResource, threading.Event], Any],
        max_workers: Optional[int] = None,
        fail_fast: bool = False,
    ) -> List[Any]:
        """批量执行任务, 自动分配/回收 GPU.

        Args:
            tasks: 任务列表.
            run_fn: 执行函数, 签名 (task, gpu_resource, stop_event) -> result.
            max_workers: 最大并发线程数, 默认为 max_concurrent.
            fail_fast: 遇到失败是否立即停止剩余任务.

        Returns:
            结果列表 (顺序与 tasks 对应).
        """
        workers = min(max_workers or self.max_concurrent, len(tasks))
        stop_event = threading.Event()
        results: List[Any] = [None] * len(tasks)

        def _run(idx: int, task: Any) -> tuple:
            resource = self.acquire(stop_event)
            if resource is None:
                return idx, None
            try:
                result = run_fn(task, resource, stop_event)
                return idx, result
            finally:
                self.release(resource)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run, i, t): i for i, t in enumerate(tasks)}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                if fail_fast and result is not None and not getattr(result, 'success', True):
                    stop_event.set()

        return results
