"""TaskPlugin 抽象基类 + 自动发现."""

import importlib
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Iterator

import yaml


class TaskPlugin(ABC):
    """现有的系统, 在每个深度学习任务新集成, 需要实现这样的一个接口, 方便系统做超参数展开, 从执行输出中解析指标 
    后续我们会将所有任务的输出方式, 解析方式, 超参数展开方式进行统一. 
    这个接口在后续的版本中会被废除  
    """

    @abstractmethod
    def expand_hp(self, hp_yaml: Path, output_dir: Path) -> list[Path]:
        """超参展开: YAML 文件 → 多个展开后的 YAML 文件."""

    @abstractmethod
    def parse_output(self, stdout: str) -> dict[str, float]:
        """从执行输出中解析指标."""

    # ---- YAML 工具方法 ----

    @staticmethod
    def load_yaml(path: Path) -> dict:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    @staticmethod
    def save_yaml(data: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    @staticmethod
    def grid_expand(sweep: dict[str, list]) -> Iterator[dict]:
        """网格展开: {k: [v1, v2], ...} → [{k: v1, ...}, {k: v2, ...}, ...]."""
        keys = list(sweep.keys())
        for combo in product(*(sweep[k] for k in keys)):
            yield dict(zip(keys, combo))


def load_plugin(domain: str, task: str) -> TaskPlugin:
    """自动发现并加载 tasks/{domain}/{task}/plugin.py 中的 TaskPlugin 实现."""
    module = importlib.import_module(f"research_environment.tasks.{domain}.{task}.plugin")
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, TaskPlugin) and obj is not TaskPlugin:
            return obj()
    raise ValueError(f"No TaskPlugin found in tasks/{domain}/{task}/plugin.py")
