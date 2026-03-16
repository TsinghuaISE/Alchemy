"""挂载路径构建器 - 根据任务配置生成容器 bind 路径."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


class MountBuilder:
    """根据 config.yaml 的 mount 配置和运行时参数生成容器 bind 路径."""

    def __init__(self, shared_root: str, task_root: str | Path):
        self.shared_root = Path(shared_root)
        self.task_root = Path(task_root)

    def build(self, mount_config: Dict[str, dict], *,
              model: str, dataset: str,
              algorithm_path: Path, hyperparameter_path: Path,
              run_id: str = "default") -> List[str]:
        """构建挂载路径列表.

        Args:
            mount_config: config.yaml 中的 mount 字典,
                          每个值含 host/container 两个 key.
            其余参数: 用于路径模板变量替换.

        Returns:
            ["host_path:container_path", ...] 格式的挂载列表.
        """
        variables = {
            "model": model,
            "model_lower": model.lower(),
            "dataset": dataset,
            "algorithm_path": str(algorithm_path),
            "hyperparameter_path": str(hyperparameter_path),
            "run_id": run_id,
            "task_root": str(self.task_root),
            "shared_root": str(self.shared_root),
        }

        binds = []
        for name, mount in mount_config.items():
            host = self._expand(mount["host"], variables)
            container = self._expand(mount["container"], variables)
            host_path = self._resolve_host_path(host)

            if not host_path.exists() and any(
                    key in name.lower() for key in ("log", "output", "checkpoint", "result")):
                host_path.mkdir(parents=True, exist_ok=True)

            binds.append(f"{host_path}:{container}")
        return binds

    def _resolve_host_path(self, host: str) -> Path:
        host_path = Path(host)
        if host_path.is_absolute():
            return host_path
        return self.task_root / host_path

    @staticmethod
    def _expand(path: str, variables: Dict[str, str]) -> str:
        for k, v in variables.items():
            path = path.replace(f"{{{k}}}", v)
        return path
