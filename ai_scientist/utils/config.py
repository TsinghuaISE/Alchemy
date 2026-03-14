from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ai_scientist import PROJ_ROOT


@dataclass
class TaskConfig:
    domain: str
    task: str
    metric: str
    seeds: list[str] = field(default_factory=list)


@dataclass
class Config:
    model: str
    base_url: str
    tasks: list[TaskConfig]
    max_rounds: int = 10
    patience: int = 3


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = PROJ_ROOT / "config.yaml"
    path = Path(path)

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if "model" not in raw or "base_url" not in raw:
        raise ValueError(f"config.yaml 必须包含 'model' 和 'base_url': {path}")

    tasks = [
        TaskConfig(domain=t["domain"], task=t["task"],
                   metric=t.get("metric", ""), seeds=t.get("seeds", []))
        for t in raw.get("tasks", [])
    ]
    return Config(
        model=raw["model"], base_url=raw["base_url"],
        max_rounds=raw.get("max_rounds", Config.max_rounds),
        patience=raw.get("patience", Config.patience), tasks=tasks,
    )
