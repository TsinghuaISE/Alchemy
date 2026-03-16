import json
from pathlib import Path

from .. import PROJ_ROOT


def seed_dir(domain: str, task: str, paper_id: str) -> Path:
    return PROJ_ROOT / "seed_baseline" / domain / task / paper_id


def load_code(domain: str, task: str, paper_id: str) -> str:
    return (seed_dir(domain, task, paper_id) / "algorithm.py").read_text(encoding="utf-8")


def load_hyperparameter(domain: str, task: str, paper_id: str) -> str | None:
    path = seed_dir(domain, task, paper_id) / "hyperparameter.yaml"
    return path.read_text(encoding="utf-8") if path.exists() else None


def load_metrics(domain: str, task: str, paper_id: str) -> dict:
    path = seed_dir(domain, task, paper_id) / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("metrics", {})


def load_meta(domain: str, task: str, paper_id: str) -> dict:
    path = seed_dir(domain, task, paper_id) / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_seed_metrics(domain: str, task: str, paper_id: str, metrics: dict):
    (seed_dir(domain, task, paper_id) / "metrics.json").write_text(
        json.dumps({"paper_id": paper_id, "metrics": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
