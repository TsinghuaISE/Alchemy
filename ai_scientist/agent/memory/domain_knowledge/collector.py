from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HypothesisEvidence:
    id: int
    description: str
    outcome: str
    metrics_after: dict
    error: str | None = None
    algorithm_path: str | None = None
    hyperparameter_path: str | None = None


@dataclass
class RunTrajectory:
    paper_id: str
    seed_metrics: dict
    hypotheses: list[HypothesisEvidence] = field(default_factory=list)
    discovery_path: Path | None = None


def collect_trajectories(task_output_dir: Path) -> list[RunTrajectory]:
    trajectories: list[RunTrajectory] = []
    if not task_output_dir.exists():
        return trajectories

    for paper_dir in sorted(path for path in task_output_dir.iterdir() if path.is_dir()):
        summary_path = paper_dir / "discovery" / "summary.json"
        if not summary_path.exists():
            continue

        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        hypotheses = []
        for hypothesis in raw.get("hypotheses", []):
            hypothesis_id = int(hypothesis.get("id", 0))
            round_dir = paper_dir / f"round_{hypothesis_id - 1}" if hypothesis_id > 0 else None
            algo_path = round_dir / "algorithm.py" if round_dir else None
            hp_path = round_dir / "hyperparameter.yaml" if round_dir else None
            hypotheses.append(
                HypothesisEvidence(
                    id=hypothesis_id,
                    description=str(hypothesis.get("description", "")).strip(),
                    outcome=str(hypothesis.get("outcome", "")).strip(),
                    metrics_after=hypothesis.get("metrics_after", {}) or {},
                    error=hypothesis.get("error"),
                    algorithm_path=str(algo_path) if algo_path and algo_path.exists() else None,
                    hyperparameter_path=str(hp_path) if hp_path and hp_path.exists() else None,
                )
            )

        trajectories.append(
            RunTrajectory(
                paper_id=str(raw.get("paper_id", paper_dir.name)),
                seed_metrics=raw.get("seed_metrics", {}) or {},
                hypotheses=hypotheses,
                discovery_path=summary_path,
            )
        )
    return trajectories
