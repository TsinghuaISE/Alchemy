from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...prompt import render
from ....utils.config import load_config
from ....utils.llm_client import LLMClient

from .collector import RunTrajectory, collect_trajectories
from .path_resolver import output_task_dir, resolve_knowledge_path
from .writer import write_knowledge_file


@dataclass
class LearnResult:
    knowledge_path: Path
    trajectories: list[RunTrajectory]
    learned_markdown: str
    updated_markdown: str | None = None


class KnowledgeLearner:

    def __init__(self, llm: LLMClient, *, domain: str, task: str, min_evidence: int = 3):
        self.llm = llm
        self.domain = domain
        self.task = task
        self.min_evidence = min_evidence
        self.metric = self._resolve_metric(domain, task)
        self.knowledge_path = resolve_knowledge_path(domain, task)

    def run(self, *, dry_run: bool = False) -> LearnResult:
        trajectories = collect_trajectories(output_task_dir(self.domain, self.task))
        learned_markdown = self._synthesize_markdown(trajectories)
        result = LearnResult(
            knowledge_path=self.knowledge_path,
            trajectories=trajectories,
            learned_markdown=learned_markdown,
        )
        if dry_run:
            return result

        result.updated_markdown = write_knowledge_file(
            self.knowledge_path,
            domain=self.domain,
            task=self.task,
            learned_markdown=learned_markdown or "## no_new_patterns\n- Title: 暂无新知识\n- Tags: trajectory-learning\n- Source: trajectory_learning\n\n当前轨迹中没有满足证据阈值的新稳定模式。",
        )
        return result

    def _resolve_metric(self, domain: str, task: str) -> str:
        cfg = load_config()
        for task_cfg in cfg.tasks:
            if task_cfg.domain == domain and task_cfg.task == task:
                return task_cfg.metric
        return ""

    def _synthesize_markdown(self, trajectories: list[RunTrajectory]) -> str:
        if not trajectories:
            return ""

        evidence_payload = {
            "domain": self.domain,
            "task": self.task,
            "metric": self.metric,
            "min_evidence": self.min_evidence,
            "trajectory_count": len(trajectories),
            "paper_ids": [trajectory.paper_id for trajectory in trajectories],
            "trajectories": [self._summarize_trajectory(trajectory) for trajectory in trajectories],
        }

        system_prompt, user_prompt = render(
            "knowledge_learning",
            domain=self.domain,
            task=self.task,
            metric=self.metric,
            min_evidence=self.min_evidence,
            trajectory_count=len(trajectories),
            paper_ids=[trajectory.paper_id for trajectory in trajectories],
            existing_knowledge=self._read_existing_knowledge(),
            evidence_payload=json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        )
        return self.llm.generate(system_prompt, user_prompt).strip()

    def _read_existing_knowledge(self) -> str:
        if not self.knowledge_path.exists():
            return ""
        content = self.knowledge_path.read_text(encoding="utf-8")
        content = content.replace("<!-- LEARNED_KNOWLEDGE:START -->", "")
        content = content.replace("<!-- LEARNED_KNOWLEDGE:END -->", "")
        return content.strip()

    def _summarize_trajectory(self, trajectory: RunTrajectory) -> dict:
        seed_value = trajectory.seed_metrics.get(self.metric) if self.metric else None
        hypotheses = []
        for hypothesis in trajectory.hypotheses:
            metric_after = hypothesis.metrics_after.get(self.metric) if self.metric else None
            metric_delta = None
            if seed_value is not None and metric_after is not None:
                metric_delta = round(metric_after - seed_value, 6)
            hypotheses.append(
                {
                    "id": hypothesis.id,
                    "description": hypothesis.description,
                    "outcome": hypothesis.outcome,
                    "metric_after": metric_after,
                    "metric_delta": metric_delta,
                    "error": _first_line(hypothesis.error),
                }
            )
        return {
            "paper_id": trajectory.paper_id,
            "seed_metric": seed_value,
            "improvement_count": sum(1 for h in trajectory.hypotheses if h.outcome == "improvement"),
            "error_count": sum(1 for h in trajectory.hypotheses if h.outcome == "error"),
            "hypotheses": hypotheses,
        }


def _first_line(text: str | None) -> str | None:
    if not text:
        return None
    return text.strip().splitlines()[0][:200]
