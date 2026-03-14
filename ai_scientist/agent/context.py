from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Hypothesis:
    id: int
    round_num: int
    description: str
    algorithm_code: str
    metrics_after: dict
    outcome: str  # improvement | regression | neutral | error
    error: str | None = None


@dataclass
class RunResult:
    paper_id: str
    improved: bool
    seed_metrics: dict
    best_metrics: dict
    total_hypotheses: int
    improvements: int
    total_rounds: int
    duration: float
    output_dir: str


@dataclass
class RunContext:
    paper_id: str
    seed_code: str
    seed_metrics: dict
    seed_method: str
    seed_hp_desc: str
    output_dir: Path
    current_code: str = ""
    current_hp: str | None = None
    hypotheses: list[Hypothesis] = field(default_factory=list)
