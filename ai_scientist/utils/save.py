import json
from datetime import datetime
from pathlib import Path

from loguru import logger


def save_results(output_dir: Path, paper_id: str, seed_metrics: dict,
                 hypotheses: list, best=None):
    discovery_dir = output_dir / "discovery"
    discovery_dir.mkdir(parents=True, exist_ok=True)

    for h in hypotheses:
        if h.algorithm_code:
            (discovery_dir / f"hypothesis_{h.id}_algorithm.py").write_text(
                h.algorithm_code, encoding="utf-8")

    summary = {
        "paper_id": paper_id,
        "seed_metrics": seed_metrics,
        "hypotheses": [
            {"id": h.id, "description": h.description, "outcome": h.outcome,
             "metrics_after": h.metrics_after, "error": h.error}
            for h in hypotheses
        ],
        "timestamp": datetime.now().isoformat(),
    }
    (discovery_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if best:
        best_dir = output_dir / "best"
        best_dir.mkdir(exist_ok=True)
        (best_dir / "algorithm.py").write_text(best.algorithm_code, encoding="utf-8")
        logger.info(f"最佳改进已保存: hypothesis #{best.id}")
