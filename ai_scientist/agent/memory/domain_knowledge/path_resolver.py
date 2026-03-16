from __future__ import annotations

from pathlib import Path

from .... import PROJ_ROOT


TASK_ALIASES = {
    ("Recsys", "MMRec"): ("recsys", "multimodal_rec"),
}


def output_task_dir(domain: str, task: str) -> Path:
    return PROJ_ROOT / "output" / domain / task


def resolve_knowledge_path(domain: str, task: str) -> Path:
    alias = TASK_ALIASES.get((domain, task))
    candidates = []
    if alias:
        candidates.append(PROJ_ROOT / "domain_knowledge" / alias[0] / alias[1] / "knowledge.md")
    candidates.extend(
        [
            PROJ_ROOT / "domain_knowledge" / domain / task / "knowledge.md",
            PROJ_ROOT / "domain_knowledge" / domain.lower() / task.lower() / "knowledge.md",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if alias:
        return PROJ_ROOT / "domain_knowledge" / alias[0] / alias[1] / "knowledge.md"
    return PROJ_ROOT / "domain_knowledge" / domain.lower() / task.lower() / "knowledge.md"
