from __future__ import annotations

import argparse

from loguru import logger

from ..agent.memory.domain_knowledge.learner import KnowledgeLearner
from ..utils.config import load_config
from ..utils.llm_client import LLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="手动执行的 domain knowledge 学习器")
    parser.add_argument("--domain", required=True, help="任务 domain，例如 Recsys")
    parser.add_argument("--task", required=True, help="任务 task，例如 MMRec")
    parser.add_argument("--dry-run", action="store_true", help="只生成学习结果，不写 knowledge.md")
    parser.add_argument("--min-evidence", type=int, default=3, help="形成知识条目的最小证据数")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    cfg = load_config()
    llm = LLMClient(model=cfg.model, base_url=cfg.base_url)
    learner = KnowledgeLearner(
        llm,
        domain=args.domain,
        task=args.task,
        min_evidence=args.min_evidence,
    )
    result = learner.run(dry_run=args.dry_run)

    logger.info(f"收集到 {len(result.trajectories)} 条 trajectory")
    logger.info(f"知识文件路径: {result.knowledge_path}")

    if args.dry_run:
        if result.learned_markdown:
            print(result.learned_markdown)
        else:
            logger.warning("没有形成新的知识条目")
        return 0

    logger.info("knowledge.md 已更新")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
