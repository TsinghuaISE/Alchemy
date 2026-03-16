"""python -m ai_scientist"""

from loguru import logger

from .agent import ScientificAgent
from .agent import evaluator
from .utils.config import load_config
from .utils.llm_client import LLMClient
from .utils import seed_loader


def main():
    cfg = load_config()
    llm = LLMClient(model=cfg.model, base_url=cfg.base_url)

    for task in cfg.tasks:
        logger.info(f"Task: {task.domain}/{task.task}, metric={task.metric}")
        agent = ScientificAgent(
            llm, domain=task.domain, task=task.task, metric=task.metric,
            patience=cfg.patience, max_rounds=cfg.max_rounds,
        )
        for seed_id in task.seeds:
            code = seed_loader.load_code(task.domain, task.task, seed_id)
            hp = seed_loader.load_hyperparameter(task.domain, task.task, seed_id)
            metrics = seed_loader.load_metrics(task.domain, task.task, seed_id)
            meta = seed_loader.load_meta(task.domain, task.task, seed_id)

            if not metrics:
                logger.info(f"未找到种子指标, 评测 baseline: {seed_id}")
                sd = seed_loader.seed_dir(task.domain, task.task, seed_id)
                metrics = evaluator.evaluate(
                    task.domain, task.task, seed_id, code,
                    sd / "algorithm.py", sd / "hyperparameter.yaml")
                seed_loader.save_seed_metrics(task.domain, task.task, seed_id, metrics)

            result = agent.run(
                seed_id, code, metrics, hp,
                seed_method=meta.get("method", ""),
                seed_hp_desc=meta.get("hyperparameter", ""),
            )
            logger.info(f"Seed {seed_id}: improved={result.improved}, "
                        f"best={result.best_metrics}")

    logger.info(f"完成, LLM token: {llm.total_tokens}")


if __name__ == "__main__":
    main()
