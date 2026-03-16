"""python -m ai_scientist"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from .agent import ScientificAgent
from .agent import evaluator
from .utils.config import load_config
from .utils.llm_client import LLMClient
from .utils import seed_loader


def _run_experiment(llm, task, seed_id, temperature, cfg):
    """单个实验: 一个 seed × 一个 temperature."""
    tag = f"{task.domain}/{task.task}/{seed_id}/temp={temperature}"
    logger.info(f"开始实验: {tag}")

    agent = ScientificAgent(
        llm, domain=task.domain, task=task.task, metric=task.metric,
        patience=cfg.patience, max_rounds=cfg.max_rounds,
        temperature=temperature,
    )

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
    logger.info(f"实验完成: {tag}, improved={result.improved}, best={result.best_metrics}")
    return result


def main():
    cfg = load_config()
    llm = LLMClient(model=cfg.model, base_url=cfg.base_url)

    # 生成所有 (task, seed, temperature) 组合
    experiments = [
        (task, seed_id, temp)
        for task in cfg.tasks
        for seed_id in task.seeds
        for temp in cfg.temperatures
    ]

    total = len(experiments)
    logger.info(f"共 {total} 个实验, 最大并发 {cfg.max_concurrent}")

    if cfg.max_concurrent <= 1:
        # 串行
        for task, seed_id, temp in experiments:
            _run_experiment(llm, task, seed_id, temp, cfg)
    else:
        # 并发
        with ThreadPoolExecutor(max_workers=cfg.max_concurrent) as pool:
            futures = {
                pool.submit(_run_experiment, llm, task, seed_id, temp, cfg): (seed_id, temp)
                for task, seed_id, temp in experiments
            }
            for future in as_completed(futures):
                seed_id, temp = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"实验失败: {seed_id}/temp={temp}: {e}")

    logger.info(f"全部完成, LLM token: {llm.total_tokens}")


if __name__ == "__main__":
    main()
