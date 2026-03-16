"""ScientificAgent - 科学发现 Agent."""

import time
from pathlib import Path

from loguru import logger

from .. import PROJ_ROOT
from . import evaluator, prompt, response_parser
from .context import Hypothesis, RunContext, RunResult
from ..utils import save


class ScientificAgent:

    def __init__(self, llm, domain: str, task: str, metric: str, *,
                 higher_is_better: bool = True, patience: int = 3,
                 max_rounds: int = 100, threshold: float = 0.01,
                 max_fix_retries: int = 0, timeout: int = 86400,
                 temperature: float | None = None):
        self.llm = llm
        self.domain = domain
        self.task = task
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.patience = patience
        self.max_rounds = max_rounds
        self.threshold = threshold
        self.max_fix_retries = max_fix_retries
        self.timeout = timeout
        self.temperature = temperature

        kp = PROJ_ROOT / "domain_knowledge" / domain / task / "knowledge.md"
        self.domain_knowledge = kp.read_text(encoding="utf-8") if kp.exists() else ""

    def run(self, paper_id: str, seed_code: str, seed_metrics: dict,
            seed_hp_yaml: str | None = None,
            seed_method: str = "", seed_hp_desc: str = "") -> RunResult:
        ctx = RunContext(
            paper_id=paper_id, seed_code=seed_code, seed_metrics=seed_metrics,
            seed_method=seed_method, seed_hp_desc=seed_hp_desc,
            output_dir=PROJ_ROOT / "output" / self.domain / self.task / paper_id
            / (f"temp_{self.temperature}" if self.temperature is not None else "default"),
            current_code=seed_code, current_hp=seed_hp_yaml,
        )
        ctx.output_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        no_improve, round_num = 0, 0

        while no_improve < self.patience and round_num < self.max_rounds:
            hyp, improved = self._run_one_round(ctx, round_num)
            ctx.hypotheses.append(hyp)
            no_improve = 0 if improved else no_improve + 1
            round_num += 1

        improvements = [h for h in ctx.hypotheses if h.outcome == "improvement"]
        best = (evaluator.get_best_hypothesis(improvements, self.metric, self.higher_is_better)
                if improvements else None)
        save.save_results(ctx.output_dir, paper_id, seed_metrics, ctx.hypotheses, best)

        duration = time.time() - start
        logger.info(f"完成: {len(ctx.hypotheses)} hypotheses, "
                    f"{len(improvements)} improvements, {duration:.0f}s")
        return RunResult(
            paper_id=paper_id, improved=bool(improvements),
            seed_metrics=seed_metrics, best_metrics=best.metrics_after if best else seed_metrics,
            total_hypotheses=len(ctx.hypotheses), improvements=len(improvements),
            total_rounds=round_num, duration=duration, output_dir=str(ctx.output_dir),
        )

    # ── LLM 交互 ─────────────────────────────────────────────

    def _shared_vars(self, ctx: RunContext) -> dict:
        return dict(
            paper_id=ctx.paper_id, domain=self.domain, task=self.task,
            hypotheses=ctx.hypotheses, seed_metrics=ctx.seed_metrics,
            domain_knowledge_context=self.domain_knowledge or None,
        )

    def _ask_llm(self, template: str, **kw) -> str:
        system_prompt, user_prompt = prompt.render(template, **kw)
        return self.llm.generate(system_prompt, user_prompt, temperature=self.temperature)

    def _generate_hypothesis(self, ctx: RunContext) -> tuple[str, str, str | None]:
        response = self._ask_llm(
            "hypothesis", **self._shared_vars(ctx),
            seed_code=ctx.seed_code, seed_method_description=ctx.seed_method,
        )
        return response_parser.extract_hypothesis(response)

    def _generate_artifact(self, ctx: RunContext, template: str,
                           content: str | None, *, is_first: bool,
                           error: str | None, path: Path,
                           is_yaml: bool = False, **extra_vars) -> str:
        response = self._ask_llm(
            template, **self._shared_vars(ctx),
            is_first=is_first, has_error=bool(error), execution_error=error,
            **extra_vars,
        )
        return response_parser.extract_artifact(response, content, is_yaml=is_yaml, path=path)

    # ── 核心循环 ──────────────────────────────────────────────

    def _run_one_round(self, ctx: RunContext, round_num: int) -> tuple[Hypothesis, bool]:
        logger.info(f"━━━ Round {round_num + 1} ━━━")
        round_dir = ctx.output_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        desc, rationale, key_changes = self._generate_hypothesis(ctx)
        logger.info(f"假设: {desc[:80]}...")

        return self._implement_and_evaluate(
            ctx, round_num, desc, rationale, key_changes,
            is_first=(round_num == 0),
            algo_path=round_dir / "algorithm.py",
            hp_path=round_dir / "hyperparameter.yaml",
        )

    def _implement_and_evaluate(self, ctx: RunContext, round_num: int,
                                desc: str, rationale: str, key_changes: str | None,
                                *, is_first: bool, algo_path: Path,
                                hp_path: Path) -> tuple[Hypothesis, bool]:
        code, hp, error = ctx.current_code, ctx.current_hp, None
        first_attempt = is_first

        for attempt in range(1 + self.max_fix_retries):
            code = self._generate_artifact(
                ctx, "code_implementation", code,
                is_first=is_first, error=error, path=algo_path,
                hypothesis_description=desc, hypothesis_rationale=rationale,
                hypothesis_key_changes=key_changes, current_code=code,
                seed_method_description=ctx.seed_method if first_attempt else None,
            )
            hp = self._generate_artifact(
                ctx, "hyperparameter", hp,
                is_first=is_first, error=error, path=hp_path, is_yaml=True,
                algorithm_code=code, current_hyperparameter=hp,
                seed_hyperparameter_description=ctx.seed_hp_desc if first_attempt else None,
            )

            try:
                metrics = evaluator.evaluate(
                    self.domain, self.task, ctx.paper_id,
                    code, algo_path, hp_path, self.timeout)
            except Exception as e:
                error = str(e)
                logger.warning(f"执行失败 (attempt {attempt + 1}): {error[:200]}")
                code, hp = _read_back(algo_path, hp_path, hp)
                first_attempt = False
                continue

            code, hp = _read_back(algo_path, hp_path, hp)
            ctx.current_code, ctx.current_hp = code, hp

            outcome = evaluator.compare_metrics(
                self.metric, ctx.seed_metrics, metrics,
                self.higher_is_better, self.threshold)
            logger.info(f"假设 #{len(ctx.hypotheses) + 1} → {outcome}")
            return Hypothesis(
                id=len(ctx.hypotheses) + 1, round_num=round_num,
                description=desc, algorithm_code=code,
                metrics_after=metrics, outcome=outcome,
            ), outcome == "improvement"

        logger.error(f"重试达上限 ({self.max_fix_retries})")
        return Hypothesis(
            id=len(ctx.hypotheses) + 1, round_num=round_num,
            description=desc, algorithm_code="",
            metrics_after={}, outcome="error", error=error,
        ), False


def _read_back(algo_path: Path, hp_path: Path, current_hp: str | None) -> tuple[str, str | None]:
    code = algo_path.read_text(encoding="utf-8")
    hp = hp_path.read_text(encoding="utf-8") if hp_path.exists() else current_hp
    return code, hp
