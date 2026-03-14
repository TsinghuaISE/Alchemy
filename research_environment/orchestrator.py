"""编排引擎 - 串联 plugin → mount → scheduler → executor → collector."""

from __future__ import annotations

import shlex
import threading
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .collector import ResultTable, TaskResult, RunSummary, save_results
from .executor import DockerExecutor, SSHSingularityExecutor, CommandResult
from .mount import MountBuilder
from .plugin import TaskPlugin, load_plugin
from .scheduler import GPUResource, NodeGPUScheduler


# ── 配置加载 ──

@dataclass
class ClusterConfig:
    """集群配置."""
    mode: str  # docker | singularity
    shared_root: str
    nodes: List[str]
    gpu_ids: List[int]
    max_per_gpu: int
    max_concurrent: int
    timeout: int
    docker: Dict[str, Any]
    singularity: Dict[str, Any]
    ssh_timeout: int = 30


def load_cluster_config(path: Path | str | None = None) -> ClusterConfig:
    """加载集群配置."""
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    sched = raw.get("scheduler", {})
    ssh = raw.get("ssh", {})
    return ClusterConfig(
        mode=raw.get("mode", "docker"),
        shared_root=raw.get("shared_root", ""),
        nodes=ssh.get("nodes", ["localhost"]),
        gpu_ids=sched.get("gpu_ids", [0]),
        max_per_gpu=sched.get("max_per_gpu", 1),
        max_concurrent=sched.get("max_concurrent", 8),
        timeout=raw.get("timeout", 3600),
        docker=raw.get("docker", {}),
        singularity=raw.get("singularity", {}),
        ssh_timeout=ssh.get("timeout", 30),
    )


def load_task_config(task_dir: Path) -> dict:
    """加载任务 config.yaml."""
    with open(task_dir / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 任务规格 ──

@dataclass
class TaskSpec:
    """单个待执行任务的完整规格."""
    domain: str
    task: str
    model: str
    dataset: str
    algorithm_path: Path
    hyperparameter_path: Path
    hp_name: str
    run_id: str = "default"


# ── 执行器工厂 ──

def _create_executor(cluster: ClusterConfig, bind_paths: List[str],
                     gpu: GPUResource, task_config: dict):
    """根据集群配置创建执行器."""
    task_key = f"{task_config['task']['domain']}/{task_config['task']['task']}"

    if cluster.mode == "docker":
        docker = cluster.docker
        image = docker.get("images", {}).get(task_key, "")
        return DockerExecutor(
            image=image,
            bind_paths=bind_paths,
            gpu_id=gpu.gpu_id,
            conda_env=docker.get("conda_env") or None,
            extra_args=docker.get("extra_args", []),
        )

    # singularity
    sing = cluster.singularity
    sif = sing.get("sif_images", {}).get(task_key, "")
    return SSHSingularityExecutor(
        sif_path=sif,
        bind_paths=bind_paths,
        node=gpu.node,
        gpu_id=gpu.gpu_id,
        singularity_exe=sing.get("exe", "singularity"),
        containall=sing.get("containall", False),
    )


def _build_command(task_config: dict, spec: TaskSpec,
                   gpu: GPUResource) -> List[str]:
    """从 config.yaml 构建容器内执行命令."""
    execution = task_config.get("execution", {})
    entry = execution.get("entry", "python main.py")
    args_template = task_config.get("args", {}).get("template", "")

    args_str = args_template.format(
        model=spec.model,
        dataset=spec.dataset,
        gpu=gpu.gpu_id,
    )
    return shlex.split(entry) + shlex.split(args_str)


# ── 编排引擎 ──

class RunOrchestrator:
    """编排引擎: 读 config → 展开超参 → 分配 GPU → 执行 → 收集结果.

    Usage:
        orch = RunOrchestrator(task_dir, cluster_config)
        summary = orch.run(model, algorithm_path, hp_path, run_id)
    """

    def __init__(self, task_dir: Path,
                 cluster: Optional[ClusterConfig] = None):
        self.task_dir = Path(task_dir)
        self.cluster = cluster or load_cluster_config()
        self.task_config = load_task_config(self.task_dir)
        self.plugin = load_plugin(
            self.task_config["task"]["domain"],
            self.task_config["task"]["task"],
        )
        self._mount_builder = MountBuilder(self.cluster.shared_root)

    def run(
        self,
        model: str,
        algorithm_path: Path,
        hp_path: Path,
        run_id: str = "default",
        fail_fast: bool = False,
        max_workers: Optional[int] = None,
        streaming: bool = False,
    ) -> RunSummary:
        """执行完整实验流程."""
        tc = self.task_config
        domain = tc["task"]["domain"]
        task = tc["task"]["task"]
        eval_cfg = tc.get("eval", {})
        datasets = tc.get("data", {}).get("datasets", [])

        # 1. 展开超参
        hp_dir = hp_path.parent / f"{run_id}_expanded"
        hp_dir.mkdir(parents=True, exist_ok=True)
        hp_files = self.plugin.expand_hp(hp_path, hp_dir)

        # 2. 生成 TaskSpec 列表
        specs = [
            TaskSpec(domain=domain, task=task, model=model,
                     dataset=ds, algorithm_path=algorithm_path,
                     hyperparameter_path=hp, hp_name=hp.stem,
                     run_id=run_id)
            for ds in datasets
            for hp in hp_files
        ]
        if not specs:
            return RunSummary(domain=domain, task=task, model=model)

        # 3. 创建调度器 + 结果表
        scheduler = NodeGPUScheduler(
            nodes=self.cluster.nodes,
            gpu_ids=self.cluster.gpu_ids,
            max_per_gpu=self.cluster.max_per_gpu,
        )
        table = ResultTable(
            datasets=datasets,
            metrics=eval_cfg.get("metrics", []),
            total_tasks=len(specs),
            primary_metric=eval_cfg.get("primary_metric", ""),
            higher_is_better=eval_cfg.get("higher_is_better", True),
        )
        summary = RunSummary(domain=domain, task=task, model=model,
                             total_tasks=len(specs))

        # 4. 定义单任务执行函数
        def run_one(spec: TaskSpec, gpu: GPUResource,
                    stop_event: threading.Event) -> TaskResult:
            if stop_event.is_set():
                return TaskResult(dataset=spec.dataset, hp_name=spec.hp_name)

            table.mark_running(spec.dataset)

            # 构建挂载路径
            bind_paths = self._mount_builder.build(
                tc.get("mount", {}),
                model=spec.model, dataset=spec.dataset,
                algorithm_path=spec.algorithm_path,
                hyperparameter_path=spec.hyperparameter_path,
                run_id=spec.run_id,
            )

            # 创建执行器并执行
            executor = _create_executor(self.cluster, bind_paths, gpu, tc)
            command = _build_command(tc, spec, gpu)
            workdir = tc.get("execution", {}).get("workdir", "")

            # 流式指标回调
            def on_line(line: str, is_err: bool):
                if not is_err:
                    try:
                        metrics = self.plugin.parse_output(line)
                        if metrics:
                            table.update(spec.dataset, metrics,
                                         spec.hp_name, mark_done=False)
                    except Exception:
                        pass

            cmd_result: CommandResult = executor.execute(
                command=command,
                work_dir=workdir,
                timeout=self.cluster.timeout,
                streaming=streaming,
                line_callback=on_line,
                stop_event=stop_event,
            )

            # 解析最终结果
            metrics = {}
            if cmd_result.return_code == 0:
                try:
                    metrics = self.plugin.parse_output(cmd_result.stdout)
                except Exception:
                    pass

            success = cmd_result.return_code == 0 and bool(metrics)
            result = TaskResult(
                dataset=spec.dataset,
                hp_name=spec.hp_name,
                metrics=metrics,
                success=success,
                duration=cmd_result.duration,
                error=cmd_result.stderr if not success else None,
                raw_stdout=cmd_result.stdout if not success else None,
                raw_stderr=cmd_result.stderr if not success else None,
                node=gpu.node,
                gpu_id=gpu.gpu_id,
            )

            # 更新结果表和摘要
            if success and metrics:
                table.update(spec.dataset, metrics, spec.hp_name)
            table.mark_completed(spec.dataset, success)
            summary.add_result(result)

            if streaming:
                table.print()

            return result

        # 5. 批量执行
        table.start_live()
        try:
            scheduler.run_batch(
                tasks=specs,
                run_fn=run_one,
                max_workers=max_workers,
                fail_fast=fail_fast,
            )
        finally:
            table.stop_live()
            table.print()

        # 6. 汇总
        summary.finalize(table.get_best_results())
        return summary
