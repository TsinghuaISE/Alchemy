"""TimeSeries/anomaly_detection 插件."""

import re
from pathlib import Path

from research_environment.plugin import TaskPlugin

_METRIC_RE = re.compile(r"(mse|mae|rmse)\s*[:=]\s*([\d.]+)", re.IGNORECASE)


class AnomalyDetectionPlugin(TaskPlugin):
    """异常检测插件.

    指标解析: 从日志中提取 mse, mae 等 test 指标.
    """

    BLACKLIST = {
        "gpu_id", "use_gpu", "data_path", "checkpoints",
        "is_training", "root_path", "num_workers",
        "hyper_parameters", "sweep",
    }

    def expand_hp(self, hp_yaml: Path, output_dir: Path) -> list[Path]:
        config = self.load_yaml(hp_yaml)
        filtered = {k: v for k, v in config.items() if k not in self.BLACKLIST}
        hp_keys = set(config.get("hyper_parameters", []))

        sweep, fixed = {}, {}
        for k, v in filtered.items():
            if k in hp_keys and isinstance(v, list) and v:
                sweep[k] = v
            else:
                fixed[k] = v

        if not sweep:
            path = output_dir / "combo_0.yaml"
            self.save_yaml(fixed, path)
            return [path]

        paths = []
        for i, combo in enumerate(self.grid_expand(sweep)):
            path = output_dir / f"combo_{i}.yaml"
            self.save_yaml({**fixed, **combo}, path)
            paths.append(path)
        return paths

    def parse_output(self, stdout: str) -> dict[str, float]:
        metrics, in_test = {}, False
        for line in stdout.splitlines():
            lower = line.strip().lower()
            if "testing" in lower or "test shape" in lower:
                in_test = True
            if in_test:
                found = {m.group(1).lower(): float(m.group(2))
                         for m in _METRIC_RE.finditer(line)}
                if found:
                    metrics = found
        return metrics
