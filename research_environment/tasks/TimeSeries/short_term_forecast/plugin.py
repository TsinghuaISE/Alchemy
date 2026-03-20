"""TimeSeries/short_term_forecast 插件."""

import ast
import re
from pathlib import Path

from research_environment.plugin import TaskPlugin

# smape: {'Average': 14.591, 'Yearly': 13.564, ...}
_DICT_RE = re.compile(r"(smape|mape|mase|owa)\s*:\s*(\{[^}]+\})", re.IGNORECASE)


class ShortTermForecastPlugin(TaskPlugin):
    """短期预测插件 (M4 数据集).

    指标解析: 提取 smape/mape/mase/owa 字典格式, 取 Average 值.
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
                for m in _DICT_RE.finditer(line):
                    key, dict_str = m.groups()
                    try:
                        d = ast.literal_eval(dict_str)
                        if isinstance(d, dict) and "Average" in d:
                            metrics[key.lower()] = float(d["Average"])
                    except (ValueError, SyntaxError):
                        pass
        return metrics
