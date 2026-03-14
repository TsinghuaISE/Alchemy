"""Recsys/MMRec 插件."""

import re
from pathlib import Path

from research_environment.plugin import TaskPlugin


class MMRecPlugin(TaskPlugin):
    """MMRec 多模态推荐插件.

    超参展开: 黑名单过滤 + hyper_parameters 白名单网格展开.
    指标解析: 从日志中提取 recall@k, ndcg@k 等 test 指标.
    """

    # 系统配置字段, 不允许被 Agent 修改
    BLACKLIST = {
        # dataset.yaml 字段
        "USER_ID_FIELD", "ITEM_ID_FIELD", "TIME_FIELD", "RATING_FIELD",
        "inter_file_name", "vision_feature_file", "text_feature_file",
        "user_graph_dict_file", "field_separator", "filter_out_cod_start_users",
        # overall.yaml 字段
        "gpu_id", "use_gpu", "data_path", "checkpoint_dir",
        "save_recommended_topk", "recommend_topk", "NEG_PREFIX",
        "use_neg_sampling", "use_full_sampling", "is_multimodal_model",
        "req_training", "use_raw_features", "max_txt_len", "max_img_size",
        "inter_splitting_label",
        # 元数据
        "hyper_parameters", "sweep",
    }

    # 指标正则: recall@10: 0.1234 或 'recall@10': 0.1234
    _METRIC_RE = re.compile(r"['\"]?(\w+@\d+)['\"]?\s*:\s*([\d.]+)")

    # ---- 超参展开 ----

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

    # ---- 指标解析 ----

    def parse_output(self, stdout: str) -> dict[str, float]:
        metrics, await_test = {}, False
        for line in stdout.splitlines():
            lower = line.strip().lower()
            if "valid result" in lower or "best valid" in lower:
                await_test = False
                continue
            if "test result" in lower or "best test" in lower:
                await_test = True
            if await_test or "test:" in lower or "test result" in lower:
                found = {m.group(1).lower(): float(m.group(2))
                         for m in self._METRIC_RE.finditer(line)}
                if found:
                    metrics = found
                    await_test = False
        return metrics
