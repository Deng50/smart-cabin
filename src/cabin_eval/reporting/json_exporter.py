"""JSON导出模块."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cabin_eval.schemas import (
    IndicatorTree,
    QuestionnaireMeta,
    SegmentationDecision,
    UserProfile,
)


class JsonExporter:
    """JSON报告导出器."""

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export_run(
        self,
        run_id: str,
        cleaning_meta: QuestionnaireMeta,
        seg_decision: SegmentationDecision,
        profiles: list[UserProfile],
        indicator_tree: IndicatorTree,
        base_weights: dict[str, float],
    ) -> dict[str, str]:
        """导出JSON报告."""
        outputs = {}

        outputs["model"] = self._export_model(
            run_id, indicator_tree, base_weights
        )
        outputs["manifest"] = self._export_manifest(
            run_id, cleaning_meta, seg_decision, profiles, outputs
        )

        return outputs

    def _export_model(
        self,
        run_id: str,
        indicator_tree: IndicatorTree,
        base_weights: dict[str, float],
    ) -> str:
        """导出模型JSON."""
        model_data = {
            "run_id": run_id,
            "indicator_tree": {
                node_id: {
                    "indicator_id": node.indicator_id,
                    "level_1": node.level_1,
                    "level_2": node.level_2,
                    "level_3": node.level_3,
                    "level_4": node.level_4,
                    "feature_name": node.feature_name,
                    "category": node.category,
                    "local_weight": node.local_weight,
                    "global_weight": node.global_weight,
                    "score_max": node.score_max,
                    "enabled": node.enabled,
                }
                for node_id, node in indicator_tree.nodes.items()
            },
            "weights": base_weights,
        }

        filepath = self._output_dir / "model.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

        return str(filepath)

    def _export_manifest(
        self,
        run_id: str,
        cleaning_meta: QuestionnaireMeta,
        seg_decision: SegmentationDecision,
        profiles: list[UserProfile],
        artifacts: dict[str, str],
    ) -> str:
        """导出运行清单."""
        manifest = {
            "run_id": run_id,
            "cleaning": {
                "total_rows": cleaning_meta.total_rows,
                "valid_rows": cleaning_meta.valid_rows,
                "removed_rows": cleaning_meta.removed_rows,
                "removed_reasons": cleaning_meta.removed_reasons,
                "rating_direction": cleaning_meta.rating_direction,
            },
            "segmentation": {
                "selected_k": seg_decision.selected_k,
                "mode": seg_decision.mode,
                "override_reason": seg_decision.override_reason,
                "candidate_metrics": [
                    {
                        "k": m.k,
                        "sse": m.sse,
                        "silhouette_score": m.silhouette_score,
                        "calinski_harabasz_score": m.calinski_harabasz_score,
                        "cluster_sizes": m.cluster_sizes,
                    }
                    for m in seg_decision.candidate_metrics
                ],
            },
            "profiles": [
                {
                    "cluster_id": p.cluster_id,
                    "sample_count": p.sample_count,
                    "proportion": p.proportion,
                    "top_functions": p.top_differentiated_functions,
                    "need_tier_counts": {
                        k: v.value for k, v in p.need_tier.items()
                    },
                }
                for p in profiles
            ],
            "artifacts": artifacts,
        }

        filepath = self._output_dir / "run_manifest.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return str(filepath)
