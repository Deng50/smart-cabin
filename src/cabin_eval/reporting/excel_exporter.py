"""Excel导出模块."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

from cabin_eval.schemas import (
    IndicatorTree,
    QuestionnaireMeta,
    SegmentationDecision,
    UserProfile,
    WeightChangeLog,
)


class ExcelExporter:
    """Excel报告导出器."""

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
        personalized_weights: dict[str, list[WeightChangeLog]],
    ) -> dict[str, str]:
        """导出完整运行报告."""
        outputs = {}

        outputs["segmentation_metrics"] = self._export_segmentation_metrics(
            seg_decision
        )
        outputs["user_profiles"] = self._export_user_profiles(profiles)
        outputs["base_weights"] = self._export_base_weights(
            base_weights, indicator_tree
        )
        outputs["personalized_systems"] = self._export_personalized_systems(
            profiles, personalized_weights
        )
        outputs["cleaning_report"] = self._export_cleaning_report(cleaning_meta)

        return outputs

    def _export_segmentation_metrics(
        self, seg_decision: SegmentationDecision
    ) -> str:
        """导出分群指标."""
        rows = []
        for metric in seg_decision.candidate_metrics:
            rows.append({
                "k": metric.k,
                "sse": metric.sse,
                "silhouette_score": metric.silhouette_score,
                "calinski_harabasz_score": metric.calinski_harabasz_score,
                "min_cluster_ratio": metric.min_cluster_ratio,
                "selected": metric.k == seg_decision.selected_k,
            })

        df = pd.DataFrame(rows)
        filepath = self._output_dir / "segmentation_metrics.xlsx"
        df.to_excel(filepath, index=False, sheet_name="SegMetrics")
        return str(filepath)

    def _export_user_profiles(self, profiles: list[UserProfile]) -> str:
        """导出用户画像."""
        rows = []
        for profile in profiles:
            rows.append({
                "cluster_id": profile.cluster_id,
                "sample_count": profile.sample_count,
                "proportion": profile.proportion,
                "top_functions": ", ".join(profile.top_differentiated_functions[:5]),
            })

        df = pd.DataFrame(rows)
        filepath = self._output_dir / "user_profiles.xlsx"
        df.to_excel(filepath, index=False, sheet_name="Profiles")
        return str(filepath)

    def _export_base_weights(
        self,
        base_weights: dict[str, float],
        indicator_tree: IndicatorTree,
    ) -> str:
        """导出基础权重."""
        rows = []
        for node_id, weight in base_weights.items():
            node = indicator_tree.nodes.get(node_id)
            if node:
                rows.append({
                    "indicator_id": node_id,
                    "feature_name": node.feature_name,
                    "level_1": node.level_1,
                    "level_2": node.level_2,
                    "local_weight": weight,
                    "global_weight": node.global_weight,
                    "category": node.category,
                })

        df = pd.DataFrame(rows)
        filepath = self._output_dir / "base_weights.xlsx"
        df.to_excel(filepath, index=False, sheet_name="BaseWeights")
        return str(filepath)

    def _export_personalized_systems(
        self,
        profiles: list[UserProfile],
        personalized_weights: dict[str, list[WeightChangeLog]],
    ) -> str:
        """导出个性化体系."""
        filepath = self._output_dir / "personalized_systems.xlsx"
        with pd.ExcelWriter(filepath) as writer:
            for profile in profiles:
                sheet_name = f"Cluster_{profile.cluster_id}"[:31]
                logs = personalized_weights.get(profile.cluster_id, [])
                rows = [
                    {
                        "indicator_id": log.indicator_id,
                        "base_weight": log.base_weight,
                        "need_tier": log.need_tier.value if log.need_tier else None,
                        "multiplier": log.multiplier,
                        "final_local_weight": log.final_local_weight,
                        "final_global_weight": log.final_global_weight,
                        "change_reason": log.change_reason,
                    }
                    for log in logs
                ]
                df = pd.DataFrame(rows)
                df.to_excel(writer, index=False, sheet_name=sheet_name)

        return str(filepath)

    def _export_cleaning_report(
        self, cleaning_meta: QuestionnaireMeta
    ) -> str:
        """导出清洗报告."""
        summary_rows = [
            {"metric": "total_rows", "value": cleaning_meta.total_rows},
            {"metric": "valid_rows", "value": cleaning_meta.valid_rows},
            {"metric": "removed_rows", "value": cleaning_meta.removed_rows},
            {"metric": "rating_direction", "value": cleaning_meta.rating_direction},
            {"metric": "rating_min", "value": cleaning_meta.rating_min},
            {"metric": "rating_max", "value": cleaning_meta.rating_max},
        ]

        reason_rows = [
            {"reason": k, "count": v}
            for k, v in cleaning_meta.removed_reasons.items()
        ]

        filepath = self._output_dir / "cleaning_report.xlsx"
        with pd.ExcelWriter(filepath) as writer:
            pd.DataFrame(summary_rows).to_excel(
                writer, index=False, sheet_name="Summary"
            )
            pd.DataFrame(reason_rows).to_excel(
                writer, index=False, sheet_name="RemovedReasons"
            )

        return str(filepath)
