"""体系生成服务."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cabin_eval.config import Config
from cabin_eval.domain.evaluation_model import EvaluationModel
from cabin_eval.domain.indicator_tree import IndicatorTreeBuilder, IndicatorTreeManager
from cabin_eval.domain.user_profile import UserProfileGenerator
from cabin_eval.ingestion.experts import ExpertIngester
from cabin_eval.ingestion.indicators import IndicatorIngester
from cabin_eval.ingestion.questionnaire import QuestionnaireIngester
from cabin_eval.ingestion.vehicles import VehicleIngester
from cabin_eval.personalization.profile_adapter import ProfileAdapter
from cabin_eval.reporting.charts import ChartGenerator
from cabin_eval.reporting.excel_exporter import ExcelExporter
from cabin_eval.reporting.json_exporter import JsonExporter
from cabin_eval.segmentation.kmeans import KMeansSegmenter
from cabin_eval.segmentation.preprocessing import QuestionnairePreprocessor
from cabin_eval.schemas import RunRecord, RunStatus
from cabin_eval.weighting.ahp import AHPWeightCalculator
from cabin_eval.weighting.entropy import EntropyWeightCalculator
from cabin_eval.weighting.fusion import WeightFusion
from cabin_eval.weighting.history_adjustment import HistoryAdjustmentCalculator


class GenerationService:
    """体系生成服务."""

    def __init__(self, config: Config):
        self._cfg = config
        self._run_record: RunRecord | None = None
        self._run_dir: Path | None = None

    def run(
        self,
        questionnaire_path: str | Path,
        indicators_path: str | Path,
        expert_judgments_path: str | Path | None = None,
        expert_authority_path: str | Path | None = None,
        vehicle_scores_path: str | Path | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行体系生成流程."""
        run_id = str(uuid.uuid4())[:8]
        self._run_dir = self._cfg.runs_dir / f"run_{run_id}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._run_record = RunRecord(
            run_id=run_id,
            status=RunStatus.PENDING,
            created_at=datetime.now(),
            config_snapshot=self._cfg._cfg.copy(),
            input_hashes={},
        )

        try:
            self._run_record.status = RunStatus.RUNNING
            self._run_record.started_at = datetime.now()

            q_ingester = QuestionnaireIngester(self._cfg)
            questionnaire_df = q_ingester.load(questionnaire_path)
            q_errors = q_ingester.validate(questionnaire_df)
            if q_errors:
                raise ValueError(f"Questionnaire validation errors: {q_errors}")

            clean_df, cleaning_meta = q_ingester.clean(questionnaire_df)

            self._run_record.input_hashes["questionnaire"] = self._cfg.file_hash(
                Path(questionnaire_path)
            )

            seg_cfg = self._cfg.segmentation_config
            preprocessor = QuestionnairePreprocessor(seg_cfg)
            X_scaled = preprocessor.prepare_for_clustering(clean_df)

            segmenter = KMeansSegmenter(seg_cfg)
            all_metrics = segmenter.find_optimal_k(X_scaled.values)

            if seg_cfg.get("mode") == "manual":
                seg_decision = segmenter.select_k_manual(
                    all_metrics, seg_cfg.get("manual_k", 5)
                )
            else:
                seg_decision = segmenter.select_k_auto(all_metrics)

            cluster_labels = segmenter.cluster(X_scaled.values, seg_decision.selected_k)

            profile_generator = UserProfileGenerator(
                clean_df, cluster_labels, seg_cfg, self._cfg.need_tiers
            )
            profiles = profile_generator.generate_profiles()

            ind_ingester = IndicatorIngester()
            ind_df = ind_ingester.load(indicators_path)
            ind_errors = ind_ingester.validate(ind_df)
            if ind_errors:
                raise ValueError(f"Indicator validation errors: {ind_errors}")

            self._run_record.input_hashes["indicators"] = self._cfg.file_hash(
                Path(indicators_path)
            )

            tree_builder = IndicatorTreeBuilder()
            indicator_tree = tree_builder.build_from_dataframe(ind_df)

            base_weights: dict[str, float] = {}

            if expert_judgments_path:
                exp_ingester = ExpertIngester()
                judgments_df = exp_ingester.load_judgments(expert_judgments_path)
                self._run_record.input_hashes["expert_judgments"] = self._cfg.file_hash(
                    Path(expert_judgments_path)
                )

                authority_df = None
                if expert_authority_path:
                    authority_df = exp_ingester.load_authority(expert_authority_path)
                    self._run_record.input_hashes["expert_authority"] = self._cfg.file_hash(
                        Path(expert_authority_path)
                    )

                ahp_calc = AHPWeightCalculator(self._cfg.ahp_cr_threshold)
                indicator_ids = list(indicator_tree.nodes.keys())
                ahp_results = ahp_calc.calculate_weights(judgments_df, indicator_ids)
                base_weights = ahp_calc.aggregate_expert_weights(ahp_results, authority_df)

            if self._cfg.entropy_enabled:
                function_cols = [c for c in clean_df.columns if c.startswith("function_")]
                entropy_calc = EntropyWeightCalculator()
                entropy_weights = entropy_calc.calculate(clean_df, function_cols)

                fusion = WeightFusion(
                    mode=self._cfg.weight_fusion_mode,
                    alpha=self._cfg.get("weight_fusion", {}).get("alpha", 0.7),
                )
                base_weights = fusion.fuse(base_weights, entropy_weights)
            else:
                fusion = WeightFusion(mode="expert_only")
                base_weights = fusion.fuse(base_weights)

            if self._cfg.history_adjustment_enabled and vehicle_scores_path:
                veh_ingester = VehicleIngester()
                vehicle_df = veh_ingester.load(vehicle_scores_path)

                hist_calc = HistoryAdjustmentCalculator(
                    self._cfg.get("history_adjustment", {})
                )
                score_cols = [c for c in vehicle_df.columns if c.startswith("indicator_")]
                score_distributions = {
                    col: hist_calc.analyze_distribution(vehicle_df[col])
                    for col in score_cols
                }
                base_weights = hist_calc.adjust_weights(base_weights, score_distributions)

            for node_id, node in indicator_tree.nodes.items():
                if node_id in base_weights:
                    node.local_weight = base_weights[node_id]

            tree_manager = IndicatorTreeManager(indicator_tree)
            tree_manager.normalize_sibling_weights("")  # normalize all
            tree_manager.calculate_global_weights()

            profile_adapter = ProfileAdapter(
                indicator_tree, self._cfg.personalization_config
            )

            personalized_weights: dict[str, list[WeightChangeLog]] = {}
            for profile in profiles:
                change_logs = profile_adapter.adapt(profile, base_weights)
                personalized_weights[profile.cluster_id] = change_logs

            model = EvaluationModel(
                run_id=run_id,
                config=self._cfg._cfg.copy(),
                indicator_tree=indicator_tree,
                segmentation_decision=seg_decision,
                profiles=profiles,
                base_weights=[],  # TODO
            )

            chart_gen = ChartGenerator(self._run_dir)
            chart_gen.generate_elbow_chart(all_metrics)
            chart_gen.generate_cluster_sizes(seg_decision.selected_k, cluster_labels)
            chart_gen.generate_profile_heatmap(profiles)

            excel_exporter = ExcelExporter(self._run_dir)
            excel_exporter.export_run(
                run_id=run_id,
                cleaning_meta=cleaning_meta,
                seg_decision=seg_decision,
                profiles=profiles,
                indicator_tree=indicator_tree,
                base_weights=base_weights,
                personalized_weights=personalized_weights,
            )

            json_exporter = JsonExporter(self._run_dir)
            json_exporter.export_run(
                run_id=run_id,
                cleaning_meta=cleaning_meta,
                seg_decision=seg_decision,
                profiles=profiles,
                indicator_tree=indicator_tree,
                base_weights=base_weights,
            )

            self._run_record.status = RunStatus.SUCCEEDED
            self._run_record.completed_at = datetime.now()
            self._run_record.output_artifacts = {
                "manifest": f"run_manifest.json",
                "cleaning_report": "cleaning_report.json",
                "segmentation_metrics": "segmentation_metrics.xlsx",
                "user_profiles": "user_profiles.xlsx",
                "base_weights": "base_weights.xlsx",
                "personalized_systems": "personalized_systems.xlsx",
                "model": "model.json",
            }

            return {
                "run_id": run_id,
                "status": "succeeded",
                "run_dir": str(self._run_dir),
                "artifacts": self._run_record.output_artifacts,
            }

        except Exception as e:
            self._run_record.status = RunStatus.FAILED
            self._run_record.error_message = str(e)
            self._run_record.completed_at = datetime.now()
            raise


# Alias for import compatibility
from cabin_eval.schemas import WeightChangeLog
