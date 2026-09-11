"""测评模型定义."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from cabin_eval.schemas import (
    IndicatorTree,
    RunRecord,
    RunStatus,
    SegmentationDecision,
    UserProfile,
    WeightChangeLog,
)


class EvaluationModel:
    """测评模型."""

    def __init__(
        self,
        run_id: str,
        config: dict[str, Any],
        indicator_tree: IndicatorTree,
        segmentation_decision: SegmentationDecision,
        profiles: list[UserProfile],
        base_weights: list[WeightChangeLog],
    ):
        self.run_id = run_id
        self.created_at = datetime.now()
        self.config = config
        self.indicator_tree = indicator_tree
        self.segmentation_decision = segmentation_decision
        self.profiles = profiles
        self.base_weights = base_weights
        self.personalized_weights: dict[str, list[WeightChangeLog]] = {}
        self.is_frozen = False

    def add_personalized_weights(self, cluster_id: str, weights: list[WeightChangeLog]) -> None:
        """添加个性化权重."""
        self.personalized_weights[cluster_id] = weights

    def freeze(self) -> None:
        """冻结模型."""
        self.is_frozen = True

    def calculate_score(self, indicator_scores: dict[str, float]) -> float:
        """计算综合得分."""
        if self.is_frozen:
            return self._compute_score(indicator_scores, self.base_weights)
        raise ValueError("Model must be frozen before scoring")

    def _compute_score(
        self, indicator_scores: dict[str, float], weights: list[WeightChangeLog]
    ) -> float:
        """计算加权得分."""
        total_score = 0.0
        for w in weights:
            score = indicator_scores.get(w.indicator_id, 0.0)
            total_score += score * w.final_global_weight
        return total_score


class ModelVersion:
    """模型版本管理."""

    def __init__(self):
        self._models: dict[str, EvaluationModel] = {}

    def register(self, model: EvaluationModel) -> None:
        """注册模型."""
        self._models[model.run_id] = model

    def get(self, run_id: str) -> EvaluationModel | None:
        """获取模型."""
        return self._models.get(run_id)

    def list_models(self) -> list[str]:
        """列出所有模型ID."""
        return list(self._models.keys())
