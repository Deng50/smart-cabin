"""评分服务."""
from __future__ import annotations

from typing import Any

from cabin_eval.domain.evaluation_model import EvaluationModel
from cabin_eval.schemas import IndicatorTree


class ScoringService:
    """评分服务."""

    def __init__(self, model: EvaluationModel):
        self._model = model

    def score(self, indicator_scores: dict[str, float]) -> float:
        """计算综合得分."""
        if not self._model.is_frozen:
            raise ValueError("Model must be frozen before scoring")

        return self._model.calculate_score(indicator_scores)

    def score_with_breakdown(
        self, indicator_scores: dict[str, float]
    ) -> dict[str, Any]:
        """计算综合得分并返回分解."""
        total_score = self.score(indicator_scores)

        breakdown = {}
        for weight_log in self._model.base_weights:
            indicator_id = weight_log.indicator_id
            score = indicator_scores.get(indicator_id, 0.0)
            weighted_score = score * weight_log.final_global_weight
            breakdown[indicator_id] = {
                "raw_score": score,
                "weight": weight_log.final_global_weight,
                "weighted_score": weighted_score,
            }

        return {
            "total_score": total_score,
            "breakdown": breakdown,
        }
