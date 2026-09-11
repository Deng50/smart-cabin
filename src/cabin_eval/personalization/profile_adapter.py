"""用户画像适配器."""
from __future__ import annotations

import copy
from typing import Any

from cabin_eval.domain.indicator_tree import IndicatorTreeManager
from cabin_eval.personalization.feature_classifier import FeatureClassifier
from cabin_eval.schemas import (
    IndicatorCategory,
    IndicatorNode,
    IndicatorTree,
    NeedTier,
    UserProfile,
    WeightChangeLog,
)


class ProfileAdapter:
    """用户画像适配器 - 生成差异化指标体系."""

    def __init__(self, base_tree: IndicatorTree, config: dict[str, Any]):
        self._base_tree = base_tree
        self._config = config
        self._classifier = FeatureClassifier(config)

    def adapt(
        self, profile: UserProfile, base_weights: dict[str, float]
    ) -> list[WeightChangeLog]:
        """为用户画像生成差异化权重."""
        adapted_tree = copy.deepcopy(self._base_tree)
        tree_manager = IndicatorTreeManager(adapted_tree)
        leaves = tree_manager.get_leaves()

        change_logs: list[WeightChangeLog] = []

        for leaf in leaves:
            indicator_id = leaf.indicator_id
            base_weight = base_weights.get(indicator_id, 0.0)

            need_tier = profile.need_tier.get(f"function_{indicator_id}", NeedTier.NORMAL)

            multiplier = self._classifier.get_multiplier(need_tier)
            final_local = base_weight * multiplier

            is_required = leaf.category == IndicatorCategory.REQUIRED
            if self._classifier.should_remove(need_tier, is_required):
                final_local = 0.0
                reason = "low_need_removed"
            elif need_tier == NeedTier.STRONG:
                reason = "strong_need_multiplied"
            elif need_tier == NeedTier.HIGH:
                reason = "high_need_multiplied"
            else:
                reason = "base_weight"

            node = adapted_tree.nodes.get(indicator_id)
            if node:
                node.local_weight = final_local

            change_logs.append(WeightChangeLog(
                indicator_id=indicator_id,
                base_weight=base_weight,
                need_tier=need_tier,
                multiplier=multiplier,
                final_local_weight=final_local,
                final_global_weight=0.0,
                change_reason=reason,
            ))

        tree_manager.calculate_global_weights()

        for log in change_logs:
            node = adapted_tree.nodes.get(log.indicator_id)
            if node:
                log.final_global_weight = node.global_weight

        return change_logs
