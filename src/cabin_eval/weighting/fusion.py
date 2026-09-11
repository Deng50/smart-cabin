"""多源权重融合模块."""
from __future__ import annotations

from typing import Any

from cabin_eval.schemas import IndicatorTree


class WeightFusion:
    """多源权重融合器."""

    def __init__(self, mode: str = "expert_only", alpha: float = 0.7):
        self._mode = mode
        self._alpha = alpha

    def fuse(
        self,
        expert_weights: dict[str, float],
        entropy_weights: dict[str, float] | None = None,
        history_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """融合多源权重."""
        if self._mode == "expert_only":
            return expert_weights.copy()

        if self._mode == "linear_fusion" and entropy_weights:
            return self._linear_fusion(expert_weights, entropy_weights)

        return expert_weights.copy()

    def _linear_fusion(
        self,
        expert_weights: dict[str, float],
        entropy_weights: dict[str, float],
    ) -> dict[str, float]:
        """线性融合专家权重和熵权."""
        fused = {}
        all_keys = set(expert_weights.keys()) | set(entropy_weights.keys())

        for key in all_keys:
            w_exp = expert_weights.get(key, 0.0)
            w_ent = entropy_weights.get(key, 0.0)
            fused[key] = self._alpha * w_exp + (1 - self._alpha) * w_ent

        total = sum(fused.values())
        if total > 0:
            fused = {k: v / total for k, v in fused.items()}

        return fused
