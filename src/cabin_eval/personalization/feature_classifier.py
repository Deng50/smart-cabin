"""功能分类器模块."""
from cabin_eval.schemas import NeedTier


class FeatureClassifier:
    """功能分类器."""

    def __init__(self, config: dict):
        self._strong_multiplier = config.get("strong_multiplier", 1.10)
        self._high_multiplier = config.get("high_multiplier", 1.05)
        self._remove_low_need = config.get("remove_low_need", False)
        self._installation_rate_threshold = config.get("installation_rate_threshold", 0.30)

    def get_multiplier(self, need_tier: NeedTier) -> float:
        """根据需求等级获取权重倍率."""
        if need_tier == NeedTier.STRONG:
            return self._strong_multiplier
        elif need_tier == NeedTier.HIGH:
            return self._high_multiplier
        elif need_tier == NeedTier.NORMAL:
            return 1.0
        else:
            return 0.0 if self._remove_low_need else 1.0

    def should_remove(self, need_tier: NeedTier, is_required: bool) -> bool:
        """判断是否应移除功能."""
        if is_required:
            return False
        return need_tier == NeedTier.LOW and self._remove_low_need

    def should_promote_to_required(
        self,
        installation_rate: float | None,
        need_tier: NeedTier,
    ) -> bool:
        """判断是否应将功能提升为必测项."""
        if installation_rate is None:
            return False
        return installation_rate >= self._installation_rate_threshold and need_tier != NeedTier.LOW
