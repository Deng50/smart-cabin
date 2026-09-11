"""历史数据调整模块."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Any


class HistoryAdjustmentCalculator:
    """基于历史数据分布调整权重."""

    def __init__(self, config: dict[str, Any]):
        self._cfg = config
        self._factors = config.get("factors", {})

    def analyze_distribution(self, scores: pd.Series) -> str:
        """分析得分分布类型."""
        valid_scores = scores.dropna()
        if len(valid_scores) < 10:
            return "insufficient_data"

        _, p_value = stats.normaltest(valid_scores)

        if p_value > 0.05:
            if valid_scores.std() < valid_scores.mean() * 0.1:
                return "normal"
            return "normal"

        if self._is_bimodal(valid_scores):
            return "bimodal"

        mean = valid_scores.mean()
        if valid_scores.median() > mean:
            return "high"
        elif valid_scores.median() < mean:
            return "low"

        return "normal"

    def _is_bimodal(self, scores: pd.Series, threshold: float = 0.3) -> bool:
        """判断是否为双峰分布."""
        if len(scores) < 20:
            return False

        hist, _ = np.histogram(scores, bins=min(20, len(scores) // 5))
        peak_ratio = hist.max() / hist.sum()
        second_peak_ratio = sorted(hist, reverse=True)[1] / hist.sum() if len(hist) > 1 else 0

        return second_peak_ratio > threshold

    def get_adjustment_factor(self, distribution_type: str) -> float:
        """获取调整因子."""
        return self._factors.get(distribution_type, 1.0)

    def adjust_weights(
        self,
        weights: dict[str, float],
        score_distributions: dict[str, str],
    ) -> dict[str, float]:
        """调整权重."""
        adjusted = {}
        for indicator_id, weight in weights.items():
            dist_type = score_distributions.get(indicator_id, "normal")
            factor = self.get_adjustment_factor(dist_type)
            adjusted[indicator_id] = weight * factor

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted
