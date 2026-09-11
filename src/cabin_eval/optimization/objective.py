"""目标函数模块."""
from __future__ import annotations

import numpy as np


class ObjectiveFunction:
    """目标函数计算器."""

    def __init__(
        self,
        lambda_l2: float = 0.01,
        lambda_rank: float = 0.01,
    ):
        self._lambda_l2 = lambda_l2
        self._lambda_rank = lambda_rank

    def calculate(
        self,
        weights: np.ndarray,
        base_weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        indicator_ranks: np.ndarray | None = None,
    ) -> float:
        """计算综合损失."""
        mae = self._calculate_mae(weights, X, y)
        l2_penalty = self._lambda_l2 * np.sum((weights - base_weights) ** 2)

        loss = mae + l2_penalty

        if indicator_ranks is not None and self._lambda_rank > 0:
            rank_penalty = self._calculate_rank_loss(weights, indicator_ranks)
            loss += self._lambda_rank * rank_penalty

        return loss

    def _calculate_mae(
        self, weights: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        """计算MAE."""
        predictions = X @ weights
        return float(np.mean(np.abs(predictions - y)))

    def _calculate_rank_loss(
        self, weights: np.ndarray, ranks: np.ndarray
    ) -> float:
        """计算排名损失."""
        weight_ranks = np.argsort(np.argsort(-weights))
        return float(np.sum(np.abs(weight_ranks - ranks)))
