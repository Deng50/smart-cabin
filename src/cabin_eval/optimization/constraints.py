"""约束处理模块."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class ConstraintHandler:
    """约束处理器 - 处理权重约束."""

    def __init__(
        self,
        sibling_groups: list[list[int]],
        base_weights: np.ndarray,
        epsilon: float = 0.10,
    ):
        self._sibling_groups = sibling_groups
        self._base_weights = base_weights
        self._epsilon = epsilon

    def project(self, weights: np.ndarray) -> np.ndarray:
        """将权重投影到可行域 (带上下界的单纯形投影)."""
        weights = np.clip(weights, 0, 1)

        for group in self._sibling_groups:
            group_weights = weights[group]
            total = group_weights.sum()
            if total > 0:
                weights[group] = group_weights / total
            else:
                weights[group] = np.ones(len(group)) / len(group)

        lower_bound = np.maximum(0, self._base_weights - self._epsilon)
        upper_bound = np.minimum(1, self._base_weights + self._epsilon)

        weights = np.clip(weights, lower_bound, upper_bound)

        for group in self._sibling_groups:
            group_weights = weights[group]
            total = group_weights.sum()
            if abs(total - 1.0) > 1e-8 and total > 0:
                weights[group] = group_weights / total

        return weights

    def validate(self, weights: np.ndarray) -> list[str]:
        """验证权重是否满足约束."""
        violations = []

        for i, w in enumerate(weights):
            if w < 0:
                violations.append(f"Weight {i} is negative: {w}")
            if w > 1:
                violations.append(f"Weight {i} exceeds 1: {w}")

            lower_bound = max(0, self._base_weights[i] - self._epsilon)
            upper_bound = min(1, self._base_weights[i] + self._epsilon)
            if w < lower_bound - 1e-8 or w > upper_bound + 1e-8:
                violations.append(
                    f"Weight {i} = {w} outside bounds [{lower_bound}, {upper_bound}]"
                )

        for group in self._sibling_groups:
            group_weights = weights[group]
            total = group_weights.sum()
            if abs(total - 1.0) > 1e-6:
                violations.append(
                    f"Sibling group {group} sum = {total}, expected 1.0"
                )

        return violations
