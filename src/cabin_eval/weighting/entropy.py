"""熵权法权重计算."""
from __future__ import annotations

import numpy as np
import pandas as pd


class EntropyWeightCalculator:
    """熵权法计算客观权重."""

    def __init__(self):
        pass

    def calculate(self, df: pd.DataFrame, indicator_cols: list[str]) -> dict[str, float]:
        """计算熵权，常数列权重为0."""
        X = df[indicator_cols].values
        n_samples, n_cols = X.shape

        constant_mask = (X.min(axis=0) == X.max(axis=0))

        X_normalized = self._min_max_normalize(X)

        p = self._calculate_probability(X_normalized)

        entropy = self._calculate_entropy(p)

        entropy[constant_mask] = 1.0

        diversity_coefficient = 1 - entropy

        weights = self._normalize_weights(diversity_coefficient)

        result = {}
        for i, col in enumerate(indicator_cols):
            if constant_mask[i]:
                result[col] = 0.0
            else:
                result[col] = float(weights[i])

        return result

    def _min_max_normalize(self, X: np.ndarray) -> np.ndarray:
        """Min-Max标准化."""
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0
        return (X - X_min) / X_range

    def _calculate_probability(self, X_normalized: np.ndarray) -> np.ndarray:
        """计算概率矩阵."""
        col_sums = X_normalized.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        p = X_normalized / col_sums
        p[p == 0] = 1e-10
        return p

    def _calculate_entropy(self, p: np.ndarray) -> np.ndarray:
        """计算信息熵."""
        n = p.shape[0]
        entropy = -np.sum(p * np.log(p), axis=0) / np.log(n)
        return entropy

    def _normalize_weights(self, diversity: np.ndarray) -> np.ndarray:
        """归一化差异系数为权重."""
        total = diversity.sum()
        if total == 0:
            return np.ones_like(diversity) / len(diversity)
        return diversity / total
