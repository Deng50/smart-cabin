"""AHP层次分析法权重计算."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from cabin_eval.schemas import AhpResult


class AHPWeightCalculator:
    """AHP权重计算器."""

    RI_VALUES = {
        1: 0.0,
        2: 0.0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }

    def __init__(self, cr_threshold: float = 0.10):
        self._cr_threshold = cr_threshold

    def calculate_weights(
        self,
        judgments_df: pd.DataFrame,
        indicator_ids: list[str],
    ) -> list[AhpResult]:
        """计算AHP权重."""
        results = []
        expert_groups = judgments_df.groupby("expert_id")

        for expert_id, expert_df in expert_groups:
            parent_groups = expert_df.groupby("parent_indicator_id")

            for parent_id, parent_df in parent_groups:
                result = self._calculate_single_matrix(
                    expert_id, parent_id, parent_df, indicator_ids
                )
                results.append(result)

        return results

    def _calculate_single_matrix(
        self,
        expert_id: str,
        parent_id: str,
        parent_df: pd.DataFrame,
        all_indicator_ids: list[str],
    ) -> AhpResult:
        """计算单个判断矩阵的权重."""
        indicator_ids = sorted(set(parent_df["left_indicator_id"].tolist() + parent_df["right_indicator_id"].tolist()))
        n = len(indicator_ids)

        matrix = np.eye(n)
        id_to_idx = {pid: i for i, pid in enumerate(indicator_ids)}

        for _, row in parent_df.iterrows():
            left_idx = id_to_idx.get(row["left_indicator_id"])
            right_idx = id_to_idx.get(row["right_indicator_id"])
            if left_idx is not None and right_idx is not None:
                ratio = row["ratio"]
                matrix[left_idx, right_idx] = ratio
                matrix[right_idx, left_idx] = 1.0 / ratio

        for i in range(n):
            for j in range(n):
                if i != j and matrix[i, j] == 0:
                    matrix[i, j] = matrix[j, i] ** -1

        weights = self._compute_weights(matrix)
        lambda_max = self._compute_lambda_max(matrix, weights)
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
        ri = self.RI_VALUES.get(n, 1.49)
        cr = ci / ri if ri > 0 else 0.0
        is_consistent = cr < self._cr_threshold

        local_weights = {pid: float(weights[i]) for i, pid in enumerate(indicator_ids)}

        rejected_items = []
        if not is_consistent:
            rejected_items = self._find_inconsistent_pairs(parent_df, matrix, weights)

        return AhpResult(
            expert_id=expert_id,
            parent_indicator_id=parent_id,
            local_weights=local_weights,
            lambda_max=float(lambda_max),
            ci=float(ci),
            cr=float(cr),
            is_consistent=is_consistent,
            rejected_items=rejected_items,
        )

    def _compute_weights(self, matrix: np.ndarray) -> np.ndarray:
        """使用几何平均法计算权重."""
        n = matrix.shape[0]
        product = np.prod(matrix, axis=1)
        weights = product ** (1.0 / n)
        return weights / weights.sum()

    def _compute_lambda_max(self, matrix: np.ndarray, weights: np.ndarray) -> float:
        """计算最大特征值."""
        n = matrix.shape[0]
        weighted_sum = np.sum(matrix * weights, axis=1)
        return float(np.sum(weighted_sum / weights) / n)

    def _find_inconsistent_pairs(
        self,
        df: pd.DataFrame,
        matrix: np.ndarray,
        weights: np.ndarray,
    ) -> list[str]:
        """找出不一致的判断对."""
        inconsistent = []
        threshold = self._cr_threshold * 0.5

        for _, row in df.iterrows():
            left_idx = list(df["left_indicator_id"]).index(row["left_indicator_id"])
            right_idx = list(df["right_indicator_id"]).index(row["right_indicator_id"])

            expected_ratio = weights[left_idx] / weights[right_idx]
            actual_ratio = row["ratio"]

            if abs(np.log(actual_ratio / expected_ratio)) > threshold:
                inconsistent.append(f"{row['left_indicator_id']}|{row['right_indicator_id']}")

        return inconsistent[:5]

    def aggregate_expert_weights(
        self,
        ahp_results: list[AhpResult],
        authority_df: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """聚合专家权重."""
        authority_map = {}
        if authority_df is not None and "expert_id" in authority_df.columns:
            for _, row in authority_df.iterrows():
                authority_map[str(row["expert_id"])] = float(row.get("authority_score", 1.0))

        parent_weights: dict[str, dict[str, float]] = {}
        parent_counts: dict[str, int] = {}

        for result in ahp_results:
            if not result.is_consistent:
                continue

            parent_id = result.parent_indicator_id
            if parent_id not in parent_weights:
                parent_weights[parent_id] = {}
                parent_counts[parent_id] = 0

            weight_sum = sum(parent_weights[parent_id].values())
            if weight_sum == 0:
                parent_weights[parent_id] = result.local_weights.copy()
                parent_counts[parent_id] = 1
            else:
                for pid, w in result.local_weights.items():
                    parent_weights[parent_id][pid] = (
                        parent_weights[parent_id].get(pid, 0) + w
                    )
                parent_counts[parent_id] += 1

        final_weights = {}
        for parent_id, weights in parent_weights.items():
            count = parent_counts[parent_id]
            if count > 0:
                for pid in weights:
                    weights[pid] /= count
                final_weights.update(weights)

        return final_weights
