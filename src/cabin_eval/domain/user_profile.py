"""用户画像生成."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from cabin_eval.schemas import (
    AgeGroup,
    CityTier,
    Education,
    FamilyStage,
    Gender,
    NeedTier,
    OwnershipStatus,
    PurchaseBudget,
    SegmentationDecision,
    SegmentationMetrics,
    UserProfile,
)


class UserProfileGenerator:
    """用户画像生成器."""

    def __init__(
        self,
        questionnaire_df: pd.DataFrame,
        cluster_labels: np.ndarray,
        segmentation_config: dict[str, Any],
        need_tiers_config: dict[str, float],
    ):
        self._df = questionnaire_df
        self._labels = cluster_labels
        self._seg_cfg = segmentation_config
        self._need_tiers = need_tiers_config
        self._function_cols = self._detect_function_columns()

    def _detect_function_columns(self) -> list[str]:
        """检测功能评分列."""
        return [c for c in self._df.columns if c.startswith("function_")]

    def generate_profiles(self) -> list[UserProfile]:
        """生成所有用户画像."""
        unique_clusters = np.unique(self._labels)
        total_samples = len(self._df)

        profiles = []
        for cluster_id in unique_clusters:
            mask = self._labels == cluster_id
            cluster_df = self._df[mask]

            profile = self._build_profile(cluster_id, cluster_df, total_samples)
            profiles.append(profile)

        return profiles

    def _build_profile(
        self, cluster_id: int | str, cluster_df: pd.DataFrame, total: int
    ) -> UserProfile:
        """构建单个用户画像."""
        sample_count = len(cluster_df)
        proportion = sample_count / total if total > 0 else 0

        need_tier = self._calculate_need_tiers(cluster_df)
        top_functions = self._get_top_differentiated_functions(cluster_df)

        return UserProfile(
            cluster_id=str(cluster_id),
            sample_count=sample_count,
            proportion=proportion,
            need_tier=need_tier,
            top_differentiated_functions=top_functions,
            gender_distribution=self._calc_gender_distribution(cluster_df),
            age_distribution=self._calc_age_distribution(cluster_df),
            budget_distribution=self._calc_budget_distribution(cluster_df),
            function_mean_scores=self._calc_function_means(cluster_df),
            function_median_scores=self._calc_function_medians(cluster_df),
            strong_need_ratio=self._calc_strong_need_ratio(cluster_df),
        )

    def _calculate_need_tiers(self, cluster_df: pd.DataFrame) -> dict[str, NeedTier]:
        """计算需求等级."""
        need_tiers: dict[str, NeedTier] = {}
        strong_thresh = self._need_tiers.get("strong", 0.90)
        high_thresh = self._need_tiers.get("high", 0.70)
        normal_thresh = self._need_tiers.get("normal", 0.50)

        for col in self._function_cols:
            ratio = self._calc_strong_need_ratio_for_col(cluster_df, col)
            if ratio >= strong_thresh:
                need_tiers[col] = NeedTier.STRONG
            elif ratio >= high_thresh:
                need_tiers[col] = NeedTier.HIGH
            elif ratio >= normal_thresh:
                need_tiers[col] = NeedTier.NORMAL
            else:
                need_tiers[col] = NeedTier.LOW

        return need_tiers

    def _calc_strong_need_ratio_for_col(self, df: pd.DataFrame, col: str) -> float:
        """计算某列的强需求比例."""
        if col not in df.columns:
            return 0.0
        valid_scores = df[col].dropna()
        if len(valid_scores) == 0:
            return 0.0
        return (valid_scores <= 2).sum() / len(valid_scores)

    def _calc_strong_need_ratio(self, df: pd.DataFrame) -> dict[str, float]:
        """计算所有列的强需求比例."""
        return {col: self._calc_strong_need_ratio_for_col(df, col) for col in self._function_cols}

    def _get_top_differentiated_functions(self, cluster_df: pd.DataFrame, top_n: int = 5) -> list[str]:
        """获取差异化最大的功能."""
        overall_means = self._df[self._function_cols].mean()
        cluster_means = cluster_df[self._function_cols].mean()

        diff = (cluster_means - overall_means).abs()
        top_diff = diff.nlargest(top_n)
        return top_diff.index.tolist()

    def _calc_gender_distribution(self, df: pd.DataFrame) -> dict[str, float]:
        return self._calc_categorical_distribution(df, "gender")

    def _calc_age_distribution(self, df: pd.DataFrame) -> dict[str, float]:
        return self._calc_categorical_distribution(df, "age_group")

    def _calc_budget_distribution(self, df: pd.DataFrame) -> dict[str, float]:
        return self._calc_categorical_distribution(df, "purchase_budget")

    def _calc_categorical_distribution(self, df: pd.DataFrame, col: str) -> dict[str, float]:
        """计算分类分布."""
        if col not in df.columns:
            return {}
        counts = df[col].value_counts()
        total = counts.sum()
        return {str(k): v / total for k, v in counts.items()}

    def _calc_function_means(self, df: pd.DataFrame) -> dict[str, float]:
        """计算功能评分均值."""
        return {col: float(df[col].mean()) for col in self._function_cols if col in df.columns}

    def _calc_function_medians(self, df: pd.DataFrame) -> dict[str, float]:
        """计算功能评分中位数."""
        return {col: float(df[col].median()) for col in self._function_cols if col in df.columns}
