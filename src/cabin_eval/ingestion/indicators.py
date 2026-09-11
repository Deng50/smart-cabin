"""指标体系导入."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cabin_eval.domain.indicator_tree import IndicatorTreeBuilder
from cabin_eval.schemas import IndicatorTree


class IndicatorIngester:
    """指标体系导入器."""

    def load(self, filepath: str | Path) -> pd.DataFrame:
        """加载指标数据."""
        filepath = Path(filepath)
        if filepath.suffix == ".csv":
            return pd.read_csv(filepath)
        elif filepath.suffix in (".xlsx", ".xls"):
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def validate(self, df: pd.DataFrame) -> list[str]:
        """验证指标数据."""
        errors = []

        required_cols = ["indicator_id", "level_1", "category", "enabled"]
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if "indicator_id" in df.columns:
            if df["indicator_id"].duplicated().any():
                errors.append("Duplicate indicator_id found")

        valid_categories = {"required", "bonus", "excluded"}
        if "category" in df.columns:
            invalid_cats = ~df["category"].isin(valid_categories)
            if invalid_cats.any():
                errors.append(f"Invalid category values: {df[invalid_cats]['category'].unique()}")

        if "enabled" in df.columns:
            invalid_enabled = ~df["enabled"].isin([True, False, 0, 1])
            if invalid_enabled.any():
                errors.append("enabled column must be boolean")

        return errors

    def build_tree(self, df: pd.DataFrame) -> IndicatorTree:
        """构建指标树."""
        builder = IndicatorTreeBuilder()
        return builder.build_from_dataframe(df)
