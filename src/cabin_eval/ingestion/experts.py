"""专家数据导入."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cabin_eval.schemas import ExpertAuthority, ExpertJudgment


class ExpertIngester:
    """专家数据导入器."""

    def load_judgments(self, filepath: str | Path) -> pd.DataFrame:
        """加载专家判断矩阵."""
        filepath = Path(filepath)
        if filepath.suffix == ".csv":
            return pd.read_csv(filepath)
        elif filepath.suffix in (".xlsx", ".xls"):
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def load_authority(self, filepath: str | Path) -> pd.DataFrame:
        """加载专家权威度."""
        filepath = Path(filepath)
        if filepath.suffix == ".csv":
            return pd.read_csv(filepath)
        elif filepath.suffix in (".xlsx", ".xls"):
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def validate_judgments(self, df: pd.DataFrame) -> list[str]:
        """验证专家判断数据."""
        errors = []
        required_cols = ["expert_id", "parent_indicator_id", "left_indicator_id", "right_indicator_id", "ratio"]
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if "ratio" in df.columns:
            invalid_ratio = ~((df["ratio"] >= 1/9) & (df["ratio"] <= 9))
            if invalid_ratio.any():
                errors.append(f"ratio values must be in [1/9, 9], found {invalid_ratio.sum()} invalid")

        return errors
