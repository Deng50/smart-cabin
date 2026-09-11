"""车型数据导入."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class VehicleIngester:
    """车型数据导入器."""

    def load(self, filepath: str | Path) -> pd.DataFrame:
        """加载车型评分数据."""
        filepath = Path(filepath)
        if filepath.suffix == ".csv":
            return pd.read_csv(filepath)
        elif filepath.suffix in (".xlsx", ".xls"):
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def validate(self, df: pd.DataFrame) -> list[str]:
        """验证车型数据."""
        errors = []

        if "vehicle_id" not in df.columns:
            errors.append("Missing required column: vehicle_id")

        return errors
