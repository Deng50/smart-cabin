"""问卷数据导入与清洗."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from cabin_eval.config import Config
from cabin_eval.schemas import QuestionnaireMeta


class QuestionnaireIngester:
    """问卷数据导入器."""

    def __init__(self, config: Config):
        self._cfg = config
        self._questionnaire_cfg = config.questionnaire_config

    def load(self, filepath: str | Path) -> pd.DataFrame:
        """加载问卷数据."""
        filepath = Path(filepath)
        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
        elif filepath.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        return df

    def validate(self, df: pd.DataFrame) -> list[str]:
        """验证问卷数据基本结构."""
        errors = []

        if "sample_id" not in df.columns:
            errors.append("Missing required column: sample_id")

        function_cols = [c for c in df.columns if c.startswith("function_")]
        if not function_cols:
            errors.append("No function score columns found (expected function_* prefix)")

        rating_min = self._questionnaire_cfg.get("rating_min", 1)
        rating_max = self._questionnaire_cfg.get("rating_max", 5)

        for col in function_cols:
            invalid = df[col].dropna()
            invalid = invalid[(invalid < rating_min) | (invalid > rating_max)]
            if len(invalid) > 0:
                errors.append(f"Column {col} has values outside [{rating_min}, {rating_max}]")

        return errors

    def clean(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, QuestionnaireMeta]:
        """清洗问卷数据."""
        df = df.copy()
        unknown_values = self._questionnaire_cfg.get("unknown_values", [0, None, "不了解"])
        max_unknown_ratio = self._questionnaire_cfg.get("max_unknown_ratio", 0.20)

        function_cols = [c for c in df.columns if c.startswith("function_")]
        demographic_cols = [
            "gender", "age_group", "education", "city_tier",
            "family_stage", "purchase_budget", "ownership_status"
        ]

        removed_reasons: dict[str, int] = {}
        imputation_records: list[dict[str, str]] = []

        if "sample_id" in df.columns:
            duplicates = df["sample_id"].duplicated()
            dup_count = duplicates.sum()
            if dup_count > 0:
                df = df[~duplicates]
                removed_reasons["duplicate_sample_id"] = dup_count

        total_rows = len(df)
        unknown_counts = df[function_cols].apply(lambda x: x.isin(unknown_values).sum(axis=1))
        unknown_ratios = unknown_counts / len(function_cols)
        excessive_unknown = unknown_ratios > max_unknown_ratio
        excess_count = excessive_unknown.sum()
        if excess_count > 0:
            df = df[~excessive_unknown]
            removed_reasons["excessive_unknown"] = excess_count

        for col in function_cols:
            if col not in df.columns:
                continue
            unknown_mask = df[col].isin(unknown_values)
            unknown_count = unknown_mask.sum()

            if unknown_count > 0:
                median_val = df.loc[~unknown_mask, col].median()
                if pd.isna(median_val):
                    median_val = (self._questionnaire_cfg.get("rating_min", 1) +
                                  self._questionnaire_cfg.get("rating_max", 5)) / 2

                for idx in df[unknown_mask].index:
                    imputation_records.append({
                        "sample_id": str(df.loc[idx, "sample_id"]),
                        "column": col,
                        "imputed_value": str(median_val),
                    })
                df.loc[unknown_mask, col] = median_val

        valid_rows = len(df)
        removed_rows = total_rows - valid_rows

        rating_direction = "lower_is_stronger" if self._questionnaire_cfg.get(
            "lower_score_means_stronger_need", True
        ) else "higher_is_stronger"

        meta = QuestionnaireMeta(
            total_rows=total_rows,
            valid_rows=valid_rows,
            removed_rows=removed_rows,
            removed_reasons=removed_reasons,
            imputation_records=imputation_records,
            rating_direction=rating_direction,
            rating_min=self._questionnaire_cfg.get("rating_min", 1),
            rating_max=self._questionnaire_cfg.get("rating_max", 5),
        )

        return df, meta
