"""数据预处理模块."""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class QuestionnairePreprocessor:
    """问卷数据预处理."""

    def __init__(self, config: dict):
        self._cfg = config

    def prepare_for_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备用于聚类的数据."""
        function_cols = [c for c in df.columns if c.startswith("function_")]
        X = df[function_cols].copy()

        X = X.fillna(X.median())

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=function_cols,
            index=X.index
        )

        return X_scaled

    def detect_outliers_iqr(self, df: pd.DataFrame, factor: float = 1.5) -> pd.Series:
        """使用IQR方法检测异常值."""
        function_cols = [c for c in df.columns if c.startswith("function_")]
        outliers = pd.Series([False] * len(df), index=df.index)

        for col in function_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            outliers |= (df[col] < lower) | (df[col] > upper)

        return outliers
