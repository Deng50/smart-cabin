"""熵权法测试."""
import pytest
import numpy as np
import pandas as pd
from cabin_eval.weighting.entropy import EntropyWeightCalculator


class TestEntropyWeightCalculator:
    def test_calculate_weights(self):
        calc = EntropyWeightCalculator()

        data = pd.DataFrame({
            "IND001": [85, 78, 90, 72],
            "IND002": [80, 82, 88, 75],
            "IND003": [90, 85, 95, 80],
        })

        weights = calc.calculate(data, ["IND001", "IND002", "IND003"])

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(w >= 0 for w in weights.values())

    def test_min_max_normalize(self):
        calc = EntropyWeightCalculator()

        X = np.array([[80, 90], [70, 85], [90, 95]])

        X_norm = calc._min_max_normalize(X)

        assert X_norm.min() >= 0
        assert X_norm.max() <= 1

    def test_constant_column_handling(self):
        calc = EntropyWeightCalculator()

        data = pd.DataFrame({
            "IND001": [80, 80, 80, 80],
            "IND002": [70, 75, 85, 90],
        })

        weights = calc.calculate(data, ["IND001", "IND002"])

        assert weights["IND001"] == 0.0
        assert weights["IND002"] > 0
