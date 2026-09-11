"""AHP权重计算测试."""
import pytest
import numpy as np
import pandas as pd
from cabin_eval.weighting.ahp import AHPWeightCalculator


class TestAHPWeightCalculator:
    def test_compute_weights(self):
        calc = AHPWeightCalculator(cr_threshold=0.10)

        matrix = np.array([
            [1.0, 3.0],
            [1/3, 1.0],
        ])

        weights = calc._compute_weights(matrix)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert weights[0] > weights[1]

    def test_lambda_max(self):
        calc = AHPWeightCalculator(cr_threshold=0.10)

        matrix = np.array([
            [1.0, 3.0],
            [1/3, 1.0],
        ])
        weights = np.array([0.75, 0.25])

        lambda_max = calc._compute_lambda_max(matrix, weights)
        assert lambda_max > 0

    def test_calculate_weights_from_judgments(self):
        calc = AHPWeightCalculator(cr_threshold=0.10)

        judgments_data = {
            "expert_id": ["E001", "E001", "E001"],
            "parent_indicator_id": ["safety", "safety", "safety"],
            "left_indicator_id": ["IND001", "IND001", "IND002"],
            "right_indicator_id": ["IND002", "IND003", "IND003"],
            "ratio": [3.0, 2.0, 2.0],
        }
        df = pd.DataFrame(judgments_data)

        results = calc.calculate_weights(df, ["IND001", "IND002", "IND003"])

        assert len(results) > 0
        result = results[0]
        assert result.expert_id == "E001"
        assert result.is_consistent is not None

    def test_aggregate_expert_weights(self):
        calc = AHPWeightCalculator(cr_threshold=0.10)

        from cabin_eval.schemas import AhpResult

        results = [
            AhpResult(
                expert_id="E001",
                parent_indicator_id="safety",
                local_weights={"IND001": 0.6, "IND002": 0.4},
                lambda_max=2.0,
                ci=0.0,
                cr=0.0,
                is_consistent=True,
            ),
            AhpResult(
                expert_id="E002",
                parent_indicator_id="safety",
                local_weights={"IND001": 0.5, "IND002": 0.5},
                lambda_max=2.0,
                ci=0.0,
                cr=0.0,
                is_consistent=True,
            ),
        ]

        authority_df = pd.DataFrame({
            "expert_id": ["E001", "E002"],
            "authority_score": [0.6, 0.4],
        })

        weights = calc.aggregate_expert_weights(results, authority_df)

        assert "IND001" in weights
        assert "IND002" in weights
        assert abs(weights["IND001"] + weights["IND002"] - 1.0) < 1e-6
