"""权重融合测试."""
import pytest
from cabin_eval.weighting.fusion import WeightFusion


class TestWeightFusion:
    def test_expert_only_mode(self):
        fusion = WeightFusion(mode="expert_only", alpha=0.7)

        expert_weights = {"IND001": 0.6, "IND002": 0.4}
        entropy_weights = {"IND001": 0.5, "IND002": 0.5}

        result = fusion.fuse(expert_weights, entropy_weights)

        assert result == expert_weights

    def test_linear_fusion_mode(self):
        fusion = WeightFusion(mode="linear_fusion", alpha=0.7)

        expert_weights = {"IND001": 0.6, "IND002": 0.4}
        entropy_weights = {"IND001": 0.5, "IND002": 0.5}

        result = fusion.fuse(expert_weights, entropy_weights)

        expected = {
            "IND001": 0.7 * 0.6 + 0.3 * 0.5,
            "IND002": 0.7 * 0.4 + 0.3 * 0.5,
        }

        assert abs(result["IND001"] - expected["IND001"]) < 1e-6
        assert abs(result["IND002"] - expected["IND002"]) < 1e-6

    def test_weights_sum_to_one(self):
        fusion = WeightFusion(mode="linear_fusion", alpha=0.7)

        expert_weights = {"IND001": 0.6, "IND002": 0.4}
        entropy_weights = {"IND001": 0.5, "IND002": 0.5}

        result = fusion.fuse(expert_weights, entropy_weights)

        assert abs(sum(result.values()) - 1.0) < 1e-6
