"""约束处理测试."""
import pytest
import numpy as np
from cabin_eval.optimization.constraints import ConstraintHandler


class TestConstraintHandler:
    def test_project_keeps_sum_to_one(self):
        base_weights = np.array([0.3, 0.3, 0.4])
        sibling_groups = [[0, 1, 2]]
        handler = ConstraintHandler(sibling_groups, base_weights, epsilon=0.1)

        weights = np.array([0.2, 0.3, 0.5])
        projected = handler.project(weights)

        assert abs(projected.sum() - 1.0) < 1e-6

    def test_project_respects_bounds(self):
        base_weights = np.array([0.4, 0.6])
        sibling_groups = [[0, 1]]
        handler = ConstraintHandler(sibling_groups, base_weights, epsilon=0.1)

        weights = np.array([0.35, 0.65])
        projected = handler.project(weights)

        assert projected[0] <= base_weights[0] + 0.1 + 1e-8
        assert projected[1] <= base_weights[1] + 0.1 + 1e-8
        assert projected[0] >= max(0, base_weights[0] - 0.1) - 1e-8
        assert projected[1] >= max(0, base_weights[1] - 0.1) - 1e-8

    def test_validate_no_violations(self):
        base_weights = np.array([0.4, 0.6])
        sibling_groups = [[0, 1]]
        handler = ConstraintHandler(sibling_groups, base_weights, epsilon=0.1)

        weights = np.array([0.4, 0.6])
        violations = handler.validate(weights)

        assert len(violations) == 0

    def test_validate_detects_negative(self):
        base_weights = np.array([0.4, 0.6])
        sibling_groups = [[0, 1]]
        handler = ConstraintHandler(sibling_groups, base_weights, epsilon=0.1)

        weights = np.array([-0.1, 1.1])
        violations = handler.validate(weights)

        assert len(violations) > 0
