"""环境适配器模块."""
from typing import Any

from cabin_eval.schemas import IndicatorTree


class EnvironmentAdapter:
    """环境因素适配器."""

    def __init__(self, rules: dict[str, Any]):
        self._enabled = rules.get("enabled", False)
        self._rules = rules.get("rules", [])

    def is_enabled(self) -> bool:
        """检查是否启用."""
        return self._enabled

    def adapt_weights(
        self,
        tree: IndicatorTree,
        context: dict[str, Any],
    ) -> dict[str, float]:
        """根据环境上下文调整权重."""
        if not self._enabled:
            return {}

        weight_adjustments: dict[str, float] = {}

        for rule in self._rules:
            rule_context = rule.get("context", {})
            if self._matches_context(context, rule_context):
                indicator_id = rule.get("indicator_id")
                action = rule.get("action")
                value = rule.get("value", 1.0)

                if action == "multiply_weight" and indicator_id:
                    weight_adjustments[indicator_id] = value

        return weight_adjustments

    def _matches_context(
        self, context: dict[str, Any], rule_context: dict[str, Any]
    ) -> bool:
        """检查上下文是否匹配规则."""
        for key, value in rule_context.items():
            if context.get(key) != value:
                return False
        return True
