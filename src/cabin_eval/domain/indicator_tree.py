"""指标树构建与管理."""
from __future__ import annotations

from typing import Any

from cabin_eval.schemas import IndicatorCategory, IndicatorNode, IndicatorTree


class IndicatorTreeBuilder:
    """指标树构建器."""

    def __init__(self):
        self._tree = IndicatorTree()
        self._children_map: dict[str, list[str]] = {}

    def build_from_dataframe(self, df: Any) -> IndicatorTree:
        """从DataFrame构建指标树."""
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")

        required_cols = ["indicator_id", "level_1", "category", "enabled"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for _, row in df.iterrows():
            node = IndicatorNode(
                indicator_id=str(row["indicator_id"]),
                level_1=str(row.get("level_1", "")),
                level_2=str(row.get("level_2", "")),
                level_3=str(row.get("level_3", "")),
                level_4=str(row.get("level_4", "")),
                feature_name=str(row.get("feature_name", "")) if pd.notna(row.get("feature_name")) else None,
                category=IndicatorCategory(row.get("category", "required")),
                installation_rate=float(row["installation_rate"]) if pd.notna(row.get("installation_rate")) else None,
                score_max=float(row.get("score_max", 100.0)),
                enabled=bool(row.get("enabled", True)),
            )
            self._tree.add_node(node)

        self._build_hierarchy()
        return self._tree

    def _build_hierarchy(self) -> None:
        """构建指标层级关系."""
        for node_id, node in self._tree.nodes.items():
            path_key = f"{node.level_1}/{node.level_2}/{node.level_3}/{node.level_4}"
            if node.level_4:
                parent_key = f"{node.level_1}/{node.level_2}/{node.level_3}/"
                self._add_child_relation(parent_key, node_id)
            elif node.level_3:
                parent_key = f"{node.level_1}/{node.level_2}/"
                self._add_child_relation(parent_key, node_id)
            elif node.level_2:
                parent_key = node.level_1
                self._add_child_relation(parent_key, node_id)

        self._tree.root_ids = [nid for nid, n in self._tree.nodes.items() if not n.parent_id]

    def _add_child_relation(self, parent_key: str, child_id: str) -> None:
        """添加父子关系."""
        parent_node = None
        for nid, node in self._tree.nodes.items():
            path = f"{node.level_1}/{node.level_2}/{node.level_3}/{node.level_4}".rstrip("/")
            if path == parent_key or (node.level_1 == parent_key and not node.level_2):
                parent_node = node
                break

        if parent_node:
            parent_node.children_ids.append(child_id)
            self._tree.nodes[child_id].parent_id = parent_node.indicator_id


class IndicatorTreeManager:
    """指标树管理器."""

    def __init__(self, tree: IndicatorTree):
        self._tree = tree

    def get_leaves(self) -> list[IndicatorNode]:
        """获取所有叶子节点."""
        return self._tree.get_leaves()

    def get_root_nodes(self) -> list[IndicatorNode]:
        """获取根节点列表."""
        return [self._tree.nodes[rid] for rid in self._tree.root_ids if rid in self._tree.nodes]

    def normalize_sibling_weights(self, parent_id: str) -> None:
        """归一化兄弟节点权重."""
        children = self._tree.get_children(parent_id)
        if not children:
            return

        total = sum(c.local_weight for c in children)
        if total > 0:
            for child in children:
                child.local_weight /= total

    def calculate_global_weights(self) -> None:
        """计算全局权重."""
        for root in self.get_root_nodes():
            self._propagate_global_weight(root.indicator_id, 1.0)

    def _propagate_global_weight(self, node_id: str, parent_global: float) -> None:
        """递归计算全局权重."""
        node = self._tree.nodes.get(node_id)
        if not node:
            return

        node.global_weight = parent_global * node.local_weight

        children = self._tree.get_children(node_id)
        if children:
            for child in children:
                self._propagate_global_weight(child.indicator_id, node.global_weight)

    def validate_constraints(self, tolerance: float = 1e-8) -> list[str]:
        """验证权重约束."""
        violations = []

        for node_id, node in self._tree.nodes.items():
            if node.children_ids:
                children = self._tree.get_children(node_id)
                weight_sum = sum(c.local_weight for c in children)
                if abs(weight_sum - 1.0) > tolerance:
                    violations.append(
                        f"Node {node_id} children weight sum = {weight_sum}, expected 1.0"
                    )

        return violations
