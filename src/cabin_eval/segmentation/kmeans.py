"""K-means++ 分群模块."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from cabin_eval.schemas import SegmentationDecision, SegmentationMetrics


class KMeansSegmenter:
    """K-means++ 分群器."""

    def __init__(self, config: dict[str, Any]):
        self._cfg = config
        self._random_seed = config.get("random_seed", 42)
        self._n_init = config.get("n_init", 50)
        self._min_cluster_ratio = config.get("min_cluster_ratio", 0.08)

    def find_optimal_k(self, X: np.ndarray) -> list[SegmentationMetrics]:
        """寻找最优K值 (不自动选择，只计算指标供人工决策)."""
        n_samples = X.shape[0]
        max_k = min(n_samples, self._cfg.get("max_k", 10))
        candidate_k = [k for k in self._cfg.get("candidate_k", list(range(2, 11))) if 2 <= k <= max_k]
        metrics = []

        for k in candidate_k:
            kmeans = KMeans(
                n_clusters=k,
                n_init=self._n_init,
                random_state=self._random_seed,
            )
            labels = kmeans.fit_predict(X)

            sse = float(kmeans.inertia_)

            silhouette = silhouette_score(X, labels) if k > 1 else None
            ch_score = calinski_harabasz_score(X, labels) if k > 1 else None

            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes = {str(int(u)): int(c) for u, c in zip(unique, counts)}
            min_ratio = min(counts) / len(labels) if len(counts) > 0 else 0

            metrics.append(SegmentationMetrics(
                k=k,
                sse=sse,
                silhouette_score=silhouette,
                calinski_harabasz_score=ch_score,
                cluster_sizes=cluster_sizes,
                min_cluster_ratio=min_ratio,
            ))

        return metrics

    def cluster(self, X: np.ndarray, k: int) -> np.ndarray:
        """执行K-means聚类."""
        kmeans = KMeans(
            n_clusters=k,
            n_init=self._n_init,
            random_state=self._random_seed,
        )
        return kmeans.fit_predict(X)

    def select_k_manual(self, metrics: list[SegmentationMetrics], manual_k: int) -> SegmentationDecision:
        """手动选择K值."""
        valid_metrics = [m for m in metrics if m.k == manual_k]
        if not valid_metrics:
            raise ValueError(f"K={manual_k} not in candidate metrics")

        return SegmentationDecision(
            selected_k=manual_k,
            mode="manual",
            override_reason=f"Manual selection: {manual_k} clusters (config: manual_k)",
            candidate_metrics=metrics,
        )

    def select_k_auto(self, metrics: list[SegmentationMetrics]) -> SegmentationDecision:
        """自动选择K值 (基于多指标综合评估)."""
        filtered = [m for m in metrics if m.min_cluster_ratio and m.min_cluster_ratio >= self._min_cluster_ratio]

        if not filtered:
            filtered = metrics

        best = max(filtered, key=lambda m: (
            m.silhouette_score if m.silhouette_score else 0,
            m.calinski_harabasz_score if m.calinski_harabasz_score else 0,
        ))

        return SegmentationDecision(
            selected_k=best.k,
            mode="auto",
            override_reason="Auto selection based on silhouette and Calinski-Harabasz scores",
            candidate_metrics=metrics,
        )
