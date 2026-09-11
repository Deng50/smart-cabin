"""图表生成模块."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cabin_eval.schemas import SegmentationMetrics, UserProfile


class ChartGenerator:
    """图表生成器."""

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._charts_dir = output_dir / "charts"
        self._charts_dir.mkdir(parents=True, exist_ok=True)

    def generate_elbow_chart(self, metrics: list[SegmentationMetrics]) -> str:
        """生成肘部图."""
        ks = [m.k for m in metrics]
        sses = [m.sse for m in metrics]

        plt.figure(figsize=(10, 6))
        plt.plot(ks, sses, "bo-")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("SSE / Inertia")
        plt.title("Elbow Method for Optimal K")
        plt.grid(True, alpha=0.3)

        filepath = self._charts_dir / "elbow.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def generate_cluster_sizes(
        self, selected_k: int, labels: np.ndarray
    ) -> str:
        """生成聚类大小图."""
        unique, counts = np.unique(labels, return_counts=True)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique.astype(str), counts, color="steelblue")
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Samples")
        plt.title(f"Cluster Sizes (K={selected_k})")

        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
            )

        filepath = self._charts_dir / "cluster_sizes.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def generate_profile_heatmap(self, profiles: list[UserProfile]) -> str:
        """生成用户画像热力图."""
        if not profiles:
            return ""

        function_cols = list(profiles[0].function_mean_scores.keys())
        data_matrix = []
        for profile in profiles:
            row = [profile.function_mean_scores.get(col, 0) for col in function_cols]
            data_matrix.append(row)

        data = np.array(data_matrix)

        plt.figure(figsize=(14, 8))
        plt.imshow(data, aspect="auto", cmap="YlOrRd")
        plt.colorbar(label="Mean Score")
        plt.xlabel("Function")
        plt.ylabel("Cluster")
        plt.title("User Profile Heatmap (Function Mean Scores)")

        plt.xticks(
            range(len(function_cols)),
            [col.replace("function_", "") for col in function_cols],
            rotation=45,
            ha="right",
        )
        plt.yticks(range(len(profiles)), [p.cluster_id for p in profiles])

        filepath = self._charts_dir / "profile_heatmap.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        return str(filepath)
