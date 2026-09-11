"""K-means分群测试."""
import pytest
import numpy as np
from cabin_eval.segmentation.kmeans import KMeansSegmenter
from cabin_eval.schemas import SegmentationMetrics


class TestKMeansSegmenter:
    def test_find_optimal_k(self):
        config = {
            "random_seed": 42,
            "n_init": 10,
            "min_cluster_ratio": 0.08,
            "candidate_k": [2, 3, 4],
        }
        segmenter = KMeansSegmenter(config)

        X = np.random.rand(100, 10)
        metrics = segmenter.find_optimal_k(X)

        assert len(metrics) == 3
        assert all(isinstance(m, SegmentationMetrics) for m in metrics)
        assert all(m.k in [2, 3, 4] for m in metrics)

    def test_cluster(self):
        config = {
            "random_seed": 42,
            "n_init": 10,
            "min_cluster_ratio": 0.08,
        }
        segmenter = KMeansSegmenter(config)

        X = np.random.rand(50, 5)
        labels = segmenter.cluster(X, k=3)

        assert len(labels) == 50
        assert len(np.unique(labels)) == 3

    def test_select_k_manual(self):
        config = {
            "random_seed": 42,
            "n_init": 10,
            "min_cluster_ratio": 0.08,
            "candidate_k": [2, 3, 4, 5],
            "manual_k": 3,
        }
        segmenter = KMeansSegmenter(config)

        X = np.random.rand(50, 5)
        metrics = segmenter.find_optimal_k(X)

        decision = segmenter.select_k_manual(metrics, manual_k=3)

        assert decision.selected_k == 3
        assert decision.mode == "manual"

    def test_select_k_auto(self):
        config = {
            "random_seed": 42,
            "n_init": 10,
            "min_cluster_ratio": 0.08,
            "candidate_k": [2, 3, 4],
        }
        segmenter = KMeansSegmenter(config)

        X = np.random.rand(50, 5)
        metrics = segmenter.find_optimal_k(X)

        decision = segmenter.select_k_auto(metrics)

        assert decision.mode == "auto"
        assert decision.selected_k in [2, 3, 4]
