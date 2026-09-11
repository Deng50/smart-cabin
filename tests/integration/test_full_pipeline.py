"""完整流程集成测试."""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from cabin_eval.config import Config
from cabin_eval.ingestion.questionnaire import QuestionnaireIngester
from cabin_eval.ingestion.indicators import IndicatorIngester
from cabin_eval.segmentation.preprocessing import QuestionnairePreprocessor
from cabin_eval.segmentation.kmeans import KMeansSegmenter


class TestFullPipeline:
    @pytest.fixture
    def demo_data_path(self):
        return Path(__file__).parent.parent.parent / "data" / "demo"

    def test_questionnaire_loading(self, demo_data_path):
        q_path = demo_data_path / "questionnaire_demo.csv"
        if not q_path.exists():
            pytest.skip("Demo data not found")

        cfg = Config()
        ingester = QuestionnaireIngester(cfg)
        df = ingester.load(q_path)

        assert len(df) > 0
        assert "sample_id" in df.columns
        function_cols = [c for c in df.columns if c.startswith("function_")]
        assert len(function_cols) > 0

    def test_questionnaire_cleaning(self, demo_data_path):
        q_path = demo_data_path / "questionnaire_demo.csv"
        if not q_path.exists():
            pytest.skip("Demo data not found")

        cfg = Config()
        ingester = QuestionnaireIngester(cfg)
        df = ingester.load(q_path)

        clean_df, meta = ingester.clean(df)

        assert meta.total_rows >= meta.valid_rows
        assert meta.valid_rows > 0

    def test_indicator_loading(self, demo_data_path):
        ind_path = demo_data_path / "indicators_demo.csv"
        if not ind_path.exists():
            pytest.skip("Demo data not found")

        ingester = IndicatorIngester()
        df = ingester.load(ind_path)

        assert len(df) > 0
        assert "indicator_id" in df.columns
        assert "level_1" in df.columns

    def test_end_to_end_small_dataset(self, demo_data_path):
        q_path = demo_data_path / "questionnaire_demo.csv"
        ind_path = demo_data_path / "indicators_demo.csv"

        if not q_path.exists() or not ind_path.exists():
            pytest.skip("Demo data not found")

        cfg = Config()
        q_ingester = QuestionnaireIngester(cfg)
        ind_ingester = IndicatorIngester()

        df = q_ingester.load(q_path)
        clean_df, _ = q_ingester.clean(df)

        ind_df = ind_ingester.load(ind_path)
        tree = ind_ingester.build_tree(ind_df)

        assert len(tree.nodes) > 0

        preprocessor = QuestionnairePreprocessor(cfg.segmentation_config)
        X = preprocessor.prepare_for_clustering(clean_df)

        segmenter = KMeansSegmenter(cfg.segmentation_config)
        metrics = segmenter.find_optimal_k(X.values)

        assert len(metrics) > 0
