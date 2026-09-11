"""集成测试."""
import pytest
from pathlib import Path
from cabin_eval.config import Config


class TestConfig:
    def test_load_default_config(self):
        cfg = Config()
        assert cfg.questionnaire_config is not None
        assert cfg.segmentation_config is not None

    def test_get_nested_value(self):
        cfg = Config()
        val = cfg.get("questionnaire.rating_min")
        assert val is not None


class TestIngestion:
    def test_questionnaire_template_exists(self):
        template_path = Path(__file__).parent.parent.parent / "data" / "templates" / "questionnaire_template.xlsx"
        assert template_path.exists() or template_path.with_suffix(".csv").exists()
