"""问卷数据模型测试."""
import pytest
from cabin_eval.schemas import (
    QuestionnaireMeta,
    QuestionnaireRow,
    Gender,
    AgeGroup,
    NeedTier,
)


class TestQuestionnaireSchemas:
    def test_questionnaire_row_valid(self):
        row = QuestionnaireRow(
            sample_id="S001",
            gender=Gender.MALE,
            age_group=AgeGroup.AGE_25_34,
            function_scores={"function_1": 4, "function_2": 3},
        )
        assert row.sample_id == "S001"
        assert row.gender == "male"

    def test_questionnaire_meta(self):
        meta = QuestionnaireMeta(
            total_rows=100,
            valid_rows=90,
            removed_rows=10,
            removed_reasons={"duplicate": 5, "excessive_unknown": 5},
            imputation_records=[],
            rating_direction="lower_is_stronger",
            rating_min=1,
            rating_max=5,
        )
        assert meta.valid_rows == 90
        assert meta.removed_reasons["duplicate"] == 5


class TestNeedTier:
    def test_need_tier_values(self):
        assert NeedTier.STRONG.value == "strong"
        assert NeedTier.HIGH.value == "high"
        assert NeedTier.NORMAL.value == "normal"
        assert NeedTier.LOW.value == "low"
