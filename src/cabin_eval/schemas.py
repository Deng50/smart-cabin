"""数据模型定义 (Pydantic schemas)."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class AgeGroup(str, Enum):
    UNDER_25 = "under_25"
    AGE_25_34 = "25_34"
    AGE_35_44 = "35_44"
    AGE_45_54 = "45_54"
    OVER_55 = "over_55"


class Education(str, Enum):
    HIGH_SCHOOL_OR_LESS = "high_school_or_less"
    COLLEGE = "college"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTOR = "doctor"


class CityTier(str, Enum):
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4_OR_BELOW = "tier_4_or_below"


class FamilyStage(str, Enum):
    SINGLE = "single"
    COUPLE_NO_KIDS = "couple_no_kids"
    YOUNG_PARENTS = "young_parents"
    MIDDLE_AGED_PARENTS = "middle_aged_parents"
    EMPTY_NEST = "empty_nest"


class PurchaseBudget(str, Enum):
    UNDER_150K = "under_150k"
    TIER_150K_250K = "150k_250k"
    TIER_250K_400K = "250k_400k"
    TIER_400K_600K = "400k_600k"
    OVER_600K = "over_600k"


class OwnershipStatus(str, Enum):
    FIRST_TIME = "first_time"
    REPEAT_BUYER = "repeat_buyer"
    LUXURY_UPGRADE = "luxury_upgrade"


class IndicatorCategory(str, Enum):
    REQUIRED = "required"
    BONUS = "bonus"
    EXCLUDED = "excluded"


class NeedTier(str, Enum):
    STRONG = "strong"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# === 问卷数据模型 ===

class QuestionnaireRow(BaseModel):
    """问卷数据行."""
    sample_id: str
    gender: Gender | None = None
    age_group: AgeGroup | None = None
    education: Education | None = None
    city_tier: CityTier | None = None
    family_stage: FamilyStage | None = None
    purchase_budget: PurchaseBudget | None = None
    ownership_status: OwnershipStatus | None = None
    function_scores: Annotated[dict[str, int | None], Field(default_factory=dict)]

    class Config:
        use_enum_values = True


class QuestionnaireMeta(BaseModel):
    """问卷元数据."""
    total_rows: int
    valid_rows: int
    removed_rows: int
    removed_reasons: dict[str, int]
    imputation_records: list[dict[str, str]]
    rating_direction: str
    rating_min: int
    rating_max: int


# === 指标体系模型 ===

class IndicatorNode(BaseModel):
    """指标树节点."""
    indicator_id: str
    level_1: str = ""
    level_2: str = ""
    level_3: str = ""
    level_4: str = ""
    feature_name: str | None = None
    category: IndicatorCategory = IndicatorCategory.REQUIRED
    installation_rate: float | None = None
    score_max: float = 100.0
    enabled: bool = True
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    local_weight: float = 0.0
    global_weight: float = 0.0

    class Config:
        use_enum_values = True


class IndicatorTree(BaseModel):
    """指标树."""
    nodes: dict[str, IndicatorNode] = Field(default_factory=dict)
    root_ids: list[str] = Field(default_factory=list)

    def add_node(self, node: IndicatorNode) -> None:
        self.nodes[node.indicator_id] = node

    def get_children(self, parent_id: str) -> list[IndicatorNode]:
        parent = self.nodes.get(parent_id)
        if not parent:
            return []
        return [self.nodes[cid] for cid in parent.children_ids if cid in self.nodes]

    def get_leaves(self) -> list[IndicatorNode]:
        return [n for n in self.nodes.values() if not n.children_ids]


# === 专家数据模型 ===

class ExpertJudgment(BaseModel):
    """专家判断矩阵条目."""
    expert_id: str
    parent_indicator_id: str
    left_indicator_id: str
    right_indicator_id: str
    ratio: float

    @field_validator("ratio")
    @classmethod
    def validate_ratio(cls, v: float) -> float:
        if not (1 / 9 <= v <= 9):
            raise ValueError(f"Saaty ratio must be in [1/9, 9], got {v}")
        return v


class ExpertAuthority(BaseModel):
    """专家权威度."""
    expert_id: str
    authority_score: float = Field(ge=0.0, le=1.0)


class AhpResult(BaseModel):
    """AHP计算结果."""
    expert_id: str
    parent_indicator_id: str
    local_weights: dict[str, float]
    lambda_max: float
    ci: float
    cr: float
    is_consistent: bool
    rejected_items: list[str] = Field(default_factory=list)


# === 用户画像模型 ===

class UserProfile(BaseModel):
    """用户画像."""
    cluster_id: str
    sample_count: int
    proportion: float
    need_tier: dict[str, NeedTier]
    top_differentiated_functions: list[str] = Field(default_factory=list)
    gender_distribution: dict[str, float] = Field(default_factory=dict)
    age_distribution: dict[str, float] = Field(default_factory=dict)
    budget_distribution: dict[str, float] = Field(default_factory=dict)
    function_mean_scores: dict[str, float] = Field(default_factory=dict)
    function_median_scores: dict[str, float] = Field(default_factory=dict)
    strong_need_ratio: dict[str, float] = Field(default_factory=dict)


# === 分群模型 ===

class SegmentationMetrics(BaseModel):
    """分群指标."""
    k: int
    sse: float
    silhouette_score: float | None = None
    calinski_harabasz_score: float | None = None
    cluster_sizes: dict[str, int] = Field(default_factory=dict)
    min_cluster_ratio: float | None = None


class SegmentationDecision(BaseModel):
    """分群决策."""
    selected_k: int
    mode: str
    override_reason: str | None = None
    candidate_metrics: list[SegmentationMetrics] = Field(default_factory=list)


# === 权重模型 ===

class WeightResult(BaseModel):
    """权重计算结果."""
    indicator_id: str
    base_weight: float
    expert_weight: float | None = None
    entropy_weight: float | None = None
    history_factor: float = 1.0
    final_weight: float


class WeightChangeLog(BaseModel):
    """权重变更日志."""
    indicator_id: str
    base_weight: float
    need_tier: NeedTier | None = None
    multiplier: float = 1.0
    final_local_weight: float
    final_global_weight: float
    change_reason: str


# === 车型数据模型 ===

class VehicleScore(BaseModel):
    """车型评分."""
    vehicle_id: str
    indicator_scores: dict[str, float]
    overall_score: float | None = None


# === 优化样本模型 ===

class OptimizationSample(BaseModel):
    """优化样本."""
    sample_id: str
    indicator_scores: dict[str, float]
    target_score: float
    split: Split

    class Config:
        use_enum_values = True


# === 运行记录模型 ===

class RunRecord(BaseModel):
    """运行记录."""
    run_id: str
    status: RunStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    config_snapshot: dict
    input_hashes: dict[str, str]
    output_artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
