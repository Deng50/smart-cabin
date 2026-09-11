# 智慧座舱测评体系自趋优模型：代码重建设计

## 0. 文档用途

本文是“多维影响因素下智慧座舱测评体系自趋优模型搭建”的重建规格，可直接交给 Claude 生成代码。目标不是逐行复刻遗留脚本，而是依据现有需求书、结项材料、数据表和残留原型，重建一个**可运行、可配置、可测试、可追溯**的版本。

事实基线：

- 项目面向中汽中心智慧座舱测评，融合用户需求、车型测评、专家意见和环境因素。
- 材料记录 722 份问卷、清洗后 676 份有效样本，覆盖 31 项座舱功能需求。
- 车型侧材料包含约 22 款车型、47 项测评指标，专家侧包含 13 位专家。
- 原始流程包含 K-means++、AHP、专家权威度、历史得分分布、熵权、遗传算法、指标树及 Web 文件上传/下载。
- 原材料存在五类、六类、十类用户结果冲突。新系统必须把聚类数设为可配置项，并保存选择依据，禁止以“最小 SSE”直接选 K。

## 1. 项目目标

系统需要实现以下闭环：

```text
问卷导入 -> 数据清洗 -> 用户分群 -> 用户画像
                                      |
指标体系 + 专家判断 + 车型历史数据 -> 基础权重
                                      |
用户画像 + 功能分类 + 环境上下文 -> 差异化指标树
                                      |
标注真值/用户评分 -> 遗传算法约束优化 -> 版本化测评体系
                                      |
                         Excel/JSON/图表/Web 输出
```

核心业务能力：

1. 从问卷中识别典型用户群体并生成可解释画像。
2. 从专家判断和车型数据中生成分层指标权重。
3. 根据不同用户需求生成差异化测评体系。
4. 使用带约束的优化算法，根据新一批数据迭代权重。
5. 保存每次运行的输入、配置、结果和权重变化记录。

## 2. 非目标与边界

- 不把聚类标签自动包装成营销人群名称；名称需要业务人员确认。
- 不把熵权、环境修正等“材料中存在但集成状态不明确”的功能默认写入最终评分。它们作为可开关模块实现。
- 不使用 `eval` 解析 Excel 内容，不允许导入模块时自动执行任务。
- 不承诺量产车机端运行；本项目是智慧座舱测评与决策支持工具。
- 原始问卷可能包含个人属性，系统只保存匿名样本编号，不采集姓名、电话等直接身份信息。

## 3. 推荐技术栈

| 层级 | 技术 |
| --- | --- |
| 语言 | Python 3.11+ |
| 数据处理 | pandas、NumPy、openpyxl |
| 聚类与指标 | scikit-learn、SciPy |
| 数据校验 | Pydantic 2 |
| Web API | FastAPI、Uvicorn |
| 页面 | Jinja2 + 原生 HTML/CSS/JS，优先保持轻量 |
| 数据库 | SQLite + SQLAlchemy 2；后续可切 PostgreSQL |
| 图表 | matplotlib / seaborn |
| 测试 | pytest、pytest-cov、httpx |
| 代码质量 | ruff、mypy、pre-commit |
| 容器化 | Docker、docker compose |

要求通过 `pyproject.toml` 固定依赖，所有随机算法必须从配置读取 `random_seed`。

## 4. 建议目录结构

```text
smart-cabin-evaluation/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── configs/
│   ├── default.yaml
│   ├── feature_taxonomy.yaml
│   └── environment_rules.yaml
├── data/
│   ├── templates/                 # 空白导入模板
│   ├── demo/                      # 脱敏演示数据
│   └── runs/                      # 每次运行的隔离目录
├── src/cabin_eval/
│   ├── config.py
│   ├── schemas.py
│   ├── domain/
│   │   ├── indicator_tree.py
│   │   ├── user_profile.py
│   │   └── evaluation_model.py
│   ├── ingestion/
│   │   ├── questionnaire.py
│   │   ├── indicators.py
│   │   ├── experts.py
│   │   ├── vehicles.py
│   │   └── environment.py
│   ├── segmentation/
│   │   ├── preprocessing.py
│   │   ├── kmeans.py
│   │   └── profiling.py
│   ├── weighting/
│   │   ├── ahp.py
│   │   ├── expert_aggregation.py
│   │   ├── entropy.py
│   │   ├── history_adjustment.py
│   │   └── fusion.py
│   ├── personalization/
│   │   ├── feature_classifier.py
│   │   ├── profile_adapter.py
│   │   └── environment_adapter.py
│   ├── optimization/
│   │   ├── genetic.py
│   │   ├── constraints.py
│   │   └── objective.py
│   ├── services/
│   │   ├── generation_service.py
│   │   ├── optimization_service.py
│   │   └── scoring_service.py
│   ├── reporting/
│   │   ├── excel_exporter.py
│   │   ├── json_exporter.py
│   │   └── charts.py
│   ├── api/
│   │   ├── app.py
│   │   ├── routes_runs.py
│   │   └── routes_downloads.py
│   └── cli.py
├── templates/
│   └── index.html
├── static/
│   ├── app.css
│   └── app.js
└── tests/
    ├── unit/
    ├── integration/
    ├── fixtures/
    └── golden/                    # 小规模固定输入/期望输出
```

## 5. 核心数据模型

### 5.1 问卷数据 `questionnaire.xlsx`

一行代表一位匿名用户。

必需字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `sample_id` | string | 匿名样本 ID，唯一 |
| `gender` | category | 性别，可配置枚举 |
| `age_group` | category | 年龄段 |
| `education` | category | 学历 |
| `city_tier` | category | 城市/地区层级 |
| `family_stage` | category | 婚育/家庭阶段 |
| `purchase_budget` | category | 购车预算 |
| `ownership_status` | category | 购车与拥车情况 |
| `function_*` | integer | 31 项功能需求评分 |

评分方向必须由配置声明：

```yaml
questionnaire:
  rating_min: 1
  rating_max: 5
  lower_score_means_stronger_need: true
  unknown_values: [0, null, "不了解"]
  max_unknown_ratio: 0.20
```

若 `lower_score_means_stronger_need=true`，则 1、2 表示强需求。任何算法不得隐式假设评分方向。

### 5.2 指标体系 `indicators.xlsx`

| 字段 | 必需 | 说明 |
| --- | --- | --- |
| `indicator_id` | 是 | 稳定 ID，不使用名称作为主键 |
| `level_1`～`level_4` | 是 | 分层路径，无下级时留空 |
| `feature_name` | 否 | 对应的座舱功能名称 |
| `category` | 是 | `required`、`bonus`、`excluded` |
| `installation_rate` | 否 | 功能装机率，0～1 |
| `score_max` | 是 | 指标满分 |
| `enabled` | 是 | 是否启用 |

每个非叶节点的直接子节点权重之和必须为 1。加分项可设置独立权重池，默认总占比 `0.10`，由配置控制。

### 5.3 专家判断

推荐使用长表 `expert_pairwise.xlsx`：

| 字段 | 说明 |
| --- | --- |
| `expert_id` | 匿名专家编号 |
| `parent_indicator_id` | 同层比较所属父节点 |
| `left_indicator_id` | 左侧指标 |
| `right_indicator_id` | 右侧指标 |
| `ratio` | Saaty 1/9～9 标度 |

专家权威度可用直接系数或另一份 AHP 判断矩阵输入。系统必须保留专家原始判断、CR、一致性状态和聚合结果。

### 5.4 车型历史测评 `vehicle_scores.xlsx`

格式：`vehicle_id`、车型元数据、各叶子指标得分。缺失值不能自动填 0，应按配置选择删除、插补或标记不可用。

### 5.5 权重优化数据 `optimization_samples.xlsx`

| 字段 | 说明 |
| --- | --- |
| `sample_id` | 车型或测评批次 ID |
| `indicator_score_*` | 叶子指标得分 |
| `target_score` | 经确认的综合真值/用户主观评分 |
| `split` | `train`、`validation`、`test` |

禁止只用一条总分真值优化全部权重。没有至少两批数据时，允许演示算法，但结果必须标记为“未完成泛化验证”。

## 6. 算法设计

### 6.1 问卷清洗

处理步骤：

1. 校验列名、评分范围、唯一 ID 和人口属性枚举。
2. 计算每行未知功能比例，超过 20% 的样本剔除并记录原因。
3. 对少量缺失值按功能中位数填补，填补行为写入运行报告。
4. 去除完全重复样本；疑似低质量样本只标记，不凭提交时间臆断。
5. 输出 `clean_questionnaire.parquet` 和 `cleaning_report.json`。

验收：若使用原材料，应得到 676 条有效样本；若结果不同，报告差异原因而不是强制凑数。

### 6.2 K-means++ 用户分群

输入仅使用 31 项需求评分，人口属性用于画像解释，不默认参与距离计算。

```yaml
segmentation:
  mode: manual              # manual | auto
  manual_k: 5               # 需求书口径；可改为 6
  candidate_k: [2, 3, 4, 5, 6, 7, 8, 9, 10]
  random_seed: 42
  n_init: 50
  min_cluster_ratio: 0.08
```

自动选 K 时同时计算：

- SSE/inertia：只用于观察边际下降，不取最小值。
- Silhouette score：越高越好。
- Calinski-Harabasz score：越高越好。
- 最小簇占比和跨随机种子稳定性。

选择规则：先排除最小簇占比不足的 K，再从候选中选择轮廓系数较高且肘部合理的最小 K。最终 K、指标曲线和人工覆盖理由写入 `segmentation_decision.json`。

### 6.3 用户画像与需求分级

对每个簇计算：

- 样本数和占比。
- 31 项功能的均值、中位数、强需求比例。
- 人口属性分布及相对总体的差异。
- Top-N 差异化功能。

兼容原项目的需求分级逻辑，阈值配置如下：

```yaml
need_tiers:
  strong: 0.90
  high: 0.70
  normal: 0.50
```

当“评分 1 或 2”的占比不低于 0.90，标记强需求；0.70～0.90 为较高需求；0.50～0.70 为一般需求；低于 0.50 为低需求。命名仅生成 `cluster_1` 等中性标签，业务名称通过 `profile_names.yaml` 人工配置。

### 6.4 AHP 与专家聚合

对指标树每个父节点分别计算：

1. 校验判断矩阵正互反性、对角线为 1、值域为 1/9～9。
2. 使用几何平均法或最大特征向量法计算局部权重。
3. 计算 `CI=(lambda_max-n)/(n-1)`、`CR=CI/RI`。
4. 智慧座舱默认要求 `CR < 0.10`；不通过时拒绝该专家该层结果，并输出需要复核的比较项。
5. 使用专家权威度 `q_e` 聚合：`w_expert = sum(q_e * w_e)`，再归一化。

不得自动静默修改专家判断矩阵。

### 6.5 客观权重与历史数据调整

熵权为可选模块：先处理正向、负向和区间型指标，再进行 Min-Max 标准化，计算信息熵与差异系数。常数列权重为 0，并记录警告。

历史分布模块兼容原原型：识别常态、双峰、偏高、偏低等分布。原脚本仅对“常态分布”乘以 1.05；新实现把倍率放入配置，不硬编码：

```yaml
history_adjustment:
  enabled: true
  factors:
    normal: 1.05
    bimodal: 1.00
    high: 1.00
    low: 1.00
```

### 6.6 多源权重融合

由于现有材料没有唯一、可复核的主客观融合公式，系统支持两种模式：

1. `expert_only`：复现现有主流程，以专家聚合权重为基础。
2. `linear_fusion`：`w_base = alpha * w_expert + (1-alpha) * w_entropy`。

`alpha` 必须写入运行配置和报告，默认使用 `expert_only`，避免无证据地选择融合比例。

### 6.7 用户差异化体系生成

以基础指标树为模板，对每个用户群体深拷贝后处理：

- 强需求功能权重乘 `1.10`。
- 较高需求功能权重乘 `1.05`。
- 一般需求保持不变。
- 低需求功能是否剔除由 `remove_low_need` 配置控制。
- 明确列为必测的安全/基础功能不得因用户低需求被删除。
- 装机率不低于阈值的功能可提升为必测项，原材料阈值为 30%。
- 每次修改后只在同一父节点内归一化，不能跨层归一化。

输出每类用户的权重变化表：`indicator_id`、基础权重、需求等级、倍率、最终局部权重、全局权重、变更原因。

### 6.8 环境影响模块

环境因素作为可选上下文，例如噪声、光照、温度和网络状态。规则使用 YAML 声明：

```yaml
rules:
  - context: {noise_level: high}
    indicator_id: voice_recognition_noise
    action: multiply_weight
    value: 1.10
```

环境模块只能调整配置允许的指标，不得删除必测项。默认关闭，直至客户确认规则与验证方法。

### 6.9 遗传算法权重自趋优

目标是在保持权重可解释和层级约束的前提下，降低预测综合分与真值的误差。

推荐目标函数：

```text
loss = MAE(predicted_score, target_score)
     + lambda_l2 * ||w - w_base||²
     + lambda_rank * ranking_loss
```

约束：

- 每组兄弟指标权重非负且和为 1。
- 每个权重在 `[max(0,w0-epsilon), min(1,w0+epsilon)]` 内，默认 `epsilon=0.10`。
- 必测项不得被优化为 0。
- 优化后必须重新计算并断言所有层级约束。

建议参数：种群 50、迭代 100、变异率 0.10、固定随机种子。交叉和变异后使用“带上下界的单纯形投影”，不能只做简单归一化，否则可能破坏 `±epsilon` 约束。

模型选择只看验证集；测试集在最终冻结权重后评估一次。报告训练、验证、测试 MAE，以及相对基础权重的变化。

## 7. 运行编排

### 7.1 生成任务

`GenerationService.run()`：

1. 创建 UUID `run_id` 和独立目录。
2. 保存上传文件副本及 SHA-256。
3. 校验所有输入表。
4. 执行问卷清洗、分群和画像。
5. 构建指标树并计算专家/AHP 权重。
6. 按配置执行熵权、历史分布和环境调整。
7. 生成各用户群体的差异化指标树。
8. 导出 Excel、JSON、图表、运行清单和警告。

### 7.2 优化任务

`OptimizationService.run()`：

1. 读取某个已完成的 `generation_run_id`。
2. 校验优化样本与指标 ID 完整匹配。
3. 按训练/验证/测试集执行遗传优化。
4. 保存基础权重、候选权重、冻结权重及指标。
5. 生成新模型版本，不覆盖旧结果。

## 8. API 设计

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| `POST` | `/api/v1/runs/generation` | 创建体系生成任务，multipart 上传文件和配置 |
| `POST` | `/api/v1/runs/{run_id}/optimization` | 基于指定生成任务优化权重 |
| `GET` | `/api/v1/runs/{run_id}` | 查询状态、警告和产物清单 |
| `GET` | `/api/v1/runs/{run_id}/profiles` | 查询用户画像 |
| `GET` | `/api/v1/runs/{run_id}/weights` | 查询基础和差异化权重 |
| `GET` | `/api/v1/runs/{run_id}/artifacts/{name}` | 下载结果文件 |
| `POST` | `/api/v1/models/{model_id}/score` | 使用冻结体系计算某车型得分 |
| `GET` | `/health` | 存活检查 |

任务状态统一为 `pending/running/succeeded/failed`。失败响应包含稳定错误码、用户可读信息和日志定位 ID，不返回服务器堆栈。

## 9. 输出文件

每次生成任务至少输出：

```text
run_manifest.json
cleaning_report.json
segmentation_metrics.xlsx
user_profiles.xlsx
base_weights.xlsx
personalized_systems.xlsx
weight_change_log.xlsx
model.json
charts/elbow.png
charts/cluster_sizes.png
charts/profile_heatmap.png
```

`personalized_systems.xlsx` 包含：总览、数据质量、聚类选择、基础权重、每个用户群体、变更日志等工作表。工作表名称必须截断并去重，符合 Excel 31 字符限制。

## 10. Web 页面

第一版只需三个视图：

1. **生成体系：** 上传问卷、指标、专家、车型、用户需求和配置文件。
2. **优化权重：** 选择已有模型，上传带真值的优化数据。
3. **运行结果：** 显示数据质量、K 值依据、用户画像、权重变化、警告和下载按钮。

前端不得把所有文件写入固定文件名；每个任务使用独立 `run_id`，避免并发覆盖。

## 11. 安全、审计与可复现性

- 上传文件只允许 `.xlsx/.csv/.yaml`，同时校验 MIME、大小和内容结构。
- 文件名由服务端生成，禁止路径穿越。
- 不使用 `eval/exec`，不加载 Excel 宏。
- 问卷数据默认保留 30 天后清理，配置可调整。
- 每次运行保存代码版本、依赖版本、随机种子、配置、输入哈希和输出哈希。
- 日志不得打印完整问卷行和专家敏感信息。

## 12. 测试设计

### 12.1 单元测试

- 问卷评分方向、未知比例和重复样本处理。
- K 值选择不允许退化为 `argmin(SSE)`。
- 用户画像需求阈值边界：0.50、0.70、0.90。
- AHP 正互反矩阵、权重和、CI/CR 及不一致拒绝。
- 熵权常数列、负向/区间型指标处理。
- 指标树父子关系、局部/全局权重和。
- 用户倍率与必测项保护。
- 遗传算法优化后上下界及权重和约束。
- Excel 导出工作表名、公式与数值一致性。

### 12.2 集成测试

- 使用 30 条问卷、2 位专家、6 个指标、4 款车型的固定数据跑完整生成流程。
- 对同一输入和随机种子，两次输出数值完全一致。
- 并发执行两个任务，文件和结果互不覆盖。
- 错误输入返回结构化错误，不留下半成品模型。

### 12.3 回归测试

建立脱敏黄金数据集，固定以下产物：有效样本数、选定 K、每簇规模、AHP 权重、各用户体系权重和评分结果。算法修改必须显式更新黄金结果并说明原因。

## 13. 验收标准

- 能使用模板数据从零完成一次体系生成和一次权重优化。
- 问卷清洗过程可追溯，原材料输入时能够解释是否得到 676 个有效样本。
- 自动 K 选择输出至少三种聚类指标；手动 K 会在报告中标记为业务覆盖。
- 所有 AHP 判断都有 CR 结果；不一致判断不会静默进入最终权重。
- 每个父节点的子权重和误差小于 `1e-8`。
- 遗传优化后所有权重满足非负、和为 1、相对基础权重变化不超过配置边界。
- 不同用户群体能够输出不同但可解释的指标体系。
- 同输入、同配置、同随机种子可复现同一结果。
- API、CLI、Web 三种入口调用同一服务层，不复制算法逻辑。
- 单元测试覆盖核心算法，整体覆盖率建议不低于 85%。

## 14. 实施顺序

### Phase 1：最小闭环

完成数据校验、指标树、AHP、用户画像读取、差异化体系和 Excel 导出。先使用人工确认的用户画像，不做聚类和遗传算法。

### Phase 2：用户分群

加入问卷清洗、K-means++、K 值评估、画像生成及图表。确认五类或六类最终业务口径。

### Phase 3：多源权重

加入专家权威度、车型历史分布和可选熵权，完成消融对比。

### Phase 4：自趋优

加入带上下界投影的遗传算法、数据切分和优化前后报告。

### Phase 5：Web 与工程化

加入 FastAPI、页面、任务隔离、审计、Docker 和完整回归测试。

## 15. Claude 实现约束

把本文交给 Claude 时附加以下指令：

1. 严格按 Phase 分批实现，每个 Phase 完成后运行测试并输出结果。
2. 不从遗留脚本复制 `eval`、导入即执行、固定文件名和当前目录写入等行为。
3. 所有算法函数保持纯函数或显式依赖，API 层不得包含算法逻辑。
4. 先生成模板和脱敏演示数据，再实现真实文件适配。
5. 对五类/六类口径不做猜测，使用配置并在 README 中说明。
6. 对材料没有给出明确公式的权重融合，默认使用 `expert_only`，不得自行声称复现原结果。
7. 每次提交同时更新测试、README 和一份可运行命令示例。

## 16. 待确认参数

- 最终验收采用五类还是六类用户，以及画像名称与聚类簇的映射关系。
- 31 项问卷评分中，1～5 的确切语义及“不了解”编码。
- 专家输入是原始判断矩阵还是指标打分后转换的矩阵。
- 熵权与专家权重的融合公式及启用范围。
- 历史数据对权重的具体调整倍率。
- 环境因素的正式规则、测试场景和是否进入最终交付。
- 遗传算法使用的综合分真值来源与训练/验证批次划分。
