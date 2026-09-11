# 智慧座舱测评体系自趋优模型

## 项目定位

本项目面向智慧座舱测评和决策支持场景，目标是把问卷需求、专家判断、车型历史得分和环境上下文组织成一套可配置的测评体系。

它解决的不是车机端控制问题，而是“如何定义测什么、不同用户群体应关注什么、指标权重如何解释和迭代”的测评建模问题。

目标业务闭环：

问卷导入 -> 数据清洗与质量报告 -> 用户分群与画像 -> 指标树和专家权重 -> 可选的熵权/历史分布调整 -> 面向用户群体的差异化体系 -> 带约束的权重优化 -> Excel、JSON、图表和 Web 查询

## 主要功能

### 1. 问卷导入与清洗

支持 CSV 和 XLSX 问卷导入。问卷以匿名 sample_id 标识样本，功能评分列使用 function_* 前缀。

清洗流程设计包括：

- 校验必需列、评分范围、样本 ID 和功能评分列。
- 根据配置识别 0、空值和“不了解”等未知编码。
- 按未知比例剔除低质量样本，并记录剔除原因。
- 对少量缺失值使用功能中位数填补，并保存填补记录。
- 去除重复样本 ID。
- 记录评分方向，例如分数越低表示需求越强。

输出应包含清洗后的数据和 cleaning_report，便于解释有效样本数与原始材料的差异。

### 2. K-means++ 用户分群

分群默认只使用功能需求评分，不把性别、年龄、预算等人口属性直接放入距离计算。人口属性用于聚类完成后的画像解释。

系统支持手动和自动两种 K 值模式。手动模式由业务人员指定五类或六类等口径；自动模式计算 SSE、Silhouette、Calinski-Harabasz、最小簇占比等指标，供规则和人工共同决策。

分群结果包括簇规模、占比、功能均值/中位数、强需求比例、人口属性分布和差异化功能列表。聚类标签保持 cluster_1 等中性形式，业务名称应由业务人员确认后配置。

### 3. 分层指标树

指标体系通过 indicator_id 和 level_1 到 level_4 表达层级关系，并支持：

- 必测项、加分项和排除项分类。
- 功能装机率、指标满分和启用状态。
- 父子节点局部权重与叶子指标全局权重。
- 同一父节点内的权重归一化和约束校验。

指标 ID 是稳定主键，不能使用展示名称作为主键。缺失值不能默认填成 0，应由配置决定删除、插补或标记不可用。

### 4. 专家 AHP 权重

专家输入采用长表形式，记录专家、父节点、左右指标和 Saaty 比较值。AHP 模块设计支持正互反矩阵校验、几何平均法计算局部权重、lambda_max/CI/CR 一致性检验、不一致判断复核和按专家权威度聚合。

默认要求 CR 小于 0.10。没有通过一致性检查的结果不应进入最终权重。

### 5. 客观权重、历史数据和多源融合

系统预留三类可选调整：

- 熵权：对正向、负向和区间型指标进行标准化，使用数据差异计算客观权重；常数列应记录为 0 权重并产生警告。
- 历史分布：识别常态、双峰、偏高、偏低等分布，倍率由 YAML 配置控制。
- 多源融合：支持 expert_only 和 linear_fusion，融合比例 alpha 必须进入运行配置和报告。

由于不同材料对融合公式的证据不一致，默认模式应保持 expert_only，不能把未经确认的融合比例描述成业务结论。

### 6. 差异化测评体系

系统以基础指标树为模板，为每个用户群体生成独立体系：

- 强需求功能提高局部权重。
- 较高需求功能使用较小倍率调整。
- 低需求项可按配置移除。
- 必测安全/基础功能不能因用户低需求被删除。
- 达到装机率阈值的功能可提升为必测项。
- 每次修改后只在同一父节点内重新归一化。

每个用户群体都应输出权重变更日志，包括指标 ID、基础权重、需求等级、倍率、最终局部/全局权重和变更原因。

### 7. 环境上下文适配

环境模块用于表达噪声、光照、温度、网络状态等上下文对指标关注度的影响。规则通过 YAML 声明，例如在高噪声环境下提高语音识别相关指标权重。

该模块默认关闭。环境规则必须经过客户确认和场景验证，且只能调整允许的指标，不能删除必测项。

### 8. 遗传算法权重优化

优化目标是在保持指标树解释性和层级约束的前提下，使综合分更接近已确认的真值或用户主观评分。

优化数据需要包含 sample_id、各指标得分、target_score 和 train/validation/test 切分。优化后应保存基础权重、候选权重、冻结权重、训练/验证/测试误差和权重变化，并生成新模型版本，不覆盖历史模型。

### 9. 报告和产物

每个运行目录应隔离保存运行清单、配置快照、输入哈希、清洗报告、K 值候选指标、选择依据、用户画像、基础权重、个性化权重、Excel、JSON、图表、警告和产物清单。

Excel 工作表名称需要符合 31 字符限制并避免重名。JSON 用于机器读取和 API 返回，Excel 用于业务人员审阅。

## 当前实现状态

仓库已经搭建了上述功能对应的模块、配置、CLI、Web 页面和 Docker 文件，但当前版本仍是未完成的原型，不能把所有设计能力视为已验收能力。

已具备的基础：

- Python 包和分层目录。
- Pydantic 数据模型与 YAML 配置。
- AHP、K-means 指标计算、熵权、历史调整、图表和基础导出代码。
- CLI、FastAPI 页面、演示数据和单元测试骨架。

当前主要阻断：

- 问卷清洗存在确定性异常。
- FastAPI 导入阶段找不到静态目录。
- 生成服务存在未导入符号。
- 演示模板扩展名与文件内容不一致。
- 优化服务没有读取生成任务的基础模型。
- 指标树、个性化权重和专家权威度聚合尚未形成正确闭环。

详细证据和修复顺序见 [工程完整度检查报告.md](工程完整度检查报告.md)。

## 技术栈

- Python 3.11+
- pandas、NumPy、openpyxl：数据处理和 Excel 读写
- scikit-learn、SciPy：聚类、统计和优化基础能力
- Pydantic 2：输入模型和数据契约
- FastAPI、Uvicorn：Web API
- Jinja2、原生 HTML/CSS/JavaScript：轻量页面
- matplotlib、seaborn：图表
- pytest：测试
- Docker、docker compose：部署入口

## 项目结构

smart-cabin/
├── configs/                    # 默认配置、功能分类、环境规则
├── data/
│   ├── templates/              # 导入模板
│   ├── demo/                   # 脱敏演示数据
│   └── runs/                   # 每次运行的隔离输出
├── src/cabin_eval/
│   ├── config.py               # 配置加载和文件哈希
│   ├── schemas.py              # Pydantic 数据模型
│   ├── domain/                 # 指标树、用户画像、测评模型
│   ├── ingestion/              # 问卷、指标、专家、车型、环境导入
│   ├── segmentation/           # 预处理、K-means、画像
│   ├── weighting/              # AHP、熵权、历史分布、融合
│   ├── personalization/        # 需求和环境适配
│   ├── optimization/           # 遗传优化和约束
│   ├── services/               # 生成、优化、评分服务
│   ├── reporting/              # Excel、JSON、图表
│   └── api/                    # FastAPI 应用和路由
├── templates/                  # Web 页面模板
├── static/                     # 页面脚本和样式
└── tests/                      # 单元、集成和黄金数据测试

## 数据文件

仓库内的演示文件当前是 CSV：

data/demo/questionnaire_demo.csv  
data/demo/indicators_demo.csv  
data/demo/expert_pairwise_demo.csv  
data/demo/authority_demo.csv  
data/demo/optimization_samples_demo.csv

核心字段约定详见 [design_one.md](design_one.md) 第 5 节。生产数据应使用匿名样本、专家和车型 ID，不应直接保存姓名、电话等身份信息。

## 安装

    pip install -e .

建议使用 Python 3.11，并在开发环境中额外安装测试和代码质量工具：

    pip install -e '.[dev]'

## CLI 入口

生成测评体系：

    python -m cabin_eval.cli generate \
      --questionnaire data/demo/questionnaire_demo.csv \
      --indicators data/demo/indicators_demo.csv

使用已有生成任务优化权重：

    python -m cabin_eval.cli optimize \
      --base-run <run_id> \
      --samples data/demo/optimization_samples_demo.csv

启动 Web 服务：

    python -m cabin_eval.cli serve --host 0.0.0.0 --port 8000

说明：以上命令描述的是目标入口。当前仓库仍存在报告中列出的阻断问题，生成和优化命令在修复前不能视为可成功跑通。

## API 入口

目标 API 包括：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | /health | 健康检查 |
| POST | /api/v1/runs/generation | 上传数据并生成测评体系 |
| POST | /api/v1/runs/{run_id}/optimization | 基于生成任务优化权重 |
| GET | /api/v1/runs/{run_id} | 查询运行状态 |
| GET | /api/v1/runs/{run_id}/profiles | 查询用户画像 |
| GET | /api/v1/runs/{run_id}/weights | 查询权重和变更 |
| GET | /api/v1/runs/{run_id}/artifacts/{name} | 下载白名单产物 |
| POST | /api/v1/models/{model_id}/score | 使用冻结模型评分 |

当前代码只实现了其中一部分，完整接口需要在工程化阶段补齐。

## 测试和质量检查

    pytest -q
    ruff check src tests
    mypy src/cabin_eval

测试重点应覆盖数据清洗边界、K 值选择、AHP 一致性、指标树约束、个性化权重、熵权常数列、遗传优化边界、导出文件和并发运行隔离。

## Docker

    docker compose up --build

容器化部署前需要先修复 Dockerfile 中的源码复制顺序，并验证容器内模板、配置、运行目录和静态资源路径。

## 设计依据

完整的重建设计、数据模型、算法约束、运行编排、API 设计、输出要求和待确认参数见 [design_one.md](design_one.md)。
