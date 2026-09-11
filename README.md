# 智慧座舱测评体系自趋优模型

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目简介

面向中汽中心智慧座舱测评场景的**可配置、可迭代**测评体系系统。融合用户问卷分析、专家权重判断和遗传算法优化，实现面向不同用户群体的差异化指标权重自动生成。

核心解决：如何定义测评指标、如何为不同用户群体分群建模、如何让权重迭代可解释可追溯。

## 技术亮点

| 能力 | 技术实现 |
|------|---------|
| **用户智能分群** | K-means++ 聚类 + 多指标 (SSE/Silhouette/CH) 辅助决策 |
| **专家权重融合** | AHP 层次分析法 + Saaty 标度 + CR 一致性检验 + 权威度加权 |
| **差异化权重** | 基于需求强度的权重倍率调整，支持必测项保护 |
| **约束优化** | 遗传算法 + 单纯形投影 + 层级约束保持 |
| **工程化** | Pydantic 数据校验、FastAPI RESTful API、Docker 容器化 |

## 系统架构

```
问卷导入 → 数据清洗 → 用户分群 → 指标树构建
                                        ↓
         Excel/JSON/图表输出 ← 个性化权重生成
                                        ↓
                              遗传算法约束优化
```

## 核心模块

| 模块 | 路径 | 说明 |
|------|------|------|
| `ingestion` | [src/cabin_eval/ingestion/](src/cabin_eval/ingestion/) | 问卷/指标/专家/车型数据导入与校验 |
| `segmentation` | [src/cabin_eval/segmentation/](src/cabin_eval/segmentation/) | K-means++ 用户分群与画像生成 |
| `weighting` | [src/cabin_eval/weighting/](src/cabin_eval/weighting/) | AHP 权重计算、熵权法、历史分布调整 |
| `personalization` | [src/cabin_eval/personalization/](src/cabin_eval/personalization/) | 差异化测评体系生成 |
| `optimization` | [src/cabin_eval/optimization/](src/cabin_eval/optimization/) | 遗传算法带约束权重优化 |
| `api` | [src/cabin_eval/api/](src/cabin_eval/api/) | FastAPI Web 服务与文件上传下载 |

## 快速开始

```bash
# 安装依赖
pip install -e .

# CLI 生成测评体系
python -m cabin_eval.cli generate \
    --questionnaire data/demo/questionnaire_demo.csv \
    --indicators data/demo/indicators_demo.csv

# 启动 Web 服务
python -m cabin_eval.cli serve --host 0.0.0.0 --port 8000
```

## API 端点

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/health` | 服务健康检查 |
| `POST` | `/api/v1/runs/generation` | 上传数据生成测评体系 |
| `GET` | `/api/v1/runs/{run_id}` | 查询运行状态与产物 |
| `POST` | `/api/v1/runs/{run_id}/optimization` | 基于已有体系优化权重 |
| `GET` | `/api/v1/runs/{run_id}/artifacts/{name}` | 下载 Excel/JSON/图表 |

## 技术栈

- **语言**: Python 3.11+
- **数据处理**: pandas, NumPy, openpyxl
- **机器学习**: scikit-learn, SciPy
- **数据校验**: Pydantic 2
- **Web 框架**: FastAPI, Uvicorn, Jinja2
- **图表**: matplotlib
- **测试**: pytest, pytest-cov
- **部署**: Docker, docker-compose

## 项目结构

```
smart-cabin/
├── configs/              # YAML 配置文件
├── data/
│   ├── demo/             # 演示数据集
│   └── runs/             # 运行输出目录
├── src/cabin_eval/
│   ├── domain/           # 领域模型 (指标树、用户画像)
│   ├── ingestion/        # 多源数据导入
│   ├── segmentation/     # K-means 聚类分群
│   ├── weighting/        # AHP/熵权/融合权重
│   ├── personalization/  # 差异化体系生成
│   ├── optimization/     # 遗传算法优化
│   ├── services/         # 业务服务层
│   ├── reporting/        # Excel/JSON/图表导出
│   └── api/              # FastAPI 接口
├── templates/            # Web 页面模板
├── static/               # CSS/JS 静态资源
└── tests/                # 单元与集成测试
```

## 测试覆盖

```bash
# 运行全部测试
pytest tests/ -v

# 测试覆盖率
pytest tests/ --cov=cabin_eval --cov-report=html
```

---

**设计依据**: 详见 [design_one.md](design_one.md)
