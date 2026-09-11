# 智慧座舱测评体系自趋优模型

## 项目概述

本项目实现智慧座舱测评体系的自趋优模型，支持用户分群、指标权重自动优化和差异化测评体系生成。

## 技术栈

- Python 3.11+
- pandas, NumPy, openpyxl - 数据处理
- scikit-learn, SciPy - 聚类与指标计算
- Pydantic 2 - 数据校验
- FastAPI, Uvicorn - Web API
- Jinja2 - 模板渲染
- matplotlib, seaborn - 图表生成
- pytest - 测试

## 快速开始

### 安装依赖

```bash
pip install -e .
```

### 使用CLI

```bash
# 生成测评体系
python -m cabin_eval.cli generate \
    --questionnaire data/demo/questionnaire_demo.xlsx \
    --indicators data/demo/indicators_demo.xlsx

# 优化权重
python -m cabin_eval.cli optimize \
    --base-run <run_id> \
    --samples data/demo/optimization_samples.xlsx

# 启动Web服务
python -m cabin_eval.cli serve --port 8000
```

### 使用API

```bash
uvicorn cabin_eval.api.app:app --reload --port 8000
```

访问 http://localhost:8000 查看Web界面。

## 项目结构

```
smart-cabin/
├── configs/              # 配置文件
├── data/
│   ├── templates/        # 导入模板
│   ├── demo/             # 演示数据
│   └── runs/             # 运行输出
├── src/cabin_eval/
│   ├── config.py         # 配置管理
│   ├── schemas.py        # 数据模型
│   ├── domain/           # 领域模型
│   ├── ingestion/        # 数据导入
│   ├── segmentation/     # 用户分群
│   ├── weighting/        # 权重计算
│   ├── personalization/  # 个性化适配
│   ├── optimization/     # 遗传算法优化
│   ├── services/         # 服务层
│   ├── reporting/        # 报告导出
│   └── api/              # Web API
├── templates/            # HTML模板
├── static/               # 静态资源
└── tests/                # 测试
```

## 配置说明

主配置文件 `configs/default.yaml` 包含:

- `questionnaire`: 问卷评分配置
- `segmentation`: 分群参数
- `need_tiers`: 需求等级阈值
- `weight_fusion`: 权重融合模式
- `personalization`: 个性化配置
- `genetic_algorithm`: 遗传算法参数

## 运行流程

1. **问卷导入与清洗**: 加载问卷数据，校验评分范围，处理缺失值
2. **用户分群**: K-means++聚类，生成用户画像
3. **指标权重计算**: AHP层次分析法计算专家权重
4. **差异化体系生成**: 根据用户画像生成个性化指标权重
5. **结果导出**: Excel/JSON/图表输出

## 测试

```bash
pytest tests/ -v
```
