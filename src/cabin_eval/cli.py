"""命令行接口."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cabin_eval.config import load_config
from cabin_eval.services.generation_service import GenerationService
from cabin_eval.services.optimization_service import OptimizationService


def main():
    parser = argparse.ArgumentParser(
        description="智慧座舱测评体系自趋优模型"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    gen_parser = subparsers.add_parser("generate", help="生成测评体系")
    gen_parser.add_argument("--questionnaire", required=True, help="问卷数据路径")
    gen_parser.add_argument("--indicators", required=True, help="指标体系路径")
    gen_parser.add_argument("--experts", help="专家判断矩阵路径")
    gen_parser.add_argument("--authority", help="专家权威度路径")
    gen_parser.add_argument("--vehicles", help="车型历史数据路径")
    gen_parser.add_argument("--config", help="配置文件路径")

    opt_parser = subparsers.add_parser("optimize", help="优化权重")
    opt_parser.add_argument("--base-run", required=True, help="基础运行ID")
    opt_parser.add_argument("--samples", required=True, help="优化样本路径")
    opt_parser.add_argument("--config", help="配置文件路径")

    serve_parser = subparsers.add_parser("serve", help="启动Web服务")
    serve_parser.add_argument("--host", default="0.0.0.0", help="服务地址")
    serve_parser.add_argument("--port", type=int, default=8000, help="服务端口")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        cfg = load_config(args.config) if args.config else load_config()
        service = GenerationService(cfg)

        result = service.run(
            questionnaire_path=args.questionnaire,
            indicators_path=args.indicators,
            expert_judgments_path=args.experts,
            expert_authority_path=args.authority,
            vehicle_scores_path=args.vehicles,
        )

        print(f"Generation completed: run_id={result['run_id']}")
        print(f"Output directory: {result['run_dir']}")

    elif args.command == "optimize":
        cfg = load_config(args.config) if args.config else load_config()
        service = OptimizationService(cfg)

        result = service.run(
            base_run_id=args.base_run,
            optimization_samples_path=args.samples,
        )

        print(f"Optimization completed: run_id={result['run_id']}")
        print(f"Train MAE: {result.get('train_mae')}")
        print(f"Validation MAE: {result.get('val_mae')}")
        print(f"Test MAE: {result.get('test_mae')}")

    elif args.command == "serve":
        import uvicorn
        from cabin_eval.api.app import app

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
