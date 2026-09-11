"""优化服务."""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cabin_eval.config import Config
from cabin_eval.optimization.genetic import GeneticOptimizer
from cabin_eval.schemas import RunStatus


class OptimizationService:
    """权重优化服务."""

    def __init__(self, config: Config):
        self._cfg = config

    def run(
        self,
        base_run_id: str,
        optimization_samples_path: str | Path,
    ) -> dict[str, Any]:
        """执行权重优化."""
        run_id = str(uuid.uuid4())[:8]
        run_dir = self._cfg.runs_dir / f"optimization_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        samples_path = Path(optimization_samples_path)
        if samples_path.suffix == ".csv":
            samples_df = pd.read_csv(samples_path)
        elif samples_path.suffix in (".xlsx", ".xls"):
            samples_df = pd.read_excel(samples_path)
        else:
            raise ValueError(f"Unsupported file format: {samples_path.suffix}")

        if "sample_id" not in samples_df.columns or "target_score" not in samples_df.columns:
            raise ValueError("Missing required columns in optimization samples")

        train_df = samples_df[samples_df.get("split", "train") == "train"]
        val_df = samples_df[samples_df.get("split", "validation") == "validation"]
        test_df = samples_df[samples_df.get("split", "test") == "test"]

        indicator_cols = [c for c in samples_df.columns if c.startswith("indicator_score_")]

        X_train = train_df[indicator_cols].values
        y_train = train_df["target_score"].values

        X_val = val_df[indicator_cols].values
        y_val = val_df["target_score"].values

        X_test = test_df[indicator_cols].values
        y_test = test_df["target_score"].values

        base_weights = np.ones(len(indicator_cols)) / len(indicator_cols)

        optimizer = GeneticOptimizer(self._cfg.genetic_config)

        sibling_constraints = [list(range(len(indicator_cols)))]

        result = optimizer.optimize(
            base_weights=base_weights,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sibling_constraints=sibling_constraints,
        )

        test_mae = None
        if len(X_test) > 0 and result["optimized_weights"] is not None:
            predictions = X_test @ result["optimized_weights"]
            test_mae = float(np.mean(np.abs(predictions - y_test)))

        return {
            "run_id": run_id,
            "base_run_id": base_run_id,
            "status": "succeeded",
            "optimized_weights": result["optimized_weights"].tolist() if result["optimized_weights"] is not None else None,
            "train_mae": result.get("best_train_mae"),
            "val_mae": result.get("best_val_mae"),
            "test_mae": test_mae,
            "fitness_history": result.get("fitness_history", []),
        }
