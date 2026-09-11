"""配置管理模块."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """应用配置类."""

    def __init__(self, config_path: str | Path | None = None):
        self._config_dir = Path(__file__).parent.parent.parent / "configs"
        self._data_dir = Path(__file__).parent.parent.parent / "data"
        self._runs_dir = self._data_dir / "runs"

        if config_path:
            with open(config_path) as f:
                self._cfg: dict[str, Any] = yaml.safe_load(f)
        else:
            default_path = self._config_dir / "default.yaml"
            with open(default_path) as f:
                self._cfg = yaml.safe_load(f)

        self._runs_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val = self._cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    @property
    def questionnaire_config(self) -> dict[str, Any]:
        return self._cfg.get("questionnaire", {})

    @property
    def segmentation_config(self) -> dict[str, Any]:
        return self._cfg.get("segmentation", {})

    @property
    def need_tiers(self) -> dict[str, float]:
        return self._cfg.get("need_tiers", {})

    @property
    def weight_fusion_mode(self) -> str:
        return self._cfg.get("weight_fusion", {}).get("mode", "expert_only")

    @property
    def entropy_enabled(self) -> bool:
        return self._cfg.get("entropy_weight", {}).get("enabled", False)

    @property
    def history_adjustment_enabled(self) -> bool:
        return self._cfg.get("history_adjustment", {}).get("enabled", False)

    @property
    def personalization_config(self) -> dict[str, Any]:
        return self._cfg.get("personalization", {})

    @property
    def genetic_config(self) -> dict[str, Any]:
        return self._cfg.get("genetic_algorithm", {})

    @property
    def ahp_cr_threshold(self) -> float:
        return self._cfg.get("ahp", {}).get("cr_threshold", 0.10)

    @property
    def runs_dir(self) -> Path:
        return self._runs_dir

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @staticmethod
    def file_hash(filepath: Path) -> str:
        """计算文件SHA-256哈希."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def load_feature_taxonomy(self) -> dict[str, Any]:
        path = self._config_dir / "feature_taxonomy.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    def load_environment_rules(self) -> dict[str, Any]:
        path = self._config_dir / "environment_rules.yaml"
        with open(path) as f:
            return yaml.safe_load(f)


def load_config(config_path: str | Path | None = None) -> Config:
    """加载配置的工厂函数."""
    return Config(config_path)
