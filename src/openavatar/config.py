"""Configuration loading. No path or model id is hardcoded in the pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.json"


class ConfigError(RuntimeError):
    pass


@dataclass
class Config:
    data: dict[str, Any]
    path: Path
    root: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path or os.environ.get("OPENAVATAR_CONFIG") or DEFAULT_CONFIG).resolve()
        if not p.exists():
            raise ConfigError(f"config not found: {p}")
        data = json.loads(p.read_text())
        return cls(data=data, path=p, root=p.parent.parent)

    # -- accessors --------------------------------------------------------
    def resolve(self, key: str) -> Path:
        """Resolve a config path key relative to the repository root."""
        raw = self.data.get(key)
        if raw is None:
            raise ConfigError(f"config key {key!r} is not set")
        p = Path(raw)
        return p if p.is_absolute() else (self.root / p)

    def model(self, key: str) -> dict[str, Any]:
        models = self.data.get("models", {})
        if key not in models:
            raise ConfigError(
                f"unknown model_key {key!r}; configured models: {sorted(models)}"
            )
        return {**models[key], "model_key": key}

    @property
    def model_keys(self) -> list[str]:
        return sorted(self.data.get("models", {}))

    @property
    def metrics(self) -> dict[str, Any]:
        return self.data.get("metrics", {})

    @property
    def validation(self) -> dict[str, Any]:
        return self.data.get("validation", {})
