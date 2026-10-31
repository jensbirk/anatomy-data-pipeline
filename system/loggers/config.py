from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLogger:
    def __init__(self, config_path: Path | str, log_dir: Path | str):
        self.config_path = Path(config_path)
        self.log_dir = Path(log_dir)
        self.snapshot_path = self.log_dir / self.config_path.name
        self.metadata_path = self.log_dir / "config_metadata.yaml"

    def save(self, extra_metadata: Optional[Dict[str, Any]] = None) -> Path:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.config_path, self.snapshot_path)

        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_config": str(self.config_path.resolve()),
            "snapshot_path": str(self.snapshot_path.resolve()),
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                metadata["config"] = yaml.safe_load(f)
        except Exception as exc:
            metadata["config_error"] = f"Failed to parse YAML: {exc}"

        with self.metadata_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)

        return self.snapshot_path
