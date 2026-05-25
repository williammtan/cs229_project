"""JSON-on-disk logger. Used by CI and quick sanity runs without W&B."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.core.types import RunMeta
from src.loggers.base import LoggerBase


class OfflineLogger(LoggerBase):
    def __init__(self, root: str = "results/offline_runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._run_dir: Path | None = None
        self._history: list[dict[str, Any]] = []
        self._summary: dict[str, Any] = {}

    def init_run(self, run_meta: RunMeta) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        self._run_dir = self.root / f"{ts}__{run_meta.name}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "meta.json").write_text(json.dumps({
            "group": run_meta.group,
            "job_type": run_meta.job_type,
            "tags": run_meta.tags,
            "name": run_meta.name,
            "config": run_meta.config_dict,
        }, indent=2, default=str))
        self._history = []
        self._summary = {}

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run_dir is None:
            return
        row = {"step": step, **metrics}
        self._history.append(row)
        (self._run_dir / "history.jsonl").open("a").write(json.dumps(row, default=str) + "\n")

    def log_summary(self, summary: dict[str, Any]) -> None:
        if self._run_dir is None:
            return
        self._summary.update(summary)
        (self._run_dir / "summary.json").write_text(json.dumps(self._summary, indent=2, default=str))

    def finish(self) -> None:
        self._run_dir = None
