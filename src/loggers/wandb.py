"""W&B logger.

Uses the convention defined in src.loggers.base.derive_run_meta. Configures
``calib/k_minutes`` as a step metric so K-min calibration curves plot in W&B
without extra work.
"""
from __future__ import annotations

from typing import Any

from src.core.types import RunMeta
from src.loggers.base import LoggerBase


class WandbLogger(LoggerBase):
    def __init__(
        self,
        project: str = "etm",
        entity: str | None = None,
        mode: str = "online",  # "online" | "offline" | "disabled"
        dir: str | None = None,
    ):
        self.project = project
        self.entity = entity
        self.mode = mode
        self.dir = dir
        self._run = None

    def init_run(self, run_meta: RunMeta) -> None:
        import wandb  # local import so the dep is optional during dev

        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            group=run_meta.group,
            job_type=run_meta.job_type,
            tags=run_meta.tags,
            name=run_meta.name,
            config=run_meta.config_dict,
            mode=self.mode,
            dir=self.dir,
            reinit=True,
        )
        # Plot K-min curves naturally: eval/* metrics use calib/k_minutes as x-axis when present.
        wandb.define_metric("calib/k_minutes")
        wandb.define_metric("eval/*", step_metric="calib/k_minutes")
        # Online curves use online/step.
        wandb.define_metric("online/step")
        wandb.define_metric("online/*", step_metric="online/step")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        import wandb

        if self._run is None:
            return
        wandb.log(metrics, step=step)

    def log_summary(self, summary: dict[str, Any]) -> None:
        if self._run is None:
            return
        for k, v in summary.items():
            self._run.summary[k] = v

    def finish(self) -> None:
        import wandb

        if self._run is not None:
            wandb.finish()
            self._run = None
