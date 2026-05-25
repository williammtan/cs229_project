"""CPU latency profiling for inference + adaptation.

Stubs only — hooked up once the FM backbones land. Hard constraint from
docs/plan.md: end-to-end inference + adaptation must stay under 200 ms p95
on consumer CPU.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


@dataclass
class LatencySamples:
    name: str
    samples_ms: list[float]

    @property
    def p50_ms(self) -> float:
        return float(np.quantile(self.samples_ms, 0.50)) if self.samples_ms else float("nan")

    @property
    def p95_ms(self) -> float:
        return float(np.quantile(self.samples_ms, 0.95)) if self.samples_ms else float("nan")

    def to_dict(self, prefix: str = "latency") -> dict[str, float]:
        return {
            f"{prefix}/{self.name}/p50_ms": self.p50_ms,
            f"{prefix}/{self.name}/p95_ms": self.p95_ms,
            f"{prefix}/{self.name}/n": float(len(self.samples_ms)),
        }


@contextmanager
def time_block(samples: LatencySamples):
    t0 = time.perf_counter()
    yield
    samples.samples_ms.append((time.perf_counter() - t0) * 1000.0)
