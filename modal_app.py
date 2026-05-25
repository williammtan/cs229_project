"""Run etm benchmarks on Modal (https://modal.com).

The existing scripts are reused unchanged; this file is the only addition.

One-time setup:
    uv add modal
    uv run modal token new
    uv run modal secret create wandb WANDB_API_KEY=<key>
    uv run modal secret create huggingface HF_TOKEN=<token>   # for REVE only

Common commands:
    # populate the data volume (subjects default: 1..12)
    uv run modal run modal_app.py::download_data

    # full LOSO sweep, FANNED OUT in parallel (one A100 per experiment)
    uv run modal run --detach modal_app.py::run_benchmarks

    # ... or serially in one container (cheaper, slower)
    uv run modal run --detach modal_app.py::run_benchmarks_serial

    # run only one experiment
    uv run modal run modal_app.py::run_experiment --name baseline_eegnet_loso

    # pull results back
    uv run modal volume get etm-results / ./modal-results
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "etm-benchmarks"
REPO = Path(__file__).parent
REMOTE_REPO = "/root/etm"
DEFAULT_GPU = "A100"
MAX_CONCURRENT_GPUS = 10  # Modal account GPU concurrency limit.

# Reuse the experiment list from the existing runner script — single source of truth.
# Locally REPO is the project dir; inside a Modal container, modal_app.py lives at
# /root/modal_app.py while the repo is copied to REMOTE_REPO by add_local_dir.
_runner_script = REPO / "scripts" / "run_full_baselines.py"
if not _runner_script.exists():
    _runner_script = Path(REMOTE_REPO) / "scripts" / "run_full_baselines.py"
_spec = importlib.util.spec_from_file_location("_run_full_baselines", _runner_script)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_run_full_baselines"] = _mod
_spec.loader.exec_module(_mod)
EXPERIMENTS: list[str] = list(_mod.EXPERIMENTS)
KMIN_EXPERIMENTS: list[str] = list(_mod.KMIN_EXPERIMENTS)

# torch wheels on linux PyPI bundle CUDA, so plain pip_install_from_pyproject
# gives a GPU-capable build when the function requests a gpu.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_pyproject(str(REPO / "pyproject.toml"))
    .add_local_dir(
        str(REPO),
        remote_path=REMOTE_REPO,
        ignore=[
            "wandb/*",
            "results/*",
            "data/*",
            "outputs/*",
            ".venv/*",
            ".git/*",
            "**/__pycache__/*",
            "*.pyc",
        ],
    )
)

data_vol = modal.Volume.from_name("etm-data", create_if_missing=True)
results_vol = modal.Volume.from_name("etm-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("etm-hf-cache", create_if_missing=True)

VOLUMES = {
    f"{REMOTE_REPO}/data": data_vol,
    f"{REMOTE_REPO}/results": results_vol,
    "/root/.cache/huggingface": hf_cache_vol,
}

SECRETS = [
    modal.Secret.from_name("wandb"),
    modal.Secret.from_name("huggingface"),
]

app = modal.App(APP_NAME, image=image)


def _sh(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REMOTE_REPO, check=True)


@app.function(volumes=VOLUMES, timeout=60 * 60)
def download_data(subjects: str = "1 2 3 4 5 6 7 8 9 10 11 12") -> None:
    """Download WAY-EEG-GAL subjects into the persistent data volume."""
    _sh(["python", "scripts/download_data.py", *subjects.split()])
    data_vol.commit()


@app.function(
    gpu=DEFAULT_GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=6 * 60 * 60,
    max_containers=MAX_CONCURRENT_GPUS,
)
def run_experiment(name: str, overrides: str = "", logger: str = "wandb") -> str:
    """Run a single Hydra experiment by name. Returns the experiment name."""
    cmd = ["python", "-m", "src.runner", f"+experiment={name}", f"logger={logger}"]
    if overrides:
        cmd += overrides.split()
    _sh(cmd)
    results_vol.commit()
    return name


def _fanout(label: str, names: list[str], logger: str) -> None:
    """Shared fanout: run_experiment.map across the given experiment names."""
    print(f"[{label}] launching {len(names)} experiments on {DEFAULT_GPU}:")
    for n in names:
        print(f"  - {n}")

    done: list[str] = []
    failed: list[str] = []
    for result in run_experiment.map(
        names,
        kwargs={"logger": logger},
        return_exceptions=True,
        order_outputs=False,
    ):
        if isinstance(result, Exception):
            print(f"[{label}]  FAILED: {result!r}")
            failed.append(repr(result))
        else:
            print(f"[{label}]  done: {result}")
            done.append(result)

    print()
    print(f"[{label}] finished: {len(done)} ok, {len(failed)} failed")
    if failed:
        for f in failed:
            print(f"  - {f}")
        raise RuntimeError(f"{len(failed)} experiments failed")


@app.function(
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=12 * 60 * 60,
)
def run_benchmarks(
    logger: str = "wandb",
    only: str = "",
    skip: str = "",
) -> None:
    """LOSO benchmark grid (incl. RA variants), fanned out one GPU per experiment.

    `only`/`skip` are space-separated experiment names, mirroring the
    --only/--skip flags on scripts/run_full_baselines.py.
    """
    names = only.split() if only else list(EXPERIMENTS)
    skip_set = set(skip.split())
    names = [n for n in names if n not in skip_set]
    _fanout("loso", names, logger)


@app.function(
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=12 * 60 * 60,
)
def run_kmin_sweep(
    logger: str = "wandb",
    only: str = "",
    skip: str = "",
) -> None:
    """k-min calibration-budget sweep, fanned out one GPU per experiment.

    Separate entry point from run_benchmarks because the kmin protocol answers a
    different research question (calibration efficiency curve, not zero-shot LOSO).
    """
    names = only.split() if only else list(KMIN_EXPERIMENTS)
    skip_set = set(skip.split())
    names = [n for n in names if n not in skip_set]
    _fanout("kmin", names, logger)


@app.function(
    gpu=DEFAULT_GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=24 * 60 * 60,
)
def run_benchmarks_serial(logger: str = "wandb", extra: str = "") -> None:
    """Single-container serial fallback: wraps scripts/run_full_baselines.py.

    Cheaper (one GPU, not N) but proportionally slower wall-clock.
    `extra` forwards args to the script, e.g. extra="--only baseline_eegnet_loso".
    """
    cmd = ["python", "scripts/run_full_baselines.py", "--logger", logger]
    if extra:
        cmd += extra.split()
    _sh(cmd)
    results_vol.commit()


@app.local_entrypoint()
def main() -> None:
    """Default `modal run modal_app.py` → fan out the full sweep on A100s."""
    run_benchmarks.remote()
